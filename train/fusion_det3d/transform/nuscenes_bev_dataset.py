import os
import numpy as np
import torch
import torch.distributed as dist


from torch.utils.data import Dataset
from PIL import Image
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion
from torch.utils.data.dataset import ConcatDataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


class NuScenesBEVDataset(Dataset):
    def __init__(
        self,
        nusc: NuScenes,
        data_root: str,
        split='train',
        queue_length=4,
        img_size=(256, 704),
        num_cams=6,
        use_lidar=True,
        class_names=None,
    ):
        super().__init__()

        self.nusc = nusc
        self.data_root = data_root
        self.queue_length = queue_length
        self.img_size = img_size
        self.use_lidar = use_lidar

        self.class_names = class_names or [
            'car', 'truck', 'bus', 'trailer',
            'construction_vehicle',
            'pedestrian', 'motorcycle', 'bicycle'
        ]

        self.cam_names = [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
        ]

        self.sample_tokens = self._build_sample_list(split)

    def _build_sample_list(self, split):
        samples = []

        split_scenes = create_splits_scenes()[split]

        for scene in self.nusc.scene:
            if scene['name'] not in split_scenes:
                continue

            token = scene['first_sample_token']
            while token:
                samples.append(token)
                sample = self.nusc.get('sample', token)
                token = sample['next']

        assert len(samples) > 0, f"No samples found for split={split}"
        return samples

    
    def __len__(self):
        return len(self.sample_tokens)
    
    def _get_sample_queue(self, idx):
        curr_token = self.sample_tokens[idx]
        curr_sample = self.nusc.get('sample', curr_token)
        
        queue = [curr_sample] # 放入当前帧
        
        temp_sample = curr_sample
        for _ in range(self.queue_length - 1):
            if temp_sample['prev']:
                temp_sample = self.nusc.get('sample', temp_sample['prev'])
                queue.append(temp_sample)
            else:
                queue.append(queue[-1])
        
        return queue[::-1] # 反转，顺序变为：[T-3, T-2, T-1, Current]
    
    def __getitem__(self, idx):
        sample_queue = self._get_sample_queue(idx)

        imgs_queue = []
        metas_queue = []

        for sample in sample_queue:
            imgs, metas = self._load_multiview_data(sample)
            imgs_queue.append(imgs)
            metas_queue.append(metas)

        imgs_queue = torch.stack(imgs_queue)  # (T, N, C, H, W)
        gt_boxes, gt_labels = self._load_annotations(sample_queue[-1])

        data = {
            'imgs': imgs_queue,
            'metas': metas_queue,
            'gt_boxes': gt_boxes,
            'gt_labels': gt_labels,
        }

        if self.use_lidar:
            lidar_points = self._load_lidar_data(sample_queue[-1])
            data['lidar_points'] = lidar_points

        return data
    
    def _load_multiview_data(self, sample):
        imgs, intrinsics, extrinsics = [], [], []

        for cam in self.cam_names:
            sd = self.nusc.get('sample_data', sample['data'][cam])
            cs = self.nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
            pose = self.nusc.get('ego_pose', sd['ego_pose_token'])

            img = Image.open(os.path.join(self.data_root, sd['filename'])).convert('RGB')
            img = img.resize(self.img_size)
            img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.
            imgs.append(img)

            intrinsics.append(torch.tensor(cs['camera_intrinsic'], dtype=torch.float32))

            cam2ego = transform_matrix(cs['translation'], Quaternion(cs['rotation']), inverse=False)
            ego2global = transform_matrix(pose['translation'], Quaternion(pose['rotation']), inverse=False)

            extrinsics.append(torch.from_numpy(ego2global @ cam2ego).float())

        return torch.stack(imgs), {
            'intrinsics': torch.stack(intrinsics),
            'extrinsics': torch.stack(extrinsics),
        }

    
    def _load_lidar_data(self, sample):
        sd = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        pc_path = os.path.join(self.data_root, sd['filename'])

        lidar_pc = LidarPointCloud.from_file(pc_path)
        points = torch.from_numpy(lidar_pc.points.T).float() # (N, 4)
        return points
    
    def _load_annotations(self, sample):
        boxes = []
        labels = []

        for ann_token in sample['anns']:
            ann = self.nusc.get('sample_annotation', ann_token)
            name = ann['category_name'].split('.')[0]
            if name not in self.class_names:
                continue

            center = ann['translation']
            size = ann['size']
            yaw = Quaternion(ann['rotation']).yaw_pitch_roll[0]

            boxes.append([
                center[0], center[1], center[2],
                size[0], size[1], size[2],
                yaw
            ])

            labels.append(self.class_names.index(name))

        if len(boxes) == 0:
            boxes = torch.zeros((0, 7), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        return boxes, labels
    
    
def custom_collate_fn(batch):
    data = {}
    data['imgs'] = torch.stack([b['imgs'] for b in batch])
    
    # 变长数据用 list 存储
    if 'lidar_points' in batch[0]:
        data['lidar_points'] = [b['lidar_points'] for b in batch]
    
    data['gt_boxes'] = [b['gt_boxes'] for b in batch]
    data['gt_labels'] = [b['gt_labels'] for b in batch]
    
    # img_metas 列表
    data['metas'] = [b['metas'] for b in batch]
    
    return data


def build_nuscenes_bev_dataloader(
    data_root: str,
    version: str = 'v1.0-trainval',
    split: str = 'train',
    queue_length: int = 4,
    img_size=(256, 704),
    num_cams: int = 6,
    use_lidar: bool = True,
    class_names=None,
    is_ddp: bool = False,
    batch_size: int = 2,
    num_workers: int = 4,
):
    nusc = NuScenes(version=version, dataroot=data_root, verbose=False)
    dataset = NuScenesBEVDataset(
        nusc=nusc,
        data_root=data_root,
        split=split,
        queue_length=queue_length,
        img_size=img_size,
        num_cams=num_cams,
        use_lidar=use_lidar,
        class_names=class_names,
    )

    sampler = None
    if is_ddp:
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=(split == 'train'),
        )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None and split == 'train'),
        num_workers=num_workers,
        sampler=sampler,
        pin_memory=True,
        drop_last=(split == 'train'),
        collate_fn=custom_collate_fn 
    )

    return dataloader, sampler