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
        imgs, intrinsics, cam2egos, ego2globals = [], [], [], []
        img_paths = []

        for cam in self.cam_names:
            sd = self.nusc.get('sample_data', sample['data'][cam])
            cs = self.nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
            pose = self.nusc.get('ego_pose', sd['ego_pose_token'])

            img_path = os.path.join(self.data_root, sd['filename'])
            img = Image.open(img_path).convert('RGB')
            orig_w, orig_h = img.size
            img_h, img_w = self.img_size
            img = img.resize((img_w, img_h))
            new_w, new_h = img.size
            sx = new_w / orig_w
            sy = new_h / orig_h
            img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img = (img - mean) / std
            imgs.append(img)
            img_paths.append(img_path)

            # 相机内参
            intrin = torch.eye(4)
            intrin[:3, :3] = torch.tensor(cs['camera_intrinsic'], dtype=torch.float32)
            intrin[0, 0] *= sx
            intrin[0, 2] *= sx
            intrin[1, 1] *= sy
            intrin[1, 2] *= sy
            intrinsics.append(intrin)

            # 外参：cam to ego
            cam2ego = transform_matrix(cs['translation'], Quaternion(cs['rotation']), inverse=False)
            cam2egos.append(torch.from_numpy(cam2ego).float())

            # 位姿：ego to global
            ego2global = transform_matrix(pose['translation'], Quaternion(pose['rotation']), inverse=False)
            ego2globals.append(torch.from_numpy(ego2global).float())

        return torch.stack(imgs), {
            'sample_token': sample['token'],
            'img_paths': img_paths,
            'intrinsics': torch.stack(intrinsics),      # (N, 4, 4)
            'cam2egos': torch.stack(cam2egos),          # (N, 4, 4)
            'ego2globals': torch.stack(ego2globals),    # (N, 4, 4)
        }

    
    def _load_lidar_data(self, sample):
        sd = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        cs = self.nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
        lidar2ego = transform_matrix(cs['translation'], Quaternion(cs['rotation']), inverse=False)
        lidar2ego = torch.from_numpy(lidar2ego).float()
        
        pc_path = os.path.join(self.data_root, sd['filename'])
        lidar_pc = LidarPointCloud.from_file(pc_path)
        points = torch.from_numpy(lidar_pc.points.T).float() # (N, 4)

        # 坐标变换
        point_xyz = points[:, :3]  # (N, 3)
        point_xyz_hom = torch.cat([point_xyz, torch.ones((point_xyz[:, :1].shape), dtype=torch.float32)], dim=1)  # (N, 4)
        point_xyz_ego = (lidar2ego @ point_xyz_hom.T).T # (N, 4)
        points[:, :3] = point_xyz_ego[:, :3]

        return points
    
    def _load_annotations(self, sample):
        boxes = []
        labels = []

        sd = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        pose = self.nusc.get('ego_pose', sd['ego_pose_token'])
        global_to_ego = transform_matrix(pose['translation'], Quaternion(pose['rotation']), inverse=True)
        global_to_ego_rot = Quaternion(pose['rotation']).inverse

        for ann_token in sample['anns']:
            ann = self.nusc.get('sample_annotation', ann_token)
            name = ann['category_name'].split('.')[0]
            if name not in self.class_names:
                continue

            size = ann['size']
            center = np.array(ann['translation'] + [1.0], dtype=np.float32)
            center_ego = global_to_ego @ center
            box_rot = Quaternion(ann['rotation'])
            box_rot_ego = global_to_ego_rot * box_rot
            yaw = box_rot_ego.yaw_pitch_roll[0]

            vel = self.nusc.box_velocity(ann_token)
            vx, vy = 0.0, 0.0
            if vel is not None:
                vel = np.array(vel, dtype=np.float32)
                if not np.isnan(vel).any():
                    vel_ego = global_to_ego_rot.rotate(vel)
                    vx, vy = float(vel_ego[0]), float(vel_ego[1])

            boxes.append([
                center_ego[0], center_ego[1], center_ego[2],
                size[0], size[1], size[2],
                yaw, vx, vy
            ])

            labels.append(self.class_names.index(name))

        if len(boxes) == 0:
            boxes = torch.zeros((0, 9), dtype=torch.float32)
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
