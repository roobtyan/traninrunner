import os
import pickle
import numpy as np
import torch.distributed as dist

from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


class CIFAR100Python(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.transform = transform
        self.data_dir = os.path.join(root, "cifar-100-python")

        if train:
            self.file_path = os.path.join(self.data_dir, "train")
        else:
            self.file_path = os.path.join(self.data_dir, "test")

        with open(self.file_path, "rb") as f:
            entry = pickle.load(f, encoding="latin1")

        self.data = entry["data"]
        self.targets = entry["fine_labels"]

        self.data = self.data.reshape(-1, 3, 32, 32)  # N, C, H, W
        self.data = self.data.transpose((0, 2, 3, 1))  # N, H, W, C

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img) # N, H, W, C

        if self.transform:
            img = self.transform(img)

        return img, target


def build_cifar100_dataloader(
    data_root="train/cifar-10/data/",
    is_ddp=False,
    batch_size=32,
    num_workers=4,
    train=True,
    transform=None,
):
    dataset = CIFAR100Python(root=data_root, train=train, transform=transform)

    sampler = None
    if is_ddp:
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=train,
            drop_last=train,
        )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=train,
    )

    return dataloader, sampler
