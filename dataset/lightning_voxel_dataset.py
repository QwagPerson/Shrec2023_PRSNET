import random

import lightning as L
import torch

from torch.utils.data import random_split, DataLoader
from dataset.voxel_dataset import VoxelDataset


class VoxelDataModule(L.LightningDataModule):
    def __init__(self,
                 train_data_path: str = "/path/to/train_data",
                 test_data_path: str = "/path/to/test_data",
                 train_val_split: float = 0.9,
                 sample_size: int = 1024,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 seed: int = None,
                 n_workers: int = 1,
                 ):
        super().__init__()
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.batch_size = batch_size
        self.train_val_split = train_val_split
        self.sample_size = sample_size
        self.shuffle = shuffle
        self.seed = seed if seed is not None else random.randint(0, 100)
        self.n_workers = n_workers

        self.save_hyperparameters()

    def setup(self, stage: str):
        if stage == "fit":
            dataset_full = VoxelDataset(
                dataset_root=self.train_data_path,
                sample_size=self.sample_size
            )

            proportions = [self.train_val_split, 1 - self.train_val_split]
            lengths = [int(p * len(dataset_full)) for p in proportions]
            lengths[-1] = len(dataset_full) - sum(lengths[:-1])

            self.voxel_train, self.voxel_val = random_split(
                dataset_full, lengths, generator=torch.Generator().manual_seed(self.seed)
            )

        if stage == "test":
            self.voxel_test = VoxelDataset(
                dataset_root=self.test_data_path,
                sample_size=self.sample_size
            )

    def train_dataloader(self):
        return DataLoader(
            self.voxel_train,
            collate_fn=self.voxel_train.collate_fn,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.n_workers,
            generator=torch.Generator().manual_seed(self.seed)
        )

    def val_dataloader(self):
        return DataLoader(
            self.voxel_val,
            collate_fn=self.voxel_train.collate_fn,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.n_workers,
            generator=torch.Generator().manual_seed(self.seed)
        )

    def test_dataloader(self):
        return DataLoader(
            self.voxel_test,
            collate_fn=self.voxel_train.collate_fn,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.n_workers,
            generator=torch.Generator().manual_seed(self.seed)
        )
