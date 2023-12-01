import random

import lightning as L
import torch

from torch.utils.data import random_split, DataLoader
from dataset.voxel_dataset import VoxelDataset, collate_fn


class VoxelDataModule(L.LightningDataModule):
    def __init__(self,
                 train_data_path: str = "/path/to/train_data",
                 test_data_path: str = "/path/to/test_data",
                 predict_data_path: str = "/path/to/predict_data",
                 train_val_split: float = 0.9,
                 sample_size: int = 1024,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 n_workers: int = 1,
                 ):
        super().__init__()
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.predict_data_path = predict_data_path
        self.batch_size = batch_size
        self.train_val_split = train_val_split
        self.sample_size = sample_size
        self.shuffle = shuffle
        self.n_workers = n_workers

        self.save_hyperparameters()

    def setup(self, stage: str):
        if stage == "fit":
            dataset_full = VoxelDataset(
                dataset_root=self.train_data_path,
                sample_size=self.sample_size,
            )

            proportions = [self.train_val_split, 1 - self.train_val_split]
            lengths = [int(p * len(dataset_full)) for p in proportions]
            lengths[-1] = len(dataset_full) - sum(lengths[:-1])

            self.voxel_train, self.voxel_val = random_split(
                dataset_full, lengths
            )

        if stage == "test":
            self.voxel_test = VoxelDataset(
                dataset_root=self.test_data_path,
                sample_size=self.sample_size
            )

        if stage == "predict":
            self.voxel_predict = VoxelDataset(
                dataset_root=self.predict_data_path,
                sample_size=self.sample_size,
                is_predict_dataset=True
            )

    def train_dataloader(self):
        return DataLoader(
            self.voxel_train,
            collate_fn=collate_fn,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.n_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.voxel_val,
            collate_fn=collate_fn,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.n_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.voxel_test,
            collate_fn=collate_fn,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.n_workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.voxel_predict,
            collate_fn=collate_fn,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.n_workers,
        )
