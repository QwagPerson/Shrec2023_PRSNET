import os

from lightning.pytorch.callbacks import EarlyStopping

from dataset.voxel_dataset import VoxelDataset
from torch.utils.data import DataLoader, random_split
from model.prsnet.lighting_prsnet import LightingPRSNet
import lightning as L
import torch

dataset = VoxelDataset("/data/gsanteli/voxel_dataset")






L.seed_everything(42)
generator = torch.Generator().manual_seed(42)

train_dataset, val_dataset = random_split(dataset, [0.9, 0.1], generator=generator)

train_loader = DataLoader(train_dataset, collate_fn=dataset.collate_fn, num_workers=3)
val_loader = DataLoader(val_dataset, collate_fn=dataset.collate_fn, num_workers=3)

# model
prsnet = LightingPRSNet(
    amount_of_heads=1,
    out_features=4,
    reg_coef=5,
)

# train model
trainer = L.Trainer(
    enable_checkpointing=True,
    fast_dev_run=False,
    callbacks=[
        EarlyStopping(monitor="val_loss", mode="min")
    ],
)
trainer.fit(model=prsnet, train_dataloaders=train_loader, val_dataloaders=val_loader)
