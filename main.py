from lightning.pytorch.cli import LightningCLI
from Shrec2023_PRSNET.src.model.prsnet import LightingPRSNet
from Shrec2023_PRSNET.src.dataset import VoxelDataModule
import torch


def cli_main():
    torch.set_float32_matmul_precision('high')
    cli = LightningCLI(LightingPRSNet, VoxelDataModule)


if __name__ == "__main__":
    cli_main()
