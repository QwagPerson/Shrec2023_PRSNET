from lightning.pytorch.cli import LightningCLI
from model.prsnet.lightning_prsnet import LightingPRSNet
from dataset.lightning_voxel_dataset import VoxelDataModule
from lightning.pytorch.callbacks import EarlyStopping


def cli_main():
    cli = LightningCLI(LightingPRSNet, VoxelDataModule)


if __name__ == "__main__":
    cli_main()
