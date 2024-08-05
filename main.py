from lightning.pytorch.cli import LightningCLI
from src.model.prsnet.lightning_prsnet import LightingPRSNet
from src.dataset.SymDataset.SymDataModule import SymDataModule
import torch


def cli_main():
    torch.set_float32_matmul_precision('high')
    cli = LightningCLI(LightingPRSNet, SymDataModule)


if __name__ == "__main__":
    cli_main()
