import pathlib
import random
import os

import lightning as L
import torch

from lightning.pytorch.callbacks import EarlyStopping
from argparse import ArgumentParser
from dataset.voxel_dataset import VoxelDataset
from torch.utils.data import DataLoader, random_split

from model.prsnet.lightning_prsnet import LightingPRSNet

parser = ArgumentParser()

parser.add_argument("--experiment_name", type=str, required=True)
parser.add_argument("--data_path", type=pathlib.Path, required=True)
parser.add_argument("--log_folder", type=pathlib.Path, required=True)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--sample_size", type=int, default=1024)
parser.add_argument("--seed", required=False, type=int, default=random.randint(0, 100), help="Seed used.")
parser.add_argument("--train_val_split", required=False, default=0.9, type=float)
parser.add_argument("--n_workers", required=False, type=int,
                    default=1, help="Amount of workers transforming the dataset.")

parser.add_argument("--input_res", type=int, required=True)
parser.add_argument("--amount_of_heads", required=False, type=int, default=4)
parser.add_argument("--out_features", required=False, type=int, default=4)
parser.add_argument("--use_bn", required=True, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument("--loss", required=True, type=str)
parser.add_argument("--reg_coef", required=False, type=int, default=0)
parser.add_argument("--patience", required=False, type=int, default=3)

if __name__ == "__main__":
    args = vars(parser.parse_args())

    NAME = args["experiment_name"]
    DATA_PATH = args["data_path"]
    LOG_FOLDER = args["log_folder"]
    SAMPLE_SIZE = args["sample_size"]
    BATCH_SIZE = args["batch_size"]
    SEED = args["seed"]
    TRAIN_VAL_SPLIT = args["train_val_split"]
    N_WORKERS = args["n_workers"]
    INPUT_RES = args["input_res"]
    N_HEADS = args["amount_of_heads"]
    OUT_FEATURES = args["out_features"]
    USE_BN = args["use_bn"]
    LOSS_USED = args["loss"]
    REG_COEF = args["reg_coef"]
    PATIENCE = args["patience"]

    dataset = VoxelDataset(DATA_PATH, sample_size=SAMPLE_SIZE)

    L.seed_everything(SEED)
    generator = torch.Generator().manual_seed(SEED)

    torch.set_float32_matmul_precision('medium')

    proportions = [TRAIN_VAL_SPLIT, 1 - TRAIN_VAL_SPLIT]
    lengths = [int(p * len(dataset)) for p in proportions]
    lengths[-1] = len(dataset) - sum(lengths[:-1])

    train_dataset, val_dataset = random_split(dataset, lengths, generator=generator)

    train_loader = DataLoader(train_dataset, collate_fn=dataset.collate_fn, num_workers=N_WORKERS,
                              batch_size=BATCH_SIZE)
    val_loader = DataLoader(val_dataset, collate_fn=dataset.collate_fn, num_workers=N_WORKERS, batch_size=BATCH_SIZE)

    lighting_model = LightingPRSNet(
        name=NAME,
        batch_size=BATCH_SIZE,
        input_resolution=INPUT_RES,
        amount_of_heads=N_HEADS,
        out_features=OUT_FEATURES,
        use_bn=USE_BN,
        loss_used=LOSS_USED,
        reg_coef=REG_COEF,
        sample_size=SAMPLE_SIZE,
    )

    trainer = L.Trainer(
        enable_checkpointing=True,
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min", patience=PATIENCE)
        ],
        default_root_dir=os.path.join(os.getcwd(), LOG_FOLDER),
        fast_dev_run=False,
        accelerator="gpu",
        devices=[1]
    )

    trainer.fit(model=lighting_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
