import os
from lightning import Trainer
from dataset.voxel_dataset import VoxelDataset
from torch.utils.data import DataLoader
import numpy as np
import pathlib

from model.prsnet.lightning_prsnet import LightingPRSNet
from model.prsnet.metrics import transform_representation
from argparse import ArgumentParser


# pred is a tensor of shape 1 x N x 4
def save_prediction(idx, pred, path):
    n_heads = pred.shape[1]
    pred = transform_representation(pred)  # returns 1 x N x 7
    pred = pred.squeeze().numpy().reshape((n_heads, 7))
    with open(os.path.join(path, f"points{idx}_res.txt"), "w") as f:
        f.write(str(pred.shape[0]))
        f.write("\n")
        np.savetxt(f, pred)


def save_predictions(preds, path):
    if not os.path.exists(path):
        os.mkdir(path)
    for idx, pred in enumerate(preds):
        save_prediction(idx, pred, path)


parser = ArgumentParser()
parser.add_argument("--data_path", type=pathlib.Path, required=True)
parser.add_argument("--output_path", type=pathlib.Path, required=True)
parser.add_argument("--model_path", type=pathlib.Path, required=True)
parser.add_argument("--n_workers", type=int, required=False, default=4)

if __name__ == "__main__":
    args = vars(parser.parse_args())
    DATA_PATH = args["data_path"]
    OUTPUT_PATH = args["output_path"]
    MODEL_PATH = args["model_path"]
    N_WORKERS = args["n_workers"]

    model = LightingPRSNet.load_from_checkpoint(MODEL_PATH)

    dataset = VoxelDataset(DATA_PATH, sample_size=model.sample_size)
    print(len(dataset))

    dataloader = DataLoader(dataset, collate_fn=dataset.collate_fn, num_workers=N_WORKERS, batch_size=1, shuffle=False)

    trainer = Trainer(
        accelerator="gpu",
        devices=[1]
    )
    predictions = trainer.predict(model, dataloader)
    save_predictions(predictions, OUTPUT_PATH)
