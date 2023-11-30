import os
from lightning import Trainer
import numpy as np
import pathlib

from model.prsnet.lightning_prsnet import LightingPRSNet
from dataset.lightning_voxel_dataset import VoxelDataModule
from model.prsnet.metrics import transform_representation
from argparse import ArgumentParser


def save_prediction(pred, path):
    idx, y_out, sample_points_out, y_pred, sample_points, y_true, y_true_out = pred
    idx = idx.item()
    n_heads = y_out.shape[1]
    pred = y_out.squeeze().numpy().reshape((n_heads, 7))
    with open(os.path.join(path, f"points{idx}_res.txt"), "w") as f:
        f.write(str(idx))
        f.write("\n")
        np.savetxt(f, pred)


def save_predictions(prediction_list, path):
    if not os.path.exists(path):
        os.mkdir(path)
    for pred in prediction_list:
        save_prediction(pred, path)


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
    data_module = VoxelDataModule(
        test_data_path=DATA_PATH,
        train_val_split=1,
        batch_size=1
    )

    trainer = Trainer()

    predictions = trainer.predict(model, data_module)
    save_predictions(predictions, OUTPUT_PATH)
