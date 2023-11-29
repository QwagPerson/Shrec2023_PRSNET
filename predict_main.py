import os
from lightning import Trainer
from dataset.voxel_dataset import VoxelDataset
from torch.utils.data import DataLoader
import numpy as np

from model.prsnet.lightning_prsnet import LightingPRSNet
from model.prsnet.metrics import transform_representation


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


dataset = VoxelDataset("/data/voxel_dataset", sample_size=1024)
dataloader = DataLoader(dataset, collate_fn=dataset.collate_fn, num_workers=3, batch_size=1, shuffle=False)

model = LightingPRSNet.load_from_checkpoint(
    "modelos_interesantes/remote_test/lightning_logs/version_21/checkpoints/epoch=28-step=3074.ckpt",
    sample_size=1)

# fun idx, tensor -> write file points_pred.txt
# will use totxt from numpy so the algorithm goes tensor->array->file
trainer = Trainer()
predictions = trainer.predict(model, dataloader)
save_predictions(predictions, "./test_pred")
