import pathlib
import random
import os

import lightning as L
import torch

from lightning.pytorch.callbacks import EarlyStopping
from argparse import ArgumentParser

import polyscope as ps
from dataset.voxel_dataset import VoxelDataset
from torch.utils.data import DataLoader, random_split

from model.prsnet.lightning_prsnet import LightingPRSNet
from model.prsnet.losses import ChamferLoss, batch_apply_symmetry, apply_symmetry
from setup.setup_voxel_dataset.symmetry_plane import SymmetryPlane
from model.prsnet.metrics import get_phc, transform_representation


def visualize_prediction(predicted_planes_4, predicted_planes_6, real_planes, points):
    """
    :param predicted_planes_4: N x 4
    :param predicted_planes_6: N x 6
    :param real_planes: M x 6
    :param points: S x 3
    :return: None
    """
    # Create symmetryPlane Objs
    original_symmetries = [
        SymmetryPlane(
            point=real_planes[idx, 3::].detach().numpy(),
            normal=real_planes[idx, 0:3].detach().numpy()
        )
        for idx in range(real_planes.shape[0])
    ]

    predicted_symmetries = [
        SymmetryPlane(
            point=predicted_planes_6[idx, 3::].detach().numpy(),
            normal=predicted_planes_6[idx, 0:3].detach().numpy()
        )
        for idx in range(predicted_planes_6.shape[0])
    ]

    # Reflect points
    reflected_points = [
        apply_symmetry(points, predicted_planes_4[idx, 0:3], predicted_planes_4[idx, 3])
        for idx in range(predicted_planes_4.shape[0])
    ]

    # Visualize
    ps.init()
    ps.remove_all_structures()
    ps.register_point_cloud("original pcd", points.detach().numpy())

    for idx, sym_plane in enumerate(original_symmetries):
        ps.register_surface_mesh(
            f"original_sym_plane_{idx}",
            sym_plane.coords,
            sym_plane.trianglesBase,
            smooth_shade=True,
            enabled=False,
        )

    for idx, sym_plane in enumerate(predicted_symmetries):
        ps.register_surface_mesh(
            f"predicted_sym_plane_{idx}",
            sym_plane.coords,
            sym_plane.trianglesBase,
            smooth_shade=True,
            enabled=False,
        )

    for idx, ref_points in enumerate(reflected_points):
        ps.register_point_cloud(f"reflected_points_{idx}", ref_points.detach().numpy(), enabled=False,)

    ps.show()


def visualize_prediction_results(batch, y_pred):
    """

    :param batch: tuple of original_points, voxel, cp, syms
    :param y_pred:
    :return:
    """
    original_points, voxel, cp, syms = batch
    y_pred[:, :, 0:3] = y_pred[:, :, 0:3] / torch.linalg.norm(y_pred[:, :, 0:3], dim=2).unsqueeze(2).repeat(1,1,3)
    transformed_y_pred = transform_representation(y_pred)[:, :, 0:6]
    for idx in range(original_points.shape[0]):
        visualize_prediction(
            predicted_planes_4=y_pred[idx, :, :],
            predicted_planes_6=transformed_y_pred[idx, :, :],
            real_planes=syms[idx, :, :],
            points=original_points[idx, :, :]
        )


# -1 => No sampling
dataset = VoxelDataset("/data/voxel_dataset", sample_size=-1)
dataloader = DataLoader(dataset, collate_fn=dataset.collate_fn, num_workers=3, batch_size=1)
iter_dataloader = iter(dataloader)
batch = next(iter_dataloader)
original_points, voxel, cp, syms_other_rep = batch

# Had to add sample size by hand because this model was trained on an earlier version of the module.
model = LightingPRSNet.load_from_checkpoint("/home/gustavo_santelices/Documents/Universidad/memoria_al_limpio/modelos_interesantes/bueno/checkpoints/epoch=0-step=422.ckpt",
                                            sample_size=1)
loss_fn = ChamferLoss(0)

# B x H x 4
y_pred = model.net.forward(voxel)
phc = get_phc(batch, y_pred, theta=3, eps_percent=0.03)

loss = loss_fn.forward(y_pred, original_points, voxel, cp)
print("PHC: ", phc)

visualize_prediction_results(batch, y_pred)
