import math

import torch.nn as nn
import torch
from model.prsnet.metrics import transform_representation
from model.prsnet.losses import SymLoss, apply_symmetry
from chamferdist import ChamferDistance


def chamfer_adapter(batch, y_pred):
    idx, transformation_params, sample_points, voxel_grids, voxel_grids_cp, _ = batch
    chamfer_matrix = torch.zeros((y_pred.shape[0], y_pred.shape[1]))

    bs = y_pred.shape[0]
    n_heads = y_pred.shape[1]

    distance = ChamferDistance()

    for batch_idx in range(bs):
        for current_head_idx in range(n_heads):
            curr_plane = y_pred[batch_idx, current_head_idx, :]
            curr_sample = sample_points[batch_idx, :, :]

            reflected_points = apply_symmetry(curr_sample, curr_plane[0:3], curr_plane[3])

            chamfer_matrix[batch_idx, current_head_idx] = distance.forward(
                reflected_points.unsqueeze(dim=0), curr_sample.unsqueeze(dim=0),
                batch_reduction="sum", point_reduction="sum",
                bidirectional=True
            )

    return chamfer_matrix


def symloss_adapter(batch, y_pred):
    idx, transformation_params, sample_points, voxel_grids, voxel_grids_cp, _ = batch
    symloss_matrix = torch.zeros((y_pred.shape[0], y_pred.shape[1]))

    bs = y_pred.shape[0]
    n_heads = y_pred.shape[1]

    for batch_idx in range(bs):
        for current_head_idx in range(n_heads):
            curr_plane = y_pred[batch_idx, current_head_idx, :]
            curr_sample = sample_points[batch_idx, :, :]
            curr_voxel_grid = voxel_grids[batch_idx, 0, :, :, :]
            curr_cp_voxel_grid = voxel_grids_cp[batch_idx, :, :, :, :]
            symloss_matrix[batch_idx, current_head_idx] = SymLoss.planar_reflective_sym_distance_loss(
                curr_plane, curr_sample, curr_voxel_grid, curr_cp_voxel_grid
            )

    return symloss_matrix


class PredictedPlane:
    def __init__(self, normal, point, confidence, sde, cm):
        self.normal = normal
        self.normal = normal / torch.linalg.norm(normal)
        self.point = point
        self.confidence = confidence
        self.sde = sde
        self.cm = cm

    def is_close(self, another_plane) -> bool:
        angle = torch.arccos(torch.dot(self.normal, another_plane.normal))
        return angle < math.pi / 6

    def to_tensor(self) -> torch.tensor:
        return torch.cat(
            (self.normal, self.point, self.confidence.unsqueeze(dim=0), self.sde.unsqueeze(dim=0), self.cm.unsqueeze(dim=0)),
            dim=0)


def minmax_normalization(sde):
    """

    :param sde: Shape B x N
    :return: Shape B x N
    """
    confidences = sde.clone()
    bs = sde.shape[0]
    for bidx in range(bs):
        curr_confidences = confidences[bidx, :]
        confidences[bidx, :] = (curr_confidences - curr_confidences.min())
        confidences[bidx, :] = confidences[bidx, :] / torch.linalg.norm(curr_confidences.max())

    return 1 - confidences


def remove_duplicated_pred_planes(curr_head_predictions):
    pop_plane_idx = []
    for a_idx, a_plane in enumerate(curr_head_predictions):
        for b_idx, b_plane in enumerate(curr_head_predictions):
            if a_idx != b_idx and a_plane.is_close(b_plane):
                if a_plane.sde > b_plane.sde:
                    pop_plane_idx.append(a_idx)
                else:
                    pop_plane_idx.append(b_idx)

    pop_plane_idx = list(set(pop_plane_idx))
    # Reverse to not mess up idx matching
    for idx in reversed(pop_plane_idx):
        curr_head_predictions.pop(idx)
    return curr_head_predictions


def get_distance_2_mass_center(batch, y_pred):
    idx, transformation_params, sample_points, voxel_grids, voxel_grids_cp, _ = batch
    center_of_mass_matrix = torch.zeros((y_pred.shape[0], y_pred.shape[1]))

    bs = y_pred.shape[0]
    n_heads = y_pred.shape[1]

    for batch_idx in range(bs):
        center_of_mass = sample_points[batch_idx, :, :].mean(dim=0)
        print("Centro real", center_of_mass)
        for current_head_idx in range(n_heads):
            curr_plane = y_pred[batch_idx, current_head_idx, :]
            curr_sample = sample_points[batch_idx, :, :]

            reflected_points = apply_symmetry(curr_sample, curr_plane[0:3], curr_plane[3])
            cm_reflected_points = reflected_points.mean(dim=0)
            print("IDX cabezal", current_head_idx)
            print("Plano 4 utilizado", curr_plane)
            print("Plano 6 utilizado", transform_representation(curr_plane.unsqueeze(dim=0).unsqueeze(dim=0)))
            print("Centro de masa reflejado", cm_reflected_points)
            center_of_mass_matrix[batch_idx, current_head_idx] = torch.linalg.norm(
                center_of_mass - cm_reflected_points
            )

    return center_of_mass_matrix


class PlaneValidator(nn.Module):
    def __init__(self, sde_fn=None):
        super(PlaneValidator, self).__init__()
        # self.sde_fn = symloss_adapter
        self.sde_fn = chamfer_adapter

    def forward(self, batch, y_pred, sde_threshold=1e3):
        idx, transformation_params, sample_points, voxel_grids, voxel_grids_cp, _ = batch

        y_pred_transformed = transform_representation(y_pred)
        sde = self.sde_fn(batch, y_pred)  # B x N
        center_of_mass = get_distance_2_mass_center(batch, y_pred)

        confidences = minmax_normalization(sde)

        bs = y_pred.shape[0]
        n_heads = y_pred.shape[1]

        predictions = []

        for bidx in range(bs):
            curr_head_predictions = []
            curr_figure_idx = idx[bidx].item()

            for nidx in range(n_heads):
                curr_plane = y_pred_transformed[bidx, nidx, :]
                curr_sde = sde[bidx, nidx]
                curr_confidence = confidences[bidx, nidx]
                curr_cm = center_of_mass[bidx, nidx]
                plane_obj = PredictedPlane(
                    normal=curr_plane[0:3],
                    point=curr_plane[3:6],
                    confidence=curr_confidence,
                    sde=curr_sde,
                    cm=curr_cm
                )
                curr_head_predictions.append(plane_obj)

            curr_head_predictions = [x for x in curr_head_predictions if x.sde <= sde_threshold]

            curr_head_predictions = [x for x in curr_head_predictions if x.cm <= 1e-2]

            curr_head_predictions.sort(key=lambda x: x.confidence, reverse=True)

            curr_head_predictions = remove_duplicated_pred_planes(curr_head_predictions)

            curr_head_predictions = [x.to_tensor() for x in curr_head_predictions]
            if len(curr_head_predictions) == 0:
                curr_head_predictions = [
                    torch.zeros(8)
                ]

            curr_head_predictions = torch.stack(curr_head_predictions, dim=0)

            predictions.append(curr_head_predictions)

        y_pred = torch.nn.utils.rnn.pad_sequence(predictions, batch_first=True)
        return y_pred
