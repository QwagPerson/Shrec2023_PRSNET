import math

import torch
import torch.nn as nn
from chamferdist import ChamferDistance

from src.model.prsnet.losses import SymLoss, apply_symmetry
from src.model.prsnet.metrics import transform_representation_cm


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
                batch_reduction="mean", point_reduction="mean",
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
    def __init__(self, normal, point, confidence, sde):
        self.normal = normal
        self.normal = normal / torch.linalg.norm(normal)
        self.point = point
        self.confidence = confidence
        self.sde = sde

    def is_close(self, another_plane, angle_threshold=30) -> bool:
        angle = torch.arccos(torch.dot(self.normal, another_plane.normal)).item() * 180 / math.pi
        return (angle < angle_threshold) | (180 - angle < angle_threshold)

    def to_tensor(self) -> torch.tensor:
        return torch.cat(
            (self.normal, self.point, self.confidence.unsqueeze(dim=0), self.sde.unsqueeze(dim=0)),
            dim=0)


def remove_duplicated_pred_planes(curr_head_predictions, angle_threshold):
    pop_plane_idx = []
    for a_idx, a_plane in enumerate(curr_head_predictions):
        for b_idx, b_plane in enumerate(curr_head_predictions):
            if a_idx != b_idx and a_plane.is_close(b_plane, angle_threshold):
                if a_plane.sde > b_plane.sde:
                    pop_plane_idx.append(a_idx)
                else:
                    pop_plane_idx.append(b_idx)

    pop_plane_idx = sorted(list(set(pop_plane_idx)))
    # Reverse to not mess up idx matching
    for idx in reversed(pop_plane_idx):
        curr_head_predictions.pop(idx)
    return curr_head_predictions


# Unused
def get_distance_2_mass_center(batch, y_pred):
    idx, transformation_params, sample_points, voxel_grids, voxel_grids_cp, _ = batch
    center_of_mass_matrix = torch.zeros((y_pred.shape[0], y_pred.shape[1]))

    bs = y_pred.shape[0]
    n_heads = y_pred.shape[1]

    for batch_idx in range(bs):
        center_of_mass = sample_points[batch_idx, :, :].mean(dim=0)
        for current_head_idx in range(n_heads):
            curr_plane = y_pred[batch_idx, current_head_idx, :]
            curr_sample = sample_points[batch_idx, :, :]

            reflected_points = apply_symmetry(curr_sample, curr_plane[0:3], curr_plane[3])
            cm_reflected_points = reflected_points.mean(dim=0)
            center_of_mass_matrix[batch_idx, current_head_idx] = torch.linalg.norm(
                center_of_mass - cm_reflected_points
            )

    return center_of_mass_matrix


class PlaneValidator(nn.Module):
    def __init__(self, sde_fn: str = None, sde_threshold: float = 1e-2, angle_threshold: float = 30):
        super(PlaneValidator, self).__init__()
        if sde_fn == "symloss" or sde_fn is None:
            self.sde_fn = symloss_adapter
        elif sde_fn =="chamfer":
            self.sde_fn = chamfer_adapter
        else:
            raise ValueError("Bad name sde_fn")
        self.sde_threshold = sde_threshold
        self.angle_threshold = angle_threshold

    def forward(self, batch, y_pred):
        idx, transformation_params, sample_points, voxel_grids, voxel_grids_cp, _ = batch

        y_pred = y_pred.clone()

        # We make sure y_pred is normalized
        y_pred[:, :, 0:3] = y_pred[:, :, 0:3] / torch.linalg.norm(y_pred[:, :, 0:3], dim=2).unsqueeze(2).repeat(1, 1, 3)

        y_pred_transformed = transform_representation_cm(y_pred, sample_points)
        sde_matrix = self.sde_fn(batch, y_pred).to(y_pred.device) # B x N

        confidences = minmax_normalization(sde_matrix).to(y_pred.device)

        bs = y_pred.shape[0]
        n_heads = y_pred.shape[1]

        predictions = []

        for bidx in range(bs):
            curr_head_predictions = []
            for nidx in range(n_heads):
                curr_plane = y_pred_transformed[bidx, nidx, :]
                curr_sde = sde_matrix[bidx, nidx]
                curr_confidence = confidences[bidx, nidx]
                plane_obj = PredictedPlane(
                    normal=curr_plane[0:3],
                    point=curr_plane[3:6],
                    confidence=curr_confidence,
                    sde=curr_sde,
                )
                curr_head_predictions.append(plane_obj)

            curr_head_predictions = [x for x in curr_head_predictions if x.sde <= self.sde_threshold]

            curr_head_predictions.sort(key=lambda x: x.confidence, reverse=True)

            curr_head_predictions = remove_duplicated_pred_planes(curr_head_predictions, self.angle_threshold)

            curr_head_predictions = [x.to_tensor() for x in curr_head_predictions]

            if len(curr_head_predictions) == 0:
                # To avoid crashing at rnn_pad_sequence
                curr_head_predictions = [
                    torch.zeros(8)
                ]

            curr_head_predictions = torch.stack(curr_head_predictions, dim=0)

            predictions.append(curr_head_predictions)

        y_pred = torch.nn.utils.rnn.pad_sequence(predictions, batch_first=True)

        return y_pred
