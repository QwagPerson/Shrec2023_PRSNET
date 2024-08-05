import torch.nn as nn
import torch
from chamferdist import ChamferDistance


def batch_dot(a, b):
    """
    Calculates the dot product over across a batch of points.

    :param a: Tensor of shape BxD
    :param b: Tensor of shape D
    :return: Tensor c of shape B where c_i is the dot product between a_i and b
    """
    return torch.einsum('bd,d->b', a, b)


def apply_symmetry(points: torch.Tensor, normal: torch.Tensor, offset: torch.Tensor):
    """
    Apply a planar reflective symmetry to a group of points.

    :param points: Tensor of shape N x 3
    :param normal: Tensor of shape 3
    :param offset: Tensor of shape 1
    :return: Reflected points, a tensor of shape N x 3 with points reflected.
    """
    n_hat = normal / torch.linalg.norm(normal)
    distance_to_plane = batch_dot(points, n_hat) + offset
    return points - 2 * torch.einsum('p,d->pd', distance_to_plane, n_hat)


def batch_apply_symmetry(points: torch.Tensor, planes: torch.Tensor):
    """
    Applies a planar reflective symmetry to a batch of a group of points with a batch of planes

    Example:
    16 batches of 1024 3D points -> points : 16 x 1024 x 3
    16 planes in R3 -> planes : 16 x 4 (Given by the parameters of the plane equation)
    Returns 16 batches of 1024 3D reflected points

    :param points:  Tensor of shape B x N x 3
    :param planes: Tensor of shape B x 4
    :return: Batch of reflected points with shape B x N x 3
    """
    normals = planes[:, 0:3] / torch.linalg.norm(planes[:, 0:3], dim=1).unsqueeze(1)  # [B x 3]
    offsets = planes[:, 3]  # [B]
    distances_to_planes = torch.einsum("bd, bnd -> bn", normals, points) + offsets.unsqueeze(1)  # [B x P]
    return points - 2 * torch.einsum('bp, bd-> bpd', distances_to_planes, normals)


def planar_reflective_sym_reg_loss(planes: torch.Tensor):  # Tensor of shape [BxCx4]
    """
    Calculates a regularization loss where is 0 when all predicted planes are orthogonal.
    It the angle between the normals of the planes.
    :param planes: Tensor of shape B x C x 4 of 3D planes
    :return: The Frobenius norm of the matrix of the normals.
    """
    normals = planes[:, :, 0:3]
    normals = normals / torch.linalg.norm(normals, dim=2, keepdim=True).expand(-1, -1, 3)
    a = torch.bmm(normals, normals.transpose(1, 2)) - torch.eye(normals.shape[1]).to(normals.device)
    return torch.linalg.matrix_norm(a)


class ChamferLoss(nn.Module):
    def __init__(self, reg_coef):
        """
        Chamfer loss is a 2 part loss.
        First part approximates the symmetry distance error (SDE) by measuring the Chamfer distance between
        the original sample of the point cloud and the reflected sample.
        Second part is a regularization loss that forces the model to predict different planes of symmetry.
        It measures the overlapping between the normals of the predicted planes.
        :param reg_coef:
        """
        super(ChamferLoss, self).__init__()
        self.distance = ChamferDistance()
        self.reg_coef = reg_coef

    def forward(
            self,
            batch,
            y_pred,
    ):
        idx, transformation_params, sample_points, voxel_grids, voxel_grids_cp, y_true = batch

        batch_size = y_pred.shape[0]
        amount_of_heads = y_pred.shape[1]
        device = y_pred.device

        # First regularization loss
        regularization_loss = planar_reflective_sym_reg_loss(y_pred).to(device)

        reflexion_loss = torch.tensor([0.0]).to(device)
        for current_head_idx in range(amount_of_heads):
            predicted_planes_by_head = y_pred[:, current_head_idx, :]
            reflected_points = batch_apply_symmetry(sample_points, predicted_planes_by_head)
            reflexion_loss += self.distance.forward(
                sample_points, reflected_points,
                batch_reduction="sum", point_reduction="sum",
                bidirectional=True
            )

        return reflexion_loss + self.reg_coef * regularization_loss.sum()


class SymLoss(nn.Module):
    def __init__(self, reg_coef):
        super(SymLoss, self).__init__()
        self.reg_coef = reg_coef

    @staticmethod
    def planar_reflective_sym_distance_loss(
            predicted_plane: torch.Tensor,  # [4]
            sample: torch.Tensor,  # [N x SampleSize]
            voxel_grid: torch.Tensor,  # [RxRxR]
            voxel_grid_cp: torch.Tensor,  # [RxRxRx3]
    ):
        n = predicted_plane[0:3]
        n_hat = n / torch.norm(n)
        d = predicted_plane[3].flatten()
        res = voxel_grid_cp.shape[0]

        # Reflect the points [Nx3]
        reflected_sample = apply_symmetry(sample, n_hat, d)

        # Classify the points onto a voxel [Nx3]
        voxel_per_point = torch.zeros_like(reflected_sample)
        voxel_length = 1.0 / res

        voxel_per_point[:, 0] = torch.div(reflected_sample[:, 0], voxel_length, rounding_mode='floor')
        voxel_per_point[:, 1] = torch.div(reflected_sample[:, 1], voxel_length, rounding_mode='floor')
        voxel_per_point[:, 2] = torch.div(reflected_sample[:, 2], voxel_length, rounding_mode='floor')

        voxel_per_point = voxel_per_point.clamp(0, res - 1).long()

        # Split into x,y,z the index to be able to use it to filter
        # the grid
        x, y, z = voxel_per_point.chunk(chunks=3, dim=1)

        # Get the closest point for every point
        cp = voxel_grid_cp[x, y, z, :].squeeze()

        # Get if the voxel at every point is filled
        is_voxel_filled = voxel_grid[x, y, z]

        # Calculate distance
        distance = torch.norm(reflected_sample - cp, dim=1) # * is_voxel_filled

        return distance.mean()

    def forward(
            self,
            batch,
            y_pred,
    ):

        device = batch.device
        batch_size = batch.size
        amount_of_heads = y_pred.shape[1]

        regularization_loss = planar_reflective_sym_reg_loss(y_pred).to(device)

        reflexion_loss = torch.tensor([0.0]).to(device)
        for batch_idx in range(batch_size):
            for current_head_idx in range(amount_of_heads):
                item = batch.get_item(batch_idx)
                voxel_obj = item.voxel_obj
                curr_plane = y_pred[batch_idx, current_head_idx, :]
                curr_sample = voxel_obj.points
                curr_voxel_grid = voxel_obj.grid
                curr_cp_voxel_grid = voxel_obj.closest_point_grid
                reflexion_loss += self.planar_reflective_sym_distance_loss(
                    curr_plane, curr_sample, curr_voxel_grid, curr_cp_voxel_grid
                )

        return reflexion_loss + self.reg_coef * regularization_loss.mean()
