import torch.nn as nn
import torch


def batch_dot(batch_of_points, vec):
    """
    Calculates the dot product over across a batch of points.
    :param batch_of_points: Tensor of shape Bx3
    :param vec: Tensor of shape 3
    :return: Tensor of shape B where each value is the dot product of a point with vec.
    """
    return torch.einsum('bd,d->b', batch_of_points, vec)


def apply_symmetry(points: torch.Tensor, normal: torch.Tensor, d: torch.Tensor):
    n_hat = normal / torch.linalg.norm(normal)
    distance_to_plane = batch_dot(points, n_hat) + d
    return points - 2 * torch.einsum('p,d->pd', distance_to_plane, n_hat)


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

        voxel_per_point[:, 0] = reflected_sample[:, 0] // voxel_length
        voxel_per_point[:, 1] = reflected_sample[:, 1] // voxel_length
        voxel_per_point[:, 2] = reflected_sample[:, 2] // voxel_length

        voxel_per_point = voxel_per_point.clamp(0, res - 1).int()

        # Split into x,y,z the index to be able to use it to filter
        # the grid
        x, y, z = voxel_per_point.chunk(chunks=3, dim=1)

        # Get the closest point for every point
        cp = voxel_grid_cp[x, y, z, :].squeeze()

        # Get if the voxel at every point is filled
        is_voxel_filled = voxel_grid[x, y, z]

        # Calculate distance
        distance = torch.norm(reflected_sample - cp, dim=1)# * is_voxel_filled

        return distance.mean()

    @staticmethod
    def planar_reflective_sym_reg_loss(
            planes,  # Tensor of shape [BxCx4]
    ):
        normals = planes[:, :, :3]
        normals = normals / torch.linalg.norm(normals, dim=2, keepdim=True).expand(-1, -1, 3)
        a = torch.bmm(normals, normals.transpose(1, 2)) - torch.eye(normals.shape[1]).to(normals.device)
        return torch.linalg.matrix_norm(a)

    def forward(
            self,
            predicted_planes,  # tensor of shape [BxCx4]
            sample_points,  # tensor of shape [BxNx3]
            voxel_grids,  # tensor of shape [Bx1xRxRxR]
            cp_grids,  # tensor of shape [BxRxRxRx3]
            device,
    ):
        batch_size = predicted_planes.shape[0]
        amount_of_heads = predicted_planes.shape[1]
        res = cp_grids.shape[1]

        # First regularization loss
        regularization_loss = self.planar_reflective_sym_reg_loss(predicted_planes).to(device)
        # Second reflexion loss
        reflexion_loss = torch.tensor([0.0]).to(device)
        for batch_idx in range(batch_size):
            for current_head_idx in range(amount_of_heads):
                curr_plane = predicted_planes[batch_idx, current_head_idx, :]
                curr_sample = sample_points[batch_idx, :, :]
                curr_voxel_grid = voxel_grids[batch_idx, 0, :, :, :]
                curr_cp_voxel_grid = cp_grids[batch_idx, :, :, :, :]
                reflexion_loss += self.planar_reflective_sym_distance_loss(
                    curr_plane, curr_sample, curr_voxel_grid, curr_cp_voxel_grid
                )
        return reflexion_loss + self.reg_coef * regularization_loss.sum()
