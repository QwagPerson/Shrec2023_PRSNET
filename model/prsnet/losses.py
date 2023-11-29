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
    normals = planes[:, :, :3]
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
            predicted_planes,  # tensor of shape [BxCx4]
            sample_points,  # tensor of shape [BxNx3]
            voxel_grids,  # tensor of shape [Bx1xRxRxR]
            cp_grids,  # tensor of shape [BxRxRxRx3]
    ):
        amount_of_heads = predicted_planes.shape[1]
        device = predicted_planes.device

        # First regularization loss
        regularization_loss = planar_reflective_sym_reg_loss(predicted_planes).to(device)

        reflexion_loss = torch.tensor([0.0]).to(device)
        for current_head_idx in range(amount_of_heads):
            predicted_planes_by_head = predicted_planes[:, current_head_idx, :]
            reflected_points = batch_apply_symmetry(sample_points, predicted_planes_by_head)
            reflexion_loss += self.distance(
                sample_points, reflected_points,
                batch_reduction="mean", point_reduction="mean",
                bidirectional=True)

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

        voxel_per_point[:, 0] = reflected_sample[:, 0] // voxel_length
        voxel_per_point[:, 1] = reflected_sample[:, 1] // voxel_length
        voxel_per_point[:, 2] = reflected_sample[:, 2] // voxel_length

        voxel_per_point = voxel_per_point.clamp(0, res - 1).long()

        # Split into x,y,z the index to be able to use it to filter
        # the grid
        x, y, z = voxel_per_point.chunk(chunks=3, dim=1)

        # Get the closest point for every point
        cp = voxel_grid_cp[x, y, z, :].squeeze()

        # Get if the voxel at every point is filled
        is_voxel_filled = voxel_grid[x, y, z]

        # Calculate distance
        distance = torch.norm(reflected_sample - cp, dim=1)  # * is_voxel_filled

        return distance.mean()

    def forward(
            self,
            predicted_planes,  # tensor of shape [BxCx4]
            sample_points,  # tensor of shape [BxNx3]
            voxel_grids,  # tensor of shape [Bx1xRxRxR]
            cp_grids,  # tensor of shape [BxRxRxRx3]
    ):
        batch_size = predicted_planes.shape[0]
        amount_of_heads = predicted_planes.shape[1]
        res = cp_grids.shape[1]
        device = predicted_planes.device

        # First regularization loss
        regularization_loss = planar_reflective_sym_reg_loss(predicted_planes).to(device)
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


"""# Make into a unit test.
if __name__ == "__main__":
    from dataset.voxel_dataset import VoxelDataset

    points, voxel, cp, syms = VoxelDataset("/data/voxel_dataset", sample_size=-1)[0]
    syms_plane = syms[:, 0:4]
    syms_plane[:, 3] = - torch.einsum("bd, bd -> b", syms[:, 0:3], syms[:, 3::])
    loss = ChamferLoss(0)
    result = loss.forward(syms_plane.unsqueeze(0), points.unsqueeze(0), None, None, "cpu")
    print(result)"""

def custom_loss(y_pred: torch.Tensor, y_true: torch.Tensor):
    """
    :param y_pred: Shape B x 1 x 6
    :param y_true: Shape B x M x 6
    :return: float of loss
    """
    M = y_true.shape[1]
    y_pred = y_pred.repeat(1, M, 1)  # B x N x 7

    # B x M x 1
    normals_true = y_true[:, :, 0:3]
    normals_pred = y_pred[:, :, 0:3]

    points_true = y_true[:, :, 3::]
    points_pred = y_pred[:, :, 3::]

    ds = - torch.einsum('bnd,bnd->bn', points_true, normals_true)

    distances = torch.abs(torch.einsum('bnd,bnd->bn', points_pred, normals_true) + ds)
    angles = get_angle(normals_pred, normals_true)

    min_distance = distances.min()
    min_angles = angles.min()

    return min_distance + min_angles
