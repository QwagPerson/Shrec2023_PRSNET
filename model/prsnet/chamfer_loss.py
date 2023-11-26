import torch.nn as nn
import torch
from pytorch3d.loss import chamfer_distance


def batch_apply_symmetry(points: torch.Tensor, planes: torch.Tensor):
    """

    :param points: [B x P x 3]
    :param planes: [B x 4]
    :return: [B x P x 3]
    """
    normals = planes[:, 0:3] / torch.linalg.norm(planes[:, 0:3], dim=1)  # [B x 3]
    offsets = planes[:, 3]  # [B]
    distances_to_planes = torch.einsum("bd, bnd -> bn", normals, points) + offsets  # [B x P]
    return points - 2 * torch.einsum('bp, bd-> bpd', distances_to_planes, normals)


class ChamferLoss(nn.Module):
    def __init__(self, reg_coef):
        super(ChamferLoss, self).__init__()
        self.reg_coef = reg_coef

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
        amount_of_heads = predicted_planes.shape[1]

        # First regularization loss
        regularization_loss = self.planar_reflective_sym_reg_loss(predicted_planes).to(device)

        reflexion_loss = torch.tensor([0.0]).to(device)
        for current_head_idx in range(amount_of_heads):
            predicted_planes_by_head = predicted_planes[:, current_head_idx, :]
            reflected_points = batch_apply_symmetry(sample_points, predicted_planes_by_head)
            reflexion_loss += chamfer_distance(sample_points, reflected_points)[0]

        return reflexion_loss + self.reg_coef * regularization_loss.sum()

# Make into a unit test.
"""if __name__ == "__main__":
    from dataset.simple_points_dataset import SimplePointsDataset

    points, syms = SimplePointsDataset("/data/shrec_2023/benchmark-train")[2]
    points, syms = points.float(), syms.float()
    syms_plane = syms[:, 0:4]
    syms_plane[:, 3] = - torch.einsum("bd, bd -> b", syms[:, 0:3], syms[:, 3::])
    loss = ChamferLoss(0)
    result = loss.forward(syms_plane.unsqueeze(0), points.unsqueeze(0), None, None, "cpu")
    print(result)"""
