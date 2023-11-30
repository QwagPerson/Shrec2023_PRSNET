import math

import torch
from model.prsnet.losses import batch_dot


def get_angle(a, b):
    """

    :param a: Shape B x N x 3
    :param b: Shape B x N x 3
    :return: Shape B x N of angles (in degrees) between each vector
    """
    inner_product = torch.einsum('bnd,bnd->bn', a, b)
    a_norm = torch.linalg.norm(a, dim=2)
    b_norm = torch.linalg.norm(b, dim=2)
    # Avoiding div by 0
    cos = inner_product / ((a_norm * b_norm) + 1e-8)
    cos = torch.clamp(cos, -1, 1)
    angle = torch.acos(cos)
    return angle * 180 / math.pi


def match(y_pred, y_true, theta, eps):
    """

    :param eps: Shape B
    :param theta:
    :param y_pred: Shape B x N x 6 with only one plane
    :param y_true: Shape B x N x 6
    :return:
    """
    # B -> B X N
    eps = eps.unsqueeze(1).repeat(1, y_pred.shape[1])

    normals_true = y_true[:, :, 0:3]
    normals_pred = y_pred[:, :, 0:3]

    points_true = y_true[:, :, 3::]
    points_pred = y_pred[:, :, 3::]

    print("y_true", y_true)
    print("y_pred", y_pred)
    ds = - torch.einsum('bnd,bnd->bn', points_true, normals_true)
    print("ds", ds)

    distances = torch.abs(torch.einsum('bnd,bnd->bn', points_pred, normals_true) + ds)  # B x N
    angles = get_angle(normals_pred, normals_true)  # B x N

    print("distances")
    print(distances)
    print("angles")
    print(angles)

    angles_match = (angles < theta) | (180 - angles < theta)
    distances_match = distances < eps

    print("epss", eps)
    print(eps - distances)

    print("distances_match")
    print(distances_match)
    print("angles_match")
    print(angles_match)

    return angles_match & distances_match


def match_planes(y_pred, y_true, theta, eps):
    """

    :param eps:
    :param theta:
    :param y_pred: Shape B x 1 x 7
    :param y_true: Shape B x N x 6
    :return: Shape B where true if there was a match
    """
    N = y_true.shape[1]
    y_pred = y_pred.repeat(1, N, 1)  # B x N x 7
    y_pred = y_pred[:, :, 0:6]
    matches = match(y_pred, y_true, theta, eps)  # B x N x 1
    return matches


def get_diagonals_length(points: torch.Tensor):
    """

    :param points: Shape B x S x 3
    :return: lengths Shape B
    """
    diagonals = points.max(dim=1).values - points.min(dim=1).values
    return torch.linalg.norm(diagonals, dim=1)


def undo_transform_representation(y_pred):
    """

    :param y_pred: B x N x 7
    :return: B x N x 4
    """
    y_pred_transformed = torch.zeros((y_pred.shape[0], y_pred.shape[1], 4), device=y_pred.device)
    # Copy normals
    y_pred_transformed[:, :, 0:3] = y_pred[:, :, 0:3]
    # Calculate offset
    y_pred_transformed[:, :, 3] = - torch.einsum('bnd,bnd->bn', y_pred[:, :, 0:3], y_pred[:, :, 3:6])
    return y_pred_transformed


def transform_representation(y_pred):
    """

    :param y_pred: B x N x 4
    :return: B x N x 7
    """
    y_pred_transformed = torch.zeros((y_pred.shape[0], y_pred.shape[1], 7), device=y_pred.device)
    # Copy normals
    y_pred_transformed[:, :, 0:3] = y_pred[:, :, 0:3]
    # Choose the parameter (ABC) with the highest absolute value
    bs = y_pred.shape[0]
    n_heads = y_pred.shape[1]
    for idx_bs in range(bs):
        for idx_head in range(n_heads):
            parameter_mag = torch.abs(y_pred[idx_bs, idx_head, 0:3])
            max_val = torch.max(parameter_mag)
            max_idx = torch.argmax(parameter_mag) + 3
            y_pred_transformed[idx_bs, idx_head, max_idx] = - y_pred[idx_bs, idx_head, 3] / max_val
    # Add confidence
    y_pred_transformed[:, :, -1] = 1.0
    return y_pred_transformed


def get_phc(batch, y_pred: torch.Tensor, theta=1, eps_percent=0.01):
    """
    :param eps_percent:
    :param batch tuple of (
    sample_points, Shape B x S x 3
    voxel_grids, Shape B x R x R x R
    voxel_grids_cp, Shape B x R x R x R x 3
    y_true, Shape B x N X 6
    )
    :param y_pred: Shape B x N x 4
    :param theta:
    :return: % of matches float 0..1
    """
    # Get eps
    idx, transformation_params, sample_points, voxel_grids, voxel_grids_cp, y_true = batch
    y_pred = y_pred.detach().clone().to(y_true.device)
    eps = get_diagonals_length(sample_points) * eps_percent
    print("eps", eps)

    # Normalize y_pred
    y_pred[:, :, 0:3] = y_pred[:, :, 0:3] / torch.linalg.norm(y_pred[:, :, 0:3], dim=2).unsqueeze(2).repeat(1, 1, 3)
    print("orig_yored", y_pred)
    # Transform representation of plane B x N x 4 -> B x N x 7
    y_pred = transform_representation(y_pred)

    # Sort y_pred by confidence
    confidences = y_pred[:, :, -1].sort(descending=True).indices
    for x in range(confidences.shape[0]):
        y_pred[x, :, :] = y_pred[x, confidences[x, :], :]

    # Select only the first plane for every y_pred in B
    y_pred = y_pred[:, 0, :].unsqueeze(1)

    # Check if they match with any in y_true in B
    matches = match_planes(y_pred, y_true, theta, eps)
    # Applying any through each prediction
    matches = matches.any(dim=1)
    print(idx, matches)
    # Return Match percent
    return matches.sum().item() / torch.numel(matches)


if __name__ == "__main__":
    mock_ypred = torch.tensor([[[1.0000, 0.0000, 0.0000, -0.2950],
                                [0.0000, 0.8090, 0.5878, -0.5735],
                                [0.0000, -0.5878, 0.8090, -0.0904]]])

    y_true = torch.tensor([[[1.0000, 0.0000, 0.0000, 0.2950, 0.4109, 0.4102],
                            [0.0000, 0.8090, 0.5878, 0.2950, 0.4109, 0.4102],
                            [0.0000, -0.5878, 0.8090, 0.2950, 0.4109, 0.4102]]])

    transform_representation(mock_ypred)
    print()
