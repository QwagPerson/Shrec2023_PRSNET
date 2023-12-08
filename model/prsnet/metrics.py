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

    ds = - torch.einsum('bnd,bnd->bn', points_true, normals_true)

    distances = torch.abs(torch.einsum('bnd,bnd->bn', points_pred, normals_true) + ds)  # B x N
    angles = get_angle(normals_pred, normals_true)  # B x N

    angles_match = (angles < theta) | (180 - angles < theta)
    distances_match = distances < eps

    print("===")
    print("y_pred", undo_transform_representation(y_pred))
    print("y_true", undo_transform_representation(y_true))
    print("distances", distances, "<", eps, distances_match)
    print("angles", angles, "<", theta, angles_match)
    print("===")

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


def transform_representation_cm(y_pred, points):
    """
    CM porque toma el punto más cercano en el plano del centro de masa de los puntos
    :param y_pred: B x N x 4
    :param points B x S x 3
    :return: B x N x 7
    """
    # get center of mass
    cm = points.mean(dim=1)

    y_pred_transformed = torch.zeros((y_pred.shape[0], y_pred.shape[1], 7), device=y_pred.device)
    # Copy normals
    y_pred_transformed[:, :, 0:3] = y_pred[:, :, 0:3]

    # Get point
    bs = y_pred.shape[0]
    n_heads = y_pred.shape[1]
    for idx_bs in range(bs):
        for idx_head in range(n_heads):
            normal = y_pred[idx_bs, idx_head, 0:3]
            offset = y_pred[idx_bs, idx_head, 3]
            point = cm[idx_bs, :]
            signed_distance = torch.dot(point, normal) + offset
            point_in_plane = point - (signed_distance * normal)
            y_pred_transformed[idx_bs, idx_head, 3:6] = point_in_plane
    # Add confidence
    y_pred_transformed[:, :, -1] = 1.0
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
            max_idx = torch.argmax(parameter_mag)  # A B C
            max_val = y_pred[idx_bs, idx_head, max_idx]
            max_idx = max_idx + 3
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
    idx, transformation_params, sample_points, _, _, y_true = batch
    y_pred = y_pred.detach().clone().to(y_true.device)
    eps = get_diagonals_length(sample_points) * eps_percent

    # Normalize y_pred
    y_pred[:, :, 0:3] = y_pred[:, :, 0:3] / torch.linalg.norm(y_pred[:, :, 0:3], dim=2).unsqueeze(2).repeat(1, 1, 3)
    # Transform representation of plane B x N x 4 -> B x N x 7
    y_pred = transform_representation_cm(y_pred, sample_points)

    # Sort y_pred by confidence
    confidences = y_pred[:, :, -1].sort(descending=True).indices
    for x in range(confidences.shape[0]):
        y_pred[x, :, :] = y_pred[x, confidences[x, :], :]

    y_pred = y_pred[:, 0, :].unsqueeze(1)

    # Check if they match with any in y_true in B
    matches = match_planes(y_pred, y_true, theta, eps)
    # Applying any through each prediction
    matches = matches.any(dim=1)

    # Return Match percent
    return matches.sum().item() / torch.numel(matches)
