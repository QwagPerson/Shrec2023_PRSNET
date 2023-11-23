import torch
from model.prsnet.sym_loss import batch_dot


def get_angle(a, b):
    """

    :param a: Shape B x N x 3
    :param b: Shape B x N x 3
    :return: Shape B x N of angles between each vector
    """
    inner_product = torch.einsum('bnd,bnd->bn', a, b)
    a_norm = torch.linalg.norm(a, dim=2)
    b_norm = torch.linalg.norm(a, dim=2)
    cos = inner_product / (a_norm * b_norm)
    cos = torch.clamp(cos, -1, 1)
    angle = torch.acos(cos)
    return angle


def match(y_pred, y_true, theta, eps):
    """

    :param y_pred: Shape B x N x 6 with only one plane
    :param y_true: Shape B x N x 6
    :return:
    """
    normals_true = y_true[:, :, 0:3]
    normals_pred = y_pred[:, :, 0:3]

    points_true = y_true[:, :, 3::]
    points_pred = y_pred[:, :, 3::]

    ds = - torch.einsum('bnd,bnd->bn', points_true, normals_true)

    distances = torch.einsum('bnd,bnd->bn', points_pred, normals_true) + ds  # batch_dot(points, n_hat) + d
    angles = get_angle(normals_pred, normals_true)

    angles_match = angles < theta
    distances_match = distances < eps

    return angles_match & distances_match


def match_planes(y_pred, y_true, theta, eps):
    """

    :param y_pred: Shape B x 1 x 7
    :param y_true: Shape B x N x 6
    :return: Shape B where true if there was a match
    """
    N = y_true.shape[1]
    y_pred = y_pred.repeat(1, N, 1)  # B x N x 7
    y_pred = y_pred[:, :, 0:6]
    matches = match(y_pred, y_true, theta, eps)  # B x N x 1
    return matches


def phc(y_pred: torch.Tensor, y_true: torch.Tensor, theta=1.0, eps=1e-8):
    """

    :param eps:
    :param theta:
    :param y_pred: Shape B x N x 7
    :param y_true: Shape B x N x 6
    :return: % of matches float 0..1
    """
    # Sort y_pred by confidence
    confidences = y_pred[:, :, -1].sort(descending=True).indices
    for x in range(confidences.shape[0]):
        y_pred[x, :, :] = y_pred[x, confidences[x, :], :]

    # Select only the first plane for every y_pred in B
    y_pred = y_pred[:, 0, :].unsqueeze(1)
    #Handling devices
    y_pred = y_pred.to(y_true.device)
    # Check if they match with any in y_true in B
    matches = match_planes(y_pred, y_true, theta, eps)
    # Return Match percent
    return matches.sum().item() / torch.numel(matches)


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


