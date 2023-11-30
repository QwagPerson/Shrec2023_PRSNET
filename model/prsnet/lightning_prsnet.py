import torch
from model.prsnet.prs_net import PRSNet
from model.prsnet.losses import SymLoss, ChamferLoss
from model.prsnet.metrics import get_phc, undo_transform_representation
import lightning as L
from model.prsnet.metrics import transform_representation


def reverse_points_scaling_transformation(points, transformation_params):
    """

    :param points: B x S x 3
    :param transformation_params: B x 4
    :return: B x S x 3
    """
    points = points.clone()

    mins = transformation_params[:, 0:3]  # B x 3
    max_norms = transformation_params[:, 3]  # B

    bs = points.shape[0]

    for bidx in range(bs):
        curr_max = max_norms[bidx]  # 1
        curr_min = mins[bidx]  # 3
        points[bidx, :, :] = points[bidx, :, :] * curr_max + curr_min

    return points


def reverse_plane_scaling_transformation(y_out, transformation_params):
    """

    :param y_out: B x N x 7
    :param transformation_params: B x 4
    :return: y_out where the point of each plane has its scaling reversed.
    """
    y_out = y_out.clone()

    mins = transformation_params[:, 0:3]  # B x 3
    max_norms = transformation_params[:, 3]  # B

    bs = y_out.shape[0]
    n_heads = y_out.shape[1]

    for bidx in range(bs):
        curr_max = max_norms[bidx]  # 1
        curr_min = mins[bidx]  # 3
        for nidx in range(n_heads):
            y_out[bidx, nidx, 3:6] = y_out[bidx, nidx, 3:6] * curr_max + curr_min

    return y_out


class LightingPRSNet(L.LightningModule):
    def __init__(self,
                 name: str = "Name of model",
                 input_resolution: int = 32,
                 amount_of_heads: int = 12,
                 out_features: int = 4,
                 use_bn: bool = True,
                 loss_used: str = "symloss",
                 reg_coef: float = 25.0,
                 ):
        super().__init__()
        self.name = name
        self.net = PRSNet(input_resolution, amount_of_heads, out_features, use_bn=use_bn)
        if loss_used == "chamfer":
            self.loss_fn = ChamferLoss(reg_coef)
        elif loss_used == "symloss":
            self.loss_fn = SymLoss(reg_coef)
        self.loss_used = loss_used
        self.save_hyperparameters(ignore=["net", "loss_fn"])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        idx, transformation_params, sample_points, voxel_grids, voxel_grids_cp, y_true = batch
        y_pred = self.net.forward(voxel_grids)

        loss = self.loss_fn.forward(batch, y_pred)
        train_phc = get_phc(batch, y_pred)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_phc", train_phc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        idx, transformation_params, sample_points, voxel_grids, voxel_grids_cp, y_true = batch
        y_pred = self.net.forward(voxel_grids)

        loss = self.loss_fn.forward(batch, y_pred)
        val_phc = get_phc(batch, y_pred)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_phc", val_phc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        idx, transformation_params, sample_points, voxel_grids, voxel_grids_cp, y_true = batch
        y_pred = self.net.forward(voxel_grids)

        loss = self.loss_fn.forward(batch, y_pred)
        test_phc = get_phc(batch, y_pred)

        out_sample_points = reverse_points_scaling_transformation(sample_points, transformation_params)
        out_y_pred = transform_representation(y_pred)
        out_y_pred = reverse_plane_scaling_transformation(out_y_pred, transformation_params)
        out_y_pred = undo_transform_representation(out_y_pred)

        out_y_true = reverse_plane_scaling_transformation(y_true, transformation_params)

        out_test_phc = get_phc(
            (idx, transformation_params, out_sample_points, voxel_grids, voxel_grids_cp, out_y_true),
            out_y_pred
        )

        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test_phc", test_phc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("out_test_phc", out_test_phc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        idx, transformation_params, sample_points, voxel_grids, voxel_grids_cp, y_true = batch
        y_pred = self.net.forward(voxel_grids)

        # Normalize y_pred
        y_pred[:, :, 0:3] = y_pred[:, :, 0:3] / torch.linalg.norm(y_pred[:, :, 0:3], dim=2).unsqueeze(2).repeat(1, 1, 3)

        y_pred = transform_representation(y_pred)

        y_out = reverse_plane_scaling_transformation(y_pred, transformation_params)
        y_true_out = reverse_plane_scaling_transformation(y_true, transformation_params)
        sample_points_out = reverse_points_scaling_transformation(sample_points, transformation_params)

        return idx, y_out, sample_points_out, y_pred, sample_points, y_true, y_true_out