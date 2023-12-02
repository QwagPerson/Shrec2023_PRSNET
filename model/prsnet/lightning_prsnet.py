import lightning as L
import torch

from model.prsnet.losses import SymLoss, ChamferLoss
from model.prsnet.metrics import get_phc
from model.prsnet.postprocessing import PlaneValidator
from model.prsnet.prs_net import PRSNet


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

    :param y_out: B x N x 8
    :param transformation_params: B x 4
    :return: y_out where the point of each plane has its scaling reversed. B x N x 7
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
                 max_sde: float = 1e-3,
                 angle_threshold: float = 30,
                 phc_angle: float = 1,
                 phc_dist_percent: float = 0.01,
                 ):
        super().__init__()
        self.name = name
        self.net = PRSNet(input_resolution, amount_of_heads, out_features, use_bn=use_bn)
        # Could (should) be implemented using submodules
        # https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_intermediate_2.html
        self.loss_used = loss_used
        if self.loss_used == "chamfer":
            self.loss_fn = ChamferLoss(reg_coef)
        elif self.loss_used == "symloss":
            self.loss_fn = SymLoss(reg_coef)
        self.val_layer = PlaneValidator(sde_threshold=max_sde)
        self.angle_threshold = angle_threshold,
        self.phc_angle = phc_angle
        self.phc_dist_percent = phc_dist_percent
        self.save_hyperparameters(ignore=["net", "loss_fn"])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        idx, transformation_params, sample_points, voxel_grids, voxel_grids_cp, y_true = batch
        y_pred = self.net.forward(voxel_grids)

        loss = self.loss_fn.forward(batch, y_pred)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        idx, transformation_params, sample_points, voxel_grids, voxel_grids_cp, y_true = batch
        y_pred = self.net.forward(voxel_grids)

        loss = self.loss_fn.forward(batch, y_pred)
        val_phc = get_phc(batch, y_pred, theta=self.phc_angle, eps_percent=self.phc_dist_percent)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_phc", val_phc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        idx, transformation_params, sample_points, voxel_grids, voxel_grids_cp, y_true = batch
        y_pred = self.net.forward(voxel_grids)

        loss = self.loss_fn.forward(batch, y_pred)
        y_pred = self.val_layer.forward(batch, y_pred)

        test_phc = get_phc(batch, y_pred, theta=self.phc_angle, eps_percent=self.phc_dist_percent)

        # Descaling
        out_sample_points = reverse_points_scaling_transformation(sample_points, transformation_params)
        out_y_pred = reverse_plane_scaling_transformation(y_pred[:, :, 0:6], transformation_params)
        out_y_true = reverse_plane_scaling_transformation(y_true, transformation_params)

        out_test_phc = get_phc(
            (idx, transformation_params, out_sample_points, voxel_grids, voxel_grids_cp, out_y_true),
            out_y_pred, theta=self.phc_angle, eps_percent=self.phc_dist_percent)

        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test_phc", test_phc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("out_test_phc", out_test_phc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        idx, transformation_params, sample_points, voxel_grids, voxel_grids_cp, _ = batch
        y_pred = self.net.forward(voxel_grids)
        y_pred = self.val_layer.forward(batch, y_pred)

        # Descaling
        out_sample_points = reverse_points_scaling_transformation(sample_points, transformation_params)
        out_y_pred = reverse_plane_scaling_transformation(y_pred[:, :, 0:6], transformation_params)

        # fig_idx, y_out, sample_points_out, y_pred, sample_points, y_true, y_true_out = prediction
        return idx, out_y_pred, out_sample_points, y_pred, sample_points, torch.zeros_like(y_pred), torch.zeros_like(
            y_pred)
