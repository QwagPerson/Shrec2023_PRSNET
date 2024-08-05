import lightning as L
import torch

from src.model.prsnet.losses import SymLoss, ChamferLoss
from src.metrics.eval_script import calculate_metrics_from_predictions, get_match_sequence_plane_symmetry
from src.model.prsnet.postprocessing import PlaneValidator
from src.model.prsnet.prs_net import PRSNet
from src.utils.voxel import transform_representation


def str_to_loss(loss_str, reg_coef):
    if loss_str == "chamfer":
        return ChamferLoss(reg_coef)
    elif loss_str == "symloss":
        return SymLoss(reg_coef)
    else:
        ValueError("Unknown Loss Chosen.")


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

    mins = transformation_params[:, 0:3].to(y_out.device)  # B x 3
    max_norms = transformation_params[:, 3].to(y_out.device)  # B

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
                 input_resolution: int = 32,
                 amount_of_heads: int = 3,
                 out_features: int = 4,
                 use_bn: bool = True,
                 loss_used: str = "symloss",
                 reg_coef: float = 25.0,
                 sde_threshold: float = 1e-3,
                 angle_threshold: float = 30,
                 sde_fn: str = "symloss",
                 eps: float = 0.01,
                 theta: float = 0.00015230484,
                 conf_threshold: float = 0.1,
                 ):
        super().__init__()
        self.sde_fn = sde_fn
        self.sde_threshold = sde_threshold
        self.angle_threshold = angle_threshold,
        self.val_layer = PlaneValidator(sde_fn=sde_fn, sde_threshold=self.sde_threshold,
                                        angle_threshold=angle_threshold)

        self.net = PRSNet(input_resolution, amount_of_heads, out_features, use_bn=use_bn)

        self.loss_used = loss_used
        self.reg_coef = reg_coef
        self.loss_fn = str_to_loss(self.loss_used, self.reg_coef)
        self.pdict = {
            "eps": eps,
            "theta": theta,
            "confidence_threshold": conf_threshold,
        }

        self.save_hyperparameters(ignore=["net", "loss_fn"])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer

    def _log_step_metrics(self, stage, scale, batch_size, map_, phc):
        self.log(f"{stage}_{scale}_map", map_,
                 batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"{stage}_{scale}_map", phc,
                 batch_size=batch_size,  on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def _register_metrics(self, stage, batch, y_pred):
        in_voxel_pred = [(batch.get_voxel_points(), transform_representation(y_pred), batch.get_voxel_plane_syms())]
        in_map, in_phc, _ = calculate_metrics_from_predictions(in_voxel_pred, get_match_sequence_plane_symmetry,
                                                               self.pdict)

        out_voxel_pred = [(batch.get_points(), batch.get_y_pred_unscaled(y_pred), batch.get_plane_syms())]
        out_map, out_phc, _ = calculate_metrics_from_predictions(out_voxel_pred, get_match_sequence_plane_symmetry,
                                                                 self.pdict)

        self._log_step_metrics(stage, "in", batch.size, in_map, in_phc)
        self._log_step_metrics(stage, "out", batch.size, out_map, out_phc)

        return in_map, in_phc, out_map, out_phc

    def _step(self, stage, batch):
        y_pred = self.net.forward(batch.get_voxel_grid_stacked())
        loss = self.loss_fn.forward(batch, y_pred)

        metrics = self._register_metrics(stage, batch, y_pred)
        self.log(f"{stage}_loss", loss, batch_size=batch.size,
                 on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss, metrics

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        loss, _ = self._step("train", batch)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss, _ = self._step("val", batch)
        return loss

    def test_step(self, batch, batch_idx):
        loss, _ = self._step("test", batch)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        y_pred = self.net.forward(batch.get_voxel_grid_stacked())
        loss = self.loss_fn.forward(batch, y_pred)
        return batch, y_pred, loss
