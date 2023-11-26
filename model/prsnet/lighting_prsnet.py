import torch
from model.prsnet.prs_net import PRSNet
from model.prsnet.sym_loss import SymLoss
from model.prsnet.chamfer_loss import ChamferLoss
from model.prsnet.metrics import phc, custom_loss
import lightning as L


def get_phc(y_pred, y_true):
    # Transform y_pred : B x N x 4 to B x N x 7
    other_y_pred = torch.zeros((y_pred.shape[0], y_pred.shape[1], 7))
    # Copying normals
    other_y_pred[:, :, 0:3] = y_pred[:, :, 0:3]
    # Completing points
    other_y_pred[:, :, 3] = 0
    other_y_pred[:, :, 4] = 0
    other_y_pred[:, :, 5] = - y_pred[:, :, 3] / y_pred[:, :, 2]
    # Filling out confidences
    other_y_pred[:, :, 6] = 1.0
    phc_metric = phc(other_y_pred, y_true)
    return phc_metric


class LightingPRSNet(L.LightningModule):
    def __init__(self,
                 name,
                 batch_size,
                 input_resolution,
                 amount_of_heads,
                 out_features,
                 loss_used,
                 reg_coef,
                 ):
        super().__init__()
        self.name = name
        self.net = PRSNet(input_resolution, amount_of_heads, out_features)
        if loss_used == "chamfer":
            self.loss_fn = ChamferLoss(reg_coef)
        elif loss_used == "symloss":
            self.loss_fn = SymLoss(reg_coef)
        self.loss_used = loss_used
        self.batch_size_used = batch_size
        self.save_hyperparameters(ignore=["net", "loss_fn"])

    def training_step(self, batch, batch_idx):
        sample_points, voxel_grids, voxel_grids_cp, y_true = batch
        y_pred = self.net.forward(voxel_grids)
        loss = self.loss_fn(y_pred, sample_points, voxel_grids, voxel_grids_cp, y_true)
        train_phc = get_phc(y_pred, y_true)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_phc", train_phc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def test_step(self, batch, batch_idx):
        sample_points, voxel_grids, voxel_grids_cp, y_true = batch
        y_pred = self.net.forward(voxel_grids)

        loss = self.loss_fn(y_pred, sample_points, voxel_grids, voxel_grids_cp, y_true)
        test_phc = get_phc(y_pred, y_true)

        self.log("test_loss", loss)
        self.log("test_phc", test_phc)

    def validation_step(self, batch, batch_idx):
        sample_points, voxel_grids, voxel_grids_cp, y_true = batch
        y_pred = self.net.forward(voxel_grids)

        loss = self.loss_fn(y_pred, sample_points, voxel_grids, voxel_grids_cp, y_true)
        val_phc = get_phc(y_pred, y_true)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_phc", val_phc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
