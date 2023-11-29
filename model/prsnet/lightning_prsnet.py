import torch
from model.prsnet.prs_net import PRSNet
from model.prsnet.sym_loss import SymLoss
from model.prsnet.chamfer_loss import ChamferLoss
from model.prsnet.metrics import get_phc, custom_loss
import lightning as L
import math


class LightingPRSNet(L.LightningModule):
    def __init__(self,
                 name,
                 batch_size,
                 input_resolution,
                 amount_of_heads,
                 out_features,
                 use_bn,
                 loss_used,
                 reg_coef,
                 sample_size
                 ):
        super().__init__()
        self.name = name
        self.net = PRSNet(input_resolution, amount_of_heads, out_features, use_bn=use_bn)
        if loss_used == "chamfer":
            self.loss_fn = ChamferLoss(reg_coef)
        elif loss_used == "symloss":
            self.loss_fn = SymLoss(reg_coef)
        self.loss_used = loss_used
        self.batch_size_used = batch_size
        self.sample_size = sample_size
        self.save_hyperparameters(ignore=["net", "loss_fn"])

    def training_step(self, batch, batch_idx):
        sample_points, voxel_grids, voxel_grids_cp, y_true = batch
        y_pred = self.net.forward(voxel_grids)
        # B x H x 4 : Normalizacion
        # Normalizer las normales de y_pred
        loss = self.loss_fn.forward(y_pred, sample_points, voxel_grids, voxel_grids_cp)
        train_phc = get_phc(batch, y_pred)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_phc", train_phc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def test_step(self, batch, batch_idx):
        sample_points, voxel_grids, voxel_grids_cp, y_true = batch
        y_pred = self.net.forward(voxel_grids)

        loss = self.loss_fn.forward(y_pred, sample_points, voxel_grids, voxel_grids_cp)

        test_phc = get_phc(batch, y_pred)

        self.log("test_loss", loss)
        self.log("test_phc", test_phc)

    def validation_step(self, batch, batch_idx):
        sample_points, voxel_grids, voxel_grids_cp, y_true = batch
        y_pred = self.net.forward(voxel_grids)

        loss = self.loss_fn.forward(y_pred, sample_points, voxel_grids, voxel_grids_cp)
        val_phc = get_phc(batch, y_pred)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_phc", val_phc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
