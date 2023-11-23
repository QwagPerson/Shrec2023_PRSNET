import torch
from model.prsnet.prs_net import PRSNet
from model.prsnet.sym_loss import SymLoss
from model.prsnet.metrics import phc
import lightning as L


class LightingPRSNet(L.LightningModule):
    def __init__(self, amount_of_heads, out_features, reg_coef):
        super().__init__()
        self.net = PRSNet(amount_of_heads, out_features)
        self.loss_fn = SymLoss(reg_coef)
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        sample_points, voxel_grids, voxel_grids_cp, y_true = batch
        y_pred = self.net.forward(voxel_grids).double()
        loss = self.loss_fn(y_pred, sample_points, voxel_grids, voxel_grids_cp, y_pred.device)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
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
        train_phc = phc(other_y_pred.double(), y_true)
        self.log("train_phc", train_phc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def test_step(self, batch, batch_idx):
        sample_points, voxel_grids, voxel_grids_cp, y_true = batch
        y_pred = self.net.forward(voxel_grids).double()
        loss = self.loss_fn(y_pred, sample_points, voxel_grids, voxel_grids_cp, y_pred.device)
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
        test_phc = phc(other_y_pred.double(), y_true)
        self.log("test_loss", loss)
        self.log("test_phc", test_phc)

    def validation_step(self, batch, batch_idx):
        sample_points, voxel_grids, voxel_grids_cp, y_true = batch
        y_pred = self.net.forward(voxel_grids).double()
        loss = self.loss_fn(y_pred, sample_points, voxel_grids, voxel_grids_cp, y_pred.device)
        # Transform y_pred : B x N x 4 to B x N x 7
        other_y_pred = torch.zeros((y_pred.shape[0], y_pred.shape[1], 7))
        # Copying normals
        other_y_pred[:, :, 0:3] = y_pred[:, :, 0:3]
        # Completing points
        other_y_pred[:, :, 3] = 0
        other_y_pred[:, :, 4] = 0
        # given ax+by+cz+d = 0, if x = 0 and y = 0 => z = - d / c
        other_y_pred[:, :, 5] = - y_pred[:, :, 3] / y_pred[:, :, 2]
        # Filling out confidences
        other_y_pred[:, :, 6] = 1.0
        val_phc = phc(other_y_pred.double(), y_true)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_phc", val_phc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
