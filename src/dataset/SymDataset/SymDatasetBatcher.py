from typing import List

import torch

from src.dataset.SymDataset.SymDatasetItem import SymDatasetItem
from src.utils.voxel import undo_voxel_transform


class SymDatasetBatcher:
    def __init__(self, item_list: List[SymDatasetItem]):
        self.item_list = item_list
        self.size = len(item_list)
        self.device = item_list[0].points.device

    def get_filenames(self):
        return [item.filename for item in self.item_list]

    def get_points(self):
        return [item.points.to(self.device) for item in self.item_list]

    def get_plane_syms(self):
        plane_syms = [item.plane_symmetries for item in self.item_list]
        for i in range(len(plane_syms)):
            if plane_syms[i] is not None:
                plane_syms[i] = plane_syms[i].to(self.device)
        return plane_syms

    def get_axis_continue_syms(self):
        axis_continue_syms = [item.axis_continue_symmetries for item in self.item_list]
        for i in range(len(axis_continue_syms)):
            if axis_continue_syms[i] is not None:
                axis_continue_syms[i] = axis_continue_syms[i].to(self.device)
        return axis_continue_syms

    def get_axis_discrete_syms(self):
        axis_discrete_symmetries = [item.axis_discrete_symmetries for item in self.item_list]
        for i in range(len(axis_discrete_symmetries)):
            if axis_discrete_symmetries[i] is not None:
                axis_discrete_symmetries[i] = axis_discrete_symmetries[i].to(self.device)
        return axis_discrete_symmetries

    def get_shape_type_classification_labels(self):
        return torch.stack([item.get_shape_type_classification_label(self.device) for item in self.item_list])

    def get_item(self, idx):
        return self.item_list[idx].to(self.device)

    def get_voxel_obj(self):
        return [item.voxel_obj for item in self.item_list]

    def get_voxel_grid_stacked(self):
        res = self.item_list[0].res
        return torch.stack([item.grid for item in self.get_voxel_obj()]).reshape((-1, 1, res, res, res)).to(self.device)

    def get_voxel_points(self):
        return torch.stack([item.points for item in self.get_voxel_obj()]).to(self.device)

    def get_voxel_plane_syms(self):
        return [item.symmetries_tensor.to(self.device) for item in self.get_voxel_obj()]

    def get_y_pred_unscaled(self, y_pred):
        return torch.stack([undo_voxel_transform(y_pred[i], self.item_list[i].voxel_obj)
                            for i in range(len(self.item_list))])
