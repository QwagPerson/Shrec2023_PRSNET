from copy import deepcopy
from typing import Optional
import torch
from pathlib import Path
import polyscope as ps

from src.dataset.SymDataset.transforms.AbstractTransform import AbstractTransform
from src.dataset.SymDataset.transforms.ReverseTransform import ReverseTransform
from src.utils.plane import SymmetryPlane
from src.utils.voxel import Voxel

SHAPE_TYPE = {
    "airplane": 0,
    "car": 1,
    "chair": 2,
    "table": 3,
}
SHAPE_TYPE_AMOUNT = len(SHAPE_TYPE.keys())


def get_shape_type(filename: Path):
    try:
        return filename.parent.name
    except IndexError:
        return 'unknown'


class SymDatasetItem:
    def __init__(
            self,
            filepath: Path,
            idx: int,
            points: torch.tensor,
            plane_symmetries: Optional[torch.tensor],
            axis_continue_symmetries: Optional[torch.tensor],
            axis_discrete_symmetries: Optional[torch.tensor],
            transform: AbstractTransform,
            resolution: int,
    ):
        self.filepath = filepath
        self.filename = str(filepath.stem)
        self.shape_type = get_shape_type(self.filepath)
        self.transform = transform
        self.res = resolution

        self.idx = idx
        self.points = points

        self.plane_symmetries = plane_symmetries
        self.axis_continue_symmetries = axis_continue_symmetries
        self.axis_discrete_symmetries = axis_discrete_symmetries

        self.voxel_obj = Voxel(
            pcd=self.points,
            sym_planes=self.plane_symmetries,
            resolution=self.res
        )

    def get_item_elements(self):
        return (
            self.idx,
            self.points,
            self.plane_symmetries,
            self.axis_continue_symmetries,
            self.axis_discrete_symmetries,
        )

    def get_untransformed_item(self):
        reverse_transform = ReverseTransform(deepcopy(self.transform))
        item_elements = reverse_transform(
            *self.get_item_elements()
        )

        return SymDatasetItem(
            self.filepath,
            *item_elements,
            reverse_transform,
            self.res,
        )

    def get_shape_type_classification_label(self, device="cpu"):
        label = torch.zeros(SHAPE_TYPE_AMOUNT, device=device)
        label[SHAPE_TYPE[self.shape_type]] = 1
        return label

    def __repr__(self):
        str_rep = ""
        str_rep += f"SymmetryDatasetItem N°{self.idx}\n"
        str_rep += f"\tFilename: {self.filename}\n"
        str_rep += f"\tClass: {self.shape_type}\n"
        str_rep += f"\tResolution: {self.res}\n"
        str_rep += f"\tElements Shape:\n"
        str_rep += f"\t\tPoints: {self.points.shape}\n"
        if self.plane_symmetries is None:
            str_rep += f"\t\tPlane Syms: Not present.\n"
        else:
            str_rep += f"\t\tPlane Syms: {self.plane_symmetries.shape}\n"
        if self.axis_discrete_symmetries is None:
            str_rep += f"\t\tAxis Discrete Syms: Not present.\n"
        else:
            str_rep += f"\t\tAxis Discrete Syms: {self.axis_discrete_symmetries.shape}\n"
        if self.axis_continue_symmetries is None:
            str_rep += f"\t\tAxis Continue Syms: Not present.\n"
        else:
            str_rep += f"\t\tAxis Continue Syms: {self.axis_continue_symmetries.shape}\n"
        return str_rep

    def to(self, device):
        self.points = self.points.to(device)
        self.plane_symmetries = self.plane_symmetries.to(device) if self.plane_symmetries is not None else None
        self.axis_continue_symmetries = self.axis_continue_symmetries.to(
            device) if self.axis_continue_symmetries is not None else None
        self.axis_discrete_symmetries = self.axis_discrete_symmetries.to(
            device) if self.axis_discrete_symmetries is not None else None
        return self

    def visualize(self):
        ps.init()
        ps.remove_all_structures()
        ps.register_point_cloud("points", self.points)

        if self.plane_symmetries is not None:
            for i, plane_sym in enumerate(self.plane_symmetries):
                self._visualize_plane_sym(i, plane_sym)

        if self.axis_continue_symmetries is not None:
            for axis_cont_sym in self.axis_continue_symmetries:
                self._visualize_axis_cont_sym(axis_cont_sym)

        if self.axis_discrete_symmetries is not None:
            for axis_disc_sym in self.axis_discrete_symmetries:
                self._visualize_axis_disc_sym(axis_disc_sym)

        self.voxel_obj.visualize()

        ps.show()

    def _visualize_plane_sym(self, i, plane_sym):
        sym = SymmetryPlane(
            normal=plane_sym[0:3].detach().numpy(),
            point=plane_sym[3::].detach().numpy(),
        )
        ps.register_surface_mesh(
            f"sym_plane_{i}",
            sym.coords,
            sym.trianglesBase
        )

    def _visualize_axis_cont_sym(self, plane_sym):
        pass

    def _visualize_axis_disc_sym(self, plane_sym):
        pass
