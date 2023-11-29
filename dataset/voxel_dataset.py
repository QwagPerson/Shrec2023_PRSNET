import pickle

import torch
import os
from torch.utils.data import Dataset


def collate_fn(batch):
    # idx, transformation_params, sample, voxel_grid, voxel_grid_cp, sym_planes
    idxs = [item[0] for item in batch]
    transformation_params_list = [item[1] for item in batch]
    samples = torch.stack([item[2] for item in batch])
    voxel_grids = torch.stack([item[3] for item in batch]).unsqueeze(1)
    voxel_grids_cp = torch.stack([item[4] for item in batch])

    sym_planes = [item[5] for item in batch]
    sym_planes = torch.nn.utils.rnn.pad_sequence(sym_planes, batch_first=True)

    return (idxs, transformation_params_list,
            samples, voxel_grids,
            voxel_grids_cp, sym_planes)


class VoxelDataset(Dataset):
    def __init__(self, dataset_root, sample_size=1024):
        self.dataset_root = dataset_root

        self.transformation_params_folder = os.path.join(self.dataset_root, "transformation_params")
        self.points_folder = os.path.join(self.dataset_root, "points")
        self.voxel_grid_folder = os.path.join(self.dataset_root, "voxel_grid")
        self.closets_point_voxel_grid_folder = os.path.join(self.dataset_root, "closest_point_voxel_grid")
        self.symmetry_planes_folder = os.path.join(self.dataset_root, "symmetry_planes")

        self.sample_size = sample_size
        self.collate_fn = collate_fn

        self.validate_folders()

    def validate_folders(self):
        assert os.path.exists(self.transformation_params_folder)
        assert os.path.exists(self.points_folder)
        assert os.path.exists(self.voxel_grid_folder)
        assert os.path.exists(self.closets_point_voxel_grid_folder)
        assert os.path.exists(self.symmetry_planes_folder)

    def __len__(self):
        return len(os.listdir(self.points_folder))

    def __getitem__(self, idx):
        # Read items
        with open(os.path.join(self.transformation_params_folder, f"trans_params_{idx}.pt"), "rb") as f:
            transformation_params = pickle.load(f)

        points = torch.load(os.path.join(self.points_folder, f"points_{idx}.pt"))
        voxel_grid = torch.load(os.path.join(self.voxel_grid_folder, f"voxel_grid_{idx}.pt"))
        voxel_grid_cp = torch.load(os.path.join(self.closets_point_voxel_grid_folder,
                                                f"closest_point_voxel_grid_{idx}.pt"))
        sym_planes = torch.load(os.path.join(self.symmetry_planes_folder, f"symmetry_planes_{idx}.pt"))

        if self.sample_size != -1:
            p_idx = torch.randperm(self.sample_size) % points.shape[0]
            sample = points[p_idx]
        else:
            sample = points

        return idx, transformation_params, sample.float(), voxel_grid.float(), voxel_grid_cp.float(), sym_planes.float()
