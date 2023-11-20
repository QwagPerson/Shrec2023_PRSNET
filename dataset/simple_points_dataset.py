import numpy as np
import torch
from torch.utils.data import Dataset
import os


def read_sym_planes(idx: int, data_path: str) -> torch.tensor:
    """
    Read symmetry planes from file with its first line being the number of symmetry planes
    and the rest being the symmetry planes.
    """

    with open(os.path.join(data_path, f"points{idx}_sym.txt")) as f:
        int(f.readline().strip())
        sym_planes = torch.tensor(np.loadtxt(f))
    return sym_planes


def read_shrec2023_points(idx: int, data_path: str) -> (torch.tensor, torch.tensor):
    """
    Reads the points and symmetry planes of the index at idx and data_path location.
    """
    points = torch.tensor(np.loadtxt(os.path.join(data_path, f"points{idx}.txt")))
    with open(os.path.join(data_path, f"points{idx}_sym.txt")) as f:
        sym_amount = int(f.readline().strip())
        sym_planes = torch.tensor(np.loadtxt(f))
        if sym_amount == 1:
            sym_planes = sym_planes.unsqueeze(0)
    return points, sym_planes


class SimplePointsDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def __len__(self):
        return len(os.listdir(self.data_dir)) // 2

    def __getitem__(self, idx):
        points, syms = read_shrec2023_points(idx, self.data_dir)
        return points, syms
