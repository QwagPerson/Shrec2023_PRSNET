from typing import List

import torch
import polyscope as ps

from src.utils.plane import SymmetryPlane


# Computes the closest point on the point cloud to each grid center
# Returns a tensor of shape ((1/voxel_size)**3,3)
# The idea is to not compute this everytime as it is O((1/voxel_size)**3)
# algorithm
def compute_closest_points(pcd: torch.Tensor,
                           resolution: int) -> torch.Tensor:
    voxel_amount = resolution
    voxel_size = 1.0 / voxel_amount
    cp_voxel_grid = torch.zeros([voxel_amount, voxel_amount, voxel_amount, 3])
    for i in range(voxel_amount):
        for j in range(voxel_amount):
            for k in range(voxel_amount):
                # Compute the center of this voxel
                voxel_center = torch.Tensor([
                    voxel_size * i + voxel_size / 2,
                    voxel_size * j + voxel_size / 2,
                    voxel_size * k + voxel_size / 2,
                ])
                # Compute the closest point to the voxel_center
                distances = torch.norm(voxel_center - pcd, dim=1)
                idx_cp = torch.argmin(distances)
                cp = pcd[idx_cp]

                # print(f"Closest point to {voxel_center} is {cp} with idx {idx_cp}")

                # Assigning the cp to the value of the cp_voxel_grid
                cp_voxel_grid[i, j, k] = cp

    return cp_voxel_grid


def normalize_pcd(pcd: torch.tensor) -> (torch.tensor, torch.Tensor, torch.Tensor):
    """
    Normalize the point cloud
    :param pcd Tensor of shape [N, 3]
    :return Tuple of Normalized tensor of shape [N, 3], pcd_min and pcd_max_norm
    """
    pcd_min = pcd.min(dim=0).values
    pcd = pcd - pcd_min
    pcd_max_norm = pcd.norm(dim=1).max()
    pcd = pcd / pcd_max_norm
    return pcd, pcd_min, pcd_max_norm


def inplace_norm(point, min, norm):
    return (point - min) / norm


def draw_quad_faces(a, b, c, transform):
    return [
        [transform(a, b, c), transform(a + 1, b, c), transform(a + 1, b + 1, c), transform(a, b + 1, c)],  # i.j.k
        [transform(a, b, c + 1), transform(a + 1, b, c + 1), transform(a + 1, b + 1, c + 1),
         transform(a, b + 1, c + 1)],  # i.j.km
        [transform(a, b, c), transform(a, b + 1, c), transform(a, b + 1, c + 1), transform(a, b, c + 1)],  # ij.k.
        [transform(a + 1, b, c), transform(a + 1, b + 1, c), transform(a + 1, b + 1, c + 1),
         transform(a + 1, b, c + 1)],  # imj.k.
        [transform(a, b, c), transform(a + 1, b, c), transform(a + 1, b, c + 1), transform(a, b, c + 1)],  # i.jk.
        [transform(a, b + 1, c), transform(a + 1, b + 1, c), transform(a + 1, b + 1, c + 1),
         transform(a, b + 1, c + 1)],  # i.jmk.
    ]


class Voxel:
    def __init__(self,
                 pcd: torch.tensor,
                 sym_planes: torch.tensor,
                 resolution=32):
        self.res = resolution

        self.points, self.min, self.norm = normalize_pcd(pcd)

        self.symmetries_tensor = torch.clone(sym_planes)
        self.symmetries_tensor[:, 3::] = inplace_norm(sym_planes[:, 3::], self.min, self.norm)

        self.symmetries = [
            SymmetryPlane(
                point=self.symmetries_tensor[idx, 3::].numpy(),
                normal=self.symmetries_tensor[idx, 0:3].numpy()
            )
            for idx in range(self.symmetries_tensor.shape[0])
        ]

        self.point_voxels = self.classify(self.points)
        self.grid = self.get_grid(self.point_voxels)
        self.closest_point_grid = compute_closest_points(self.points, self.res)

        self.grid_points = None
        self.grid_visualization_idx = None

    def visualize_voxel_sym_planes(self):
        for idx, sym_plane in enumerate(self.symmetries):
            ps.register_surface_mesh(
                f"VoxelGridPlane_{idx}",
                sym_plane.coords,
                sym_plane.trianglesBase,
                smooth_shade=True
            )

    def visualize_pcd(self):
        ps.register_point_cloud("VoxelGrid PointCloud", self.points.detach().cpu().numpy())

    def visualize_grid(self):
        if self.grid_points is None:
            self.grid_points = torch.cartesian_prod(*([torch.linspace(0.0, 1.0, steps=self.res + 1)] * 3))

        if self.grid_visualization_idx is None:
            self.grid_visualization_idx = self.calculate_idx()

        ps.register_surface_mesh("Voxelization", self.grid_points, self.grid_visualization_idx, smooth_shade=True)

    def visualize(self):
        self.visualize_pcd()
        self.visualize_voxel_sym_planes()
        self.visualize_grid()

    def calculate_idx(self):
        """Calculate the index of the faces used in the voxel visualization."""
        if self.grid_points is None:
            self.grid_points = torch.cartesian_prod(*([torch.linspace(0.0, 1.0, steps=self.res + 1)] * 3))

        faces = []
        n = self.res + 1
        transform_idx = lambda a, b, c: int((n * n * a) + (n * b) + c) % len(self.grid_points)
        cache = []
        for idx in range(self.point_voxels.shape[0]):
            i, j, k = self.point_voxels[idx, :]
            i, j, k = i.item(), j.item(), k.item()
            if (i, j, k) not in cache:
                cache.append((i, j, k))
                faces += draw_quad_faces(i, j, k, transform_idx)

        return faces

    def classify(self, pcd):
        voxel_per_point = torch.zeros_like(pcd)

        voxel_length = (1.0 / self.res)

        voxel_per_point[:, 0] = torch.div(pcd[:, 0], voxel_length, rounding_mode='floor')
        voxel_per_point[:, 1] = torch.div(pcd[:, 1], voxel_length, rounding_mode='floor')
        voxel_per_point[:, 2] = torch.div(pcd[:, 2], voxel_length, rounding_mode='floor')

        # Handle edge case where there are points on the edge of the voxelization i.e idx == self.res
        voxel_per_point[voxel_per_point == self.res] = self.res - 1
        return voxel_per_point

    def get_grid(self, voxel_filled):
        voxel_filled = voxel_filled.long()
        if voxel_filled.min().item() < 0 or voxel_filled.max().item() > self.res - 1:
            raise Exception(f"Malformed voxel_filled tensor. Found index out of range."
                            f"min: {voxel_filled.min().item()}. max:{voxel_filled.max().item()}.")
        voxel_grid = torch.zeros((self.res, self.res, self.res))
        for idx_voxel_filled in range(voxel_filled.shape[0]):
            i, j, k = voxel_filled[idx_voxel_filled, :]
            i, j, k = i.item(), j.item(), k.item()
            voxel_grid[i, j, k] = 1
        return voxel_grid
