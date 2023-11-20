import os
import unittest
import torch

from dataset.voxel_dataset import VoxelDataset
from setup.setup_voxel_dataset.voxel import Voxel, normalize_pcd
from model.prsnet.sym_loss import apply_symmetry
from dotenv import load_dotenv


class TestVoxelClass(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_points = torch.Tensor([
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
        ])

        cls.test_syms = torch.tensor([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        ])
        ambiente = "local"
        env_path = f"envs/.{ambiente}.env"
        env = load_dotenv(env_path)
        if not env:
            raise ValueError(f"Failed to load env, path used is {env_path}")
        cls.dataset = VoxelDataset(os.environ.get("VOXEL_DATASET_ROOT"), sample_size=None)

    def test_voxel_copies_not_modifies(self):
        original_test_points = torch.clone(self.test_points)
        original_test_syms = torch.clone(self.test_syms)
        test_voxel = Voxel(
            pcd=self.test_points,
            sym_planes=self.test_syms,
            env="local",
            resolution=32
        )
        self.assertTrue(
            torch.equal(
                original_test_syms, self.test_syms
            ),
            "Original syms were modified!"
        )
        self.assertTrue(
            torch.equal(
                original_test_points, self.test_points
            ),
            "Original points were modified!"
        )

    def test_voxel_constructor(self):
        test_voxel = Voxel(
            pcd=self.test_points,
            sym_planes=self.test_syms,
            env="local",
            resolution=32
        )
        for idx in range(self.test_syms.shape[0]):
            normal, point = self.test_syms[idx, 0:3], self.test_syms[idx, 3::]
            point = (point - test_voxel.min) / test_voxel.norm
            d = -torch.dot(normal, point)
            original_points = test_voxel.points.sort(dim=0).values
            reflected_points = apply_symmetry(
                points=test_voxel.points,
                normal=normal,
                d=d
            ).sort(dim=0).values
            self.assertTrue(
                torch.equal(
                    reflected_points, original_points
                ),
                msg=f"reflected:\n{reflected_points}\n"
                    f"expected:\n{original_points}\n"
            )

    def test_symmetries_between_points(self):
        points, voxel, cp, syms = self.dataset[0]
        for idx in range(syms.shape[0]):
            a_sym = syms[idx, :]
            normal, point = a_sym[0:3], a_sym[3::]
            d = -torch.dot(normal, point)
            original_points = points.sort(dim=0).values
            reflected_points = apply_symmetry(
                points=points,
                normal=normal,
                d=d
            ).sort(dim=0).values
            boolean_mask = (torch.norm(reflected_points - original_points, dim=1) > 1e-6).sum()
            discrepancy = 1 - boolean_mask / (original_points.shape[0] * original_points.shape[1])
            self.assertGreater(
                discrepancy,
                0.66,
                msg=f"reflected:\n{reflected_points}\n"
                    f"expected:\n{original_points}\n"
                    f"boolean mask: \n {boolean_mask}"
            )
