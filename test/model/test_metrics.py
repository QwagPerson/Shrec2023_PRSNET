import unittest

import torch

from model.prsnet.metrics import get_phc


class TestMetrics(unittest.TestCase):
    def test_correct_shapes_io(self):
        mock_y_pred = torch.ones((10, 8, 4))
        mock_y_test = torch.ones((10, 8, 6))

        mock_sample_points = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])

        phc_return = get_phc((
            None, None, mock_sample_points, None, None, mock_y_test
        ), mock_y_pred)

        self.assertEqual(phc_return, 0.0)

    def test_correct_matching_single_plane(self):
        mock_y_pred = torch.tensor([
            [
                [1.0, 0.0, 0.0, 0.0]
            ]
        ])
        mock_y_test = torch.tensor([
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            ]
        ])

        mock_sample_points = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])

        phc_return = get_phc((
            None, None, mock_sample_points, None, None, mock_y_test
        ), mock_y_pred)

        self.assertEqual(phc_return, 1)

    def test_correct_matching_multiple_plane(self):
        mock_y_pred = torch.tensor([
            [
                [1.0, 1.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ]
        ])
        mock_y_test = torch.tensor([
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            ]
        ])

        mock_sample_points = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])

        phc_return = get_phc((
            None, None, mock_sample_points, None, None, mock_y_test
        ), mock_y_pred)

        self.assertEqual(phc_return, 0.0)

    def test_correct_matching_batch_size_2(self):
        mock_y_pred = torch.tensor([
            [
                [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            ],
            [
                [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            ]
        ])
        mock_y_test = torch.tensor([
            [
                [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            ],
            [
                [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            ]
        ])

        mock_sample_points = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])

        phc_return = get_phc((
            None, None, mock_sample_points, None, None, mock_y_test
        ), mock_y_pred)

        self.assertEqual(phc_return, 0.5)

    def test_correct_matching_multiple_label_planes(self):
        mock_y_pred = torch.tensor([
            [
                [1.0, 0.0, 0.0, 0.0],
            ]
        ])
        mock_y_test = torch.tensor([
            [
                [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ])
        mock_sample_points = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])

        phc_return = get_phc((
            None, None, mock_sample_points, None, None, mock_y_test
        ), mock_y_pred)

        self.assertEqual(phc_return, 1)
