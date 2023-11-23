import unittest

import torch

from model.prsnet.metrics import phc


class TestMetrics(unittest.TestCase):
    def test_correct_shapes_io(self):
        # B x N x (6+1), B: Batch N: Amount of symmetries predicted per input
        # (6+1): normal, point and confindence
        mock_y_pred = torch.ones((10, 8, 7))
        mock_y_test = torch.ones((10, 8, 6))

        # Output is B a mask of true and false
        phc_return = phc(mock_y_pred, mock_y_test)

        self.assertEqual(phc_return, 1)

    def test_correct_matching_single_plane(self):
        mock_y_pred = torch.tensor([
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
            ]
        ])
        mock_y_test = torch.tensor([
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            ]
        ])

        phc_return = phc(mock_y_pred, mock_y_test)

        self.assertEqual(phc_return, 1)

    def test_correct_matching_multiple_plane(self):
        mock_y_pred = torch.tensor([
            [
                [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8],
            ]
        ])
        mock_y_test = torch.tensor([
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            ]
        ])

        phc_return = phc(mock_y_pred, mock_y_test)

        self.assertEqual(phc_return, 0)

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

        phc_return = phc(mock_y_pred, mock_y_test)

        self.assertEqual(phc_return, 0.5)

    def test_correct_matching_multiple_label_planes(self):
        mock_y_pred = torch.tensor([
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8],
            ]
        ])
        mock_y_test = torch.tensor([
            [
                [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ])

        phc_return = phc(mock_y_pred, mock_y_test)

        self.assertEqual(phc_return, 1)
