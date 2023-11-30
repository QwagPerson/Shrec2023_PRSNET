import unittest
from model.prsnet.prs_net import PRSNet


class TestPRSNet(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_model = PRSNet(amount_of_heads=1, input_resolution=32)

    def test_model_constructor(self):
        many_heads_test_model = PRSNet(amount_of_heads=5, input_resolution=32)

        self.assertEqual(len(self.test_model.heads), 1)
        self.assertEqual(len(many_heads_test_model.heads), 5)

    def test_parameter_validation_model_constructor(self):
        with self.assertRaises(ValueError):
            PRSNet(amount_of_heads=-1, input_resolution=32)

        with self.assertRaises(ValueError):
            PRSNet(amount_of_heads=0, input_resolution=32)


