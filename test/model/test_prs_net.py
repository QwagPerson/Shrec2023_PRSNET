import unittest
from model.prsnet.prs_net import PRSNet


class TestPRSNet(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_model = PRSNet(amount_of_heads=1)

    def test_model_constructor(self):
        many_heads_test_model = PRSNet(amount_of_heads=5)

        self.assertEqual(len(self.test_model.heads), 1)
        self.assertEqual(len(many_heads_test_model.heads), 5)

    def test_parameter_validation_model_constructor(self):
        with self.assertRaises(ValueError):
            PRSNet(amount_of_heads=-1)

        with self.assertRaises(ValueError):
            PRSNet(amount_of_heads=0)

    def test_dimension_input_output(self):
        self.assertEqual(True, True)  # add assertion here
