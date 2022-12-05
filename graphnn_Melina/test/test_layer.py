import torch
import numpy as np
from torch import nn
import unittest
import sys

sys.path.append("..")

from graphnn.layer import Interaction, sum_splits, unpad_and_cat


class SimpleMessageFunction(nn.Module):
    def forward(self, node, edge_state):
        return node * edge_state


class SimpleInteraction(Interaction):
    def __init__(self, node_size, edge_size):
        super().__init__(node_size, edge_size)

        # Override message function for simplicity
        self.message_function = SimpleMessageFunction()
        # Override state transition function with identity for simplicity
        self.state_transition_function = nn.Sequential()


class TestSimpleInteraction(unittest.TestCase):
    def test_interaction(self):
        nodes = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        edges = torch.tensor([[0, 1], [3, 0], [0, 2], [1, 2]])
        edge_state = torch.tensor([[2.0], [3.0], [1.0], [1.0]])

        expected_output = torch.tensor([[13.0], [4.0], [6.0], [4.0]])

        interaction = SimpleInteraction(1, 1)
        dut_output = interaction(nodes, edges, edge_state)

        abs_errors = torch.abs(expected_output - dut_output).detach().numpy()
        self.assertTrue(np.all(abs_errors < 1e-10))


class TestSumSplits(unittest.TestCase):
    def test_sum_splits(self):
        values = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        splits = torch.tensor([1, 2])
        expected_out = torch.tensor([[1, 2, 3], [11, 13, 15]])
        dut_out = sum_splits(values, splits)

        self.assertTrue(torch.all(torch.eq(expected_out, dut_out)))

    def test_sum_splits_float(self):
        values = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        splits = torch.tensor([1, 2])
        expected_out = torch.tensor([[1.0, 2.0, 3.0], [11.0, 13.0, 15.0]])
        dut_out = sum_splits(values, splits)

        self.assertTrue(torch.all(torch.eq(expected_out, dut_out)))


class TestUnpadAndCat(unittest.TestCase):
    def test_unpad_and_cat(self):
        sequence = torch.arange(0, 15).reshape(3, 5)
        seq_len = torch.tensor([5, 3, 2])
        expected_out = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 10, 11])
        dut_out = unpad_and_cat(sequence, seq_len)

        self.assertTrue(torch.all(torch.eq(expected_out, dut_out)))

    def test_unpad_and_cat_multidim(self):
        sequence = torch.arange(0, 30).reshape(3, 5, 2)
        seq_len = torch.tensor([5, 3, 2])
        expected_out = torch.tensor(
            [
                [0, 1],
                [2, 3],
                [4, 5],
                [6, 7],
                [8, 9],
                [10, 11],
                [12, 13],
                [14, 15],
                [20, 21],
                [22, 23],
            ]
        )
        dut_out = unpad_and_cat(sequence, seq_len)

        self.assertTrue(torch.all(torch.eq(expected_out, dut_out)))


if __name__ == "__main__":
    unittest.main()
