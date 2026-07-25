import unittest

import torch
import torch.nn as nn

from model_utils import _load_without_reinitializing
from pruning_utils import (
    _smallest_mask,
    _validate_pruning_args,
    SparseGPT_LlaMA,
    SparseGPT_OPT,
)
from quant import Quantizer


class PruningUtilityTests(unittest.TestCase):
    def test_smallest_mask_has_exact_cardinality(self):
        scores = torch.tensor([[1.0, 1.0], [2.0, 3.0]])
        self.assertEqual(_smallest_mask(scores, 0.0).sum().item(), 0)
        self.assertEqual(_smallest_mask(scores, 0.5).sum().item(), 2)
        self.assertEqual(_smallest_mask(scores, 1.0).sum().item(), 4)

    def test_invalid_pruning_parameters_are_rejected(self):
        for args in [(-0.1, 0, 0), (1.1, 0, 0), (0.5, 3, 2)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                _validate_pruning_args(*args)

    def test_sparsegpt_supports_cpu_and_endpoint_sparsities(self):
        for cls in (SparseGPT_OPT, SparseGPT_LlaMA):
            with self.subTest(cls=cls.__name__):
                layer = nn.Linear(4, 3, bias=False)
                original = layer.weight.detach().clone()
                pruner = cls(layer)
                pruner.H.copy_(torch.eye(4))
                pruner.fasterprune(0.0, blocksize=4)
                torch.testing.assert_close(layer.weight, original)

                pruner = cls(layer)
                pruner.H.copy_(torch.eye(4))
                pruner.fasterprune(1.0, blocksize=4)
                self.assertEqual(torch.count_nonzero(layer.weight).item(), 0)

    def test_nm_pruning_handles_a_short_final_group(self):
        layer = nn.Linear(5, 2, bias=False)
        pruner = SparseGPT_OPT(layer)
        pruner.H.copy_(torch.eye(5))
        pruner.fasterprune(0.0, prunen=2, prunem=4, blocksize=5)
        zeros_per_row = (layer.weight == 0).sum(dim=1)
        self.assertTrue(torch.equal(zeros_per_row, torch.tensor([3, 3])))

    def test_model_loading_restores_global_initializers_on_failure(self):
        original = torch.nn.init.uniform_

        class FailingLoader:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                self.assertIsNot(torch.nn.init.uniform_, original)
                raise RuntimeError("expected")

        with self.assertRaisesRegex(RuntimeError, "expected"):
            _load_without_reinitializing(FailingLoader, "unused")
        self.assertIs(torch.nn.init.uniform_, original)

    def test_quantizer_rejects_invalid_configuration(self):
        quantizer = Quantizer()
        with self.assertRaises(ValueError):
            quantizer.configure(0)
        with self.assertRaises(ValueError):
            quantizer.configure(4, grouprows=0)


if __name__ == "__main__":
    unittest.main()
