import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from experiments.production_eval.tied_sft_control_v3 import force_equal_original_tying, assert_tied, compare_probes


class TinyTyingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.embed_tokens = torch.nn.Embedding(2, 2)
        self.lm_head = torch.nn.Linear(2, 2, bias=False)
        self.lm_head.weight.data.copy_(self.model.embed_tokens.weight.data)
        self.config = SimpleNamespace(tie_word_embeddings=True)
        self.all_tied_weights_keys = {}

    def tie_weights(self):
        self.lm_head.weight = self.model.embed_tokens.weight

    def get_expanded_tied_weights_keys(self, all_submodels):
        return {'lm_head.weight': 'model.embed_tokens.weight'}


class TyingControlTests(unittest.TestCase):
    def test_equal_weights_share_storage_and_remove_duplicate_optimizer_parameter(self):
        model = TinyTyingModel()
        before = model.lm_head(model.model.embed_tokens(torch.tensor([0, 1]))).detach()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 8)
        with patch('experiments.production_eval.tied_sft_control_v3.TIED_PARAMETERS', 4):
            force_equal_original_tying(model)
            assert_tied(model)
        after = model.lm_head(model.model.embed_tokens(torch.tensor([0, 1]))).detach()
        self.assertTrue(torch.equal(before, after))
        self.assertEqual(model.lm_head.weight.data_ptr(), model.model.embed_tokens.weight.data_ptr())
        self.assertEqual(sum(p.numel() for p in model.parameters()), 4)
        self.assertEqual(model.all_tied_weights_keys, {'lm_head.weight': 'model.embed_tokens.weight'})

    def test_trained_unequal_matrices_cannot_be_silently_tied(self):
        model = TinyTyingModel()
        model.lm_head.weight.data.add_(1)
        before = model.lm_head.weight.clone()
        with self.assertRaises(ValueError):
            force_equal_original_tying(model)
        self.assertTrue(torch.equal(before, model.lm_head.weight))
        self.assertNotEqual(model.lm_head.weight.data_ptr(), model.model.embed_tokens.weight.data_ptr())

    def test_forward_comparison_rejects_logit_or_loss_differences(self):
        before = [(torch.tensor([1., 2.]), 0.5)]
        self.assertEqual(compare_probes(before, before)[0]['max_abs_logit_difference'], 0.)
        for after in [[(torch.tensor([1., 3.]), 0.5)], [(torch.tensor([1., 2.]), 0.6)]]:
            with self.assertRaises(ValueError):
                compare_probes(before, after)


if __name__ == '__main__':
    unittest.main()
