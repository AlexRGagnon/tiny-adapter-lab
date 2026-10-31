import copy
from pathlib import Path
import tempfile
import unittest
import torch
from torch import nn
from tiny_adapter_lab.model import (ModelConfig, TinyLM, LoRALinear, inject_lora, merged_model,
                                    base_fingerprint, save_adapter, load_adapter)
from tiny_adapter_lab.experiment import splits, tensors, train, run

torch.set_num_threads(2)

class AdapterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)

    def test_zero_initialization_is_identity(self):
        base = nn.Linear(8, 6)
        x = torch.randn(3, 8)
        lora = LoRALinear(base, 2, 4.0)
        self.assertTrue(torch.equal(lora(x), base(x)))

    def test_gradients_and_frozen_base(self):
        m = TinyLM()
        fingerprint = base_fingerprint(m)
        inject_lora(m)
        data = tensors(splits(5)['train'], 'copy_first', 'cpu')
        train(m, data, data, steps=4, seed=5, lr=0.01)
        self.assertEqual(base_fingerprint(m), fingerprint)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for n, p in m.named_parameters() if '.lora_' in n))
        self.assertTrue(all(p.grad is None for n, p in m.named_parameters() if '.lora_' not in n))

    def test_merge_equivalence(self):
        m = TinyLM()
        inject_lora(m)
        for n, p in m.named_parameters():
            if n.endswith('.lora_B'):
                with torch.no_grad(): p.normal_(0, 0.05)
        x = torch.tensor([[1, 2, 3, 10]])
        merged = merged_model(m)
        self.assertFalse(any(isinstance(t, LoRALinear) for t in merged.modules()))
        torch.testing.assert_close(m(x), merged(x), atol=1e-6, rtol=1e-5)

    def test_causal_mask(self):
        m = TinyLM().eval()
        a, b = torch.tensor([[1, 2, 3, 10]]), torch.tensor([[1, 2, 8, 10]])
        torch.testing.assert_close(m(a)[:, :2], m(b)[:, :2], atol=1e-6, rtol=1e-5)

    def test_split_disjointness(self):
        s = splits(17)
        self.assertEqual([len(s[k]) for k in ('train', 'validation', 'test')], [800, 100, 100])
        a, b, c = (set(s[k]) for k in ('train', 'validation', 'test'))
        self.assertFalse(a & b or b & c or a & c)
        self.assertEqual(len(a | b | c), 1000)
        self.assertEqual(s, splits(17))

    def test_adapter_save_restore(self):
        a = TinyLM()
        b = copy.deepcopy(a)
        inject_lora(a); inject_lora(b)
        with torch.no_grad(): a.head.lora_B.fill_(0.12)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'adapter.pt'
            save_adapter(a, p); load_adapter(b, p)
            x = torch.tensor([[2, 4, 6, 10]])
            torch.testing.assert_close(a(x), b(x), atol=0, rtol=0)

    def test_reject_wrong_base_and_scale(self):
        a = TinyLM(); b = copy.deepcopy(a)
        inject_lora(a); inject_lora(b, alpha=4.0)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'adapter.pt'; save_adapter(a, p)
            with self.assertRaises(ValueError): load_adapter(b, p)
            c = TinyLM(); inject_lora(c)
            with self.assertRaises(ValueError): load_adapter(c, p)

    def test_reject_corrupt_adapter_without_partial_write(self):
        m = TinyLM(); inject_lora(m)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'adapter.pt'; save_adapter(m, p)
            s = torch.load(p, weights_only=True)
            first = next(iter(s['tensors']))
            s['tensors'][first].fill_(float('nan'))
            torch.save(s, p)
            before = {n: p.clone() for n, p in m.named_parameters()}
            with self.assertRaises(ValueError): load_adapter(m, p)
            for n, p in m.named_parameters():
                self.assertTrue(torch.equal(p, before[n]))

    def test_bad_configuration(self):
        with self.assertRaises(ValueError): ModelConfig(width=31)
        with self.assertRaises(ValueError): ModelConfig(layers=0)
        with self.assertRaises(ValueError): LoRALinear(nn.Linear(4, 4), 8, 1.0)
        m = TinyLM()
        with self.assertRaises(ValueError): inject_lora(m, rank=100)
        self.assertTrue(all(p.requires_grad for p in m.parameters()))
        inject_lora(m)
        with self.assertRaises(ValueError): inject_lora(m)
        with self.assertRaises(ValueError): m(torch.zeros(1, 5, dtype=torch.long))
        with self.assertRaises(ValueError): m(torch.zeros(1, 4))

    def test_end_to_end_artifacts(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / 'run'
            r = run(out, base_steps=2, adapt_steps=2)
            self.assertTrue(r['base_unchanged'] and r['adapter_reload_verified'])
            self.assertLess(r['training']['lora']['trainable_parameters'], r['base_parameters'])
            for f in ('base.pt', 'adapter.pt', 'merged.pt', 'report.json', 'split_ids.json'):
                self.assertTrue((out / f).is_file())
            with self.assertRaises(FileExistsError): run(out, base_steps=1, adapt_steps=1)

if __name__ == '__main__': unittest.main()
