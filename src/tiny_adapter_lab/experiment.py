"""Train a tiny base model, then compare LoRA and full fine-tuning locally."""
from __future__ import annotations
import argparse
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import time
import torch
from torch.nn import functional as F
from .model import TinyLM, ModelConfig, inject_lora, base_fingerprint, merged_model, save_adapter, load_adapter


def splits(seed: int):
    rows = [(a, b, c) for a in range(10) for b in range(10) for c in range(10)]
    random.Random(seed).shuffle(rows)
    return {'train': rows[:800], 'validation': rows[800:900], 'test': rows[900:]}


def tensors(rows, task: str, device):
    if task not in {'copy_last', 'copy_first'}:
        raise ValueError('unknown task')
    x = torch.tensor([list(r) + [10] for r in rows], dtype=torch.long, device=device)
    y = torch.tensor([r[-1 if task == 'copy_last' else 0] for r in rows], dtype=torch.long, device=device)
    return x, y


@torch.no_grad()
def evaluate(model, data):
    model.eval()
    x, y = data
    logits = model(x)[:, -1]
    loss = F.cross_entropy(logits, y).item()
    return {'n': len(y), 'loss_nats': loss, 'accuracy': (logits.argmax(-1) == y).float().mean().item()}


def train(model, train_data, validation_data, *, steps, seed, lr, batch_size=64):
    if type(steps) is not int or steps < 1 or type(batch_size) is not int or batch_size < 1:
        raise ValueError('steps/batch_size must be positive integers')
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError('learning rate must be finite and positive')
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError('no trainable parameters')
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    rng = torch.Generator().manual_seed(seed)
    x, y = train_data
    best_loss = math.inf
    best_state = None
    best_step = 0
    history = []
    started = time.perf_counter()
    for step in range(1, steps + 1):
        model.train()
        indices = torch.randint(len(y), (batch_size,), generator=rng).to(x.device)
        loss = F.cross_entropy(model(x[indices])[:, -1], y[indices])
        if not torch.isfinite(loss):
            raise RuntimeError('non-finite training loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        if step % 20 == 0 or step == steps:
            metrics = evaluate(model, validation_data)
            history.append({'step': step, **metrics})
            if metrics['loss_nats'] < best_loss:
                best_loss, best_step = metrics['loss_nats'], step
                best_state = {n: p.detach().cpu().clone() for n, p in model.state_dict().items()}
    model.load_state_dict(best_state)
    return {'best_step': best_step, 'validation_history': history,
            'elapsed_seconds': time.perf_counter() - started, 'steps': steps, 'lr': lr,
            'trainable_parameters': sum(p.numel() for p in params)}


def run(out: Path, *, seed=17, base_steps=200, adapt_steps=200, device='cpu'):
    for name, value in {'seed': seed, 'base_steps': base_steps, 'adapt_steps': adapt_steps}.items():
        if type(value) is not int or value < (0 if name == 'seed' else 1):
            raise ValueError(f'invalid {name}')
    if device not in {'cpu', 'cuda'} or (device == 'cuda' and not torch.cuda.is_available()):
        raise ValueError('requested device is unavailable')
    out = Path(out)
    if out.exists():
        raise FileExistsError('output directory already exists; choose a new path')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    if device == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    cfg = ModelConfig()
    partitions = splits(seed)
    split_ids = {name: [''.join(map(str, r)) for r in rows] for name, rows in partitions.items()}
    split_digest = hashlib.sha256(json.dumps(split_ids, sort_keys=True).encode()).hexdigest()
    source = {name: tensors(rows, 'copy_last', device) for name, rows in partitions.items()}
    target = {name: tensors(rows, 'copy_first', device) for name, rows in partitions.items()}
    base = TinyLM(cfg).to(device)
    print('Training base copy-last task', flush=True)
    base_log = train(base, source['train'], source['validation'], steps=base_steps, seed=seed, lr=0.003)
    digest = base_fingerprint(base)
    adapter = copy.deepcopy(base)
    layers = inject_lora(adapter, rank=4, alpha=8.0)
    print('Training low-rank adapter on copy-first task', flush=True)
    adapter_log = train(adapter, target['train'], target['validation'], steps=adapt_steps, seed=seed + 1, lr=0.005)
    if base_fingerprint(adapter) != digest:
        raise AssertionError('frozen base parameters changed')
    full = copy.deepcopy(base).requires_grad_(True)
    print('Training full-fine-tuning reference', flush=True)
    full_log = train(full, target['train'], target['validation'], steps=adapt_steps, seed=seed + 1, lr=0.001)
    # Test data is accessed for scoring only after all checkpoint selection is complete.
    models = {'frozen_base': base, 'lora': adapter, 'full_finetune': full}
    metrics = {name: {'source_test': evaluate(m, source['test']), 'target_test': evaluate(m, target['test'])}
               for name, m in models.items()}
    merged = merged_model(adapter).eval()
    with torch.no_grad():
        merge_error = (merged(target['test'][0]) - adapter(target['test'][0])).abs().max().item()
    if merge_error > 1e-4:
        raise AssertionError(f'merge numerical disagreement: {merge_error}')
    out.mkdir(parents=True, exist_ok=False)
    torch.save({'config': asdict(cfg), 'state_dict': {n: p.detach().cpu() for n, p in base.state_dict().items()}}, out / 'base.pt')
    save_adapter(adapter, out / 'adapter.pt')
    restored = copy.deepcopy(base)
    inject_lora(restored, rank=4, alpha=8.0)
    load_adapter(restored, out / 'adapter.pt')
    with torch.no_grad():
        if not torch.allclose(restored(target['test'][0]), adapter(target['test'][0]), atol=1e-6):
            raise AssertionError('adapter save/load disagreement')
    torch.save({'config': asdict(cfg), 'state_dict': {n: p.detach().cpu() for n, p in merged.state_dict().items()}}, out / 'merged.pt')
    (out / 'split_ids.json').write_text(json.dumps(split_ids, indent=2) + '\n', encoding='utf-8')
    report = {'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
              'task': 'synthetic digit triples: base copy-last -> adaptation copy-first',
              'seed': seed, 'device': device, 'python': platform.python_version(), 'torch': str(torch.__version__),
              'config': asdict(cfg), 'base_sha256': digest, 'split_sha256': split_digest,
              'split_counts': {n: len(r) for n, r in partitions.items()},
              'base_parameters': sum(p.numel() for p in base.parameters()), 'adapter_layers': layers,
              'training': {'base': base_log, 'lora': adapter_log, 'full_finetune': full_log},
              'metrics': metrics, 'merged_max_abs_error': merge_error,
              'base_unchanged': True, 'adapter_reload_verified': True,
              'limitations': ['Single-seed synthetic task, not a language-model capability benchmark.',
                              'Timings include validation/checkpoint selection, not controlled speed comparisons.',
                              'Different fixed learning rates; no claim of optimal hyperparameter comparison.']}
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(out), 'metrics': metrics, 'base_unchanged': True}, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--base-steps', type=int, default=200)
    parser.add_argument('--adapt-steps', type=int, default=200)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    try:
        run(args.out, seed=args.seed, base_steps=args.base_steps, adapt_steps=args.adapt_steps, device=args.device)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f'error: {exc}\n')

if __name__ == '__main__':
    main()
