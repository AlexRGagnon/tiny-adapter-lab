# Tiny Adapter Lab

A runnable PyTorch experiment for training a small causal transformer and adapting it with LoRA. The implementation includes adapter injection, frozen-weight verification, checkpoint compatibility checks, adapter merging, and a full-fine-tuning reference.

**Scope:** a synthetic, one-token prediction experiment, not fine-tuning a downloaded foundation model or a claim of large-language-model performance. This project is deliberately small enough to inspect and reproduce locally.

## Run

Python 3.11+ and PyTorch 2.6+ (tested versions are recorded in `VALIDATION.md`). CPU is the default. Install the appropriate PyTorch wheel for your machine before using CUDA.

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
python -m tiny_adapter_lab --out runs/cpu-demo
```

Existing PyTorch installations can run without a package install:

```powershell
$env:PYTHONPATH='src'
python -m unittest discover -s tests -v
python -m tiny_adapter_lab --out runs/cuda-demo --device cuda
```

Run directories must not already exist. No data/model downloads or network calls occur during an experiment. CUDA runs use the selected machine, not a cloud API.

## Experiment

The vocabulary is ten digit tokens plus a prompt terminator. All 1,000 ordered digit triples are split into 800 training, 100 validation, and 100 test examples using a recorded seed. The prompt `123>` requests one output token. Base pretraining learns to copy the last digit (`3`); adaptation learns to copy the first (`1`). The model has two causal-attention blocks, width 32, and four attention heads.

The same example IDs remain in the same partition for both tasks. Checkpoints are selected by validation loss. Test metrics are computed only after selection. The report compares a frozen base model, rank-4 LoRA, and full fine-tuning on both source and target tasks, so loss of source behavior is visible rather than hidden.

LoRA wraps attention QKV/projection layers and the output head with `W(x) + (alpha/r) B A x`. The base weights are frozen, `B` starts at zero, and a SHA-256 fingerprint must remain unchanged. Rank/scale, module keys, tensor shapes/dtypes, finite values, and the base fingerprint are checked before loading adapters. `torch.load(..., weights_only=True)` is used; still load only trusted checkpoint files.

## Artifacts and interpretation

Each run saves `base.pt`, `adapter.pt`, `merged.pt`, `split_ids.json`, and `report.json`. Reports include seed, versions, split digest, trainable parameter counts, validation history, selected steps, test loss/accuracy, base-weight invariance, and merge/reload checks. Base/merged checkpoints contain `config` and `state_dict` keys and can be reconstructed with `TinyLM(ModelConfig(**checkpoint['config']))`.

The different training modes use fixed but different learning rates. A single run is not a controlled timing benchmark or evidence of general NLP ability. Training timings include validation and snapshot selection. Bitwise determinism across PyTorch versions/devices is not promised. No hyperparameters were selected using test results. See `examples/local_run_report.json` for an actual recorded run; binary weights and local run directories are not committed.

## Design references

- Hu et al., LoRA: https://arxiv.org/abs/2106.09685
- PyTorch causal attention API: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html

All testing and training run locally. No Actions workflows, paid APIs, or hosted computation. MIT licensed.
