# Local verification

Recorded UTC: 2026-10-31T01:00:01.748035+00:00

Environment: Python 3.13.3, PyTorch 2.11.0+cu128, Windows x64. Unit tests ran on CPU; the recorded full experiment ran on the local CUDA GPU.

## Results

- `python -m unittest discover -s tests -v`: **10/10 tests passed**.
- Includes an end-to-end short CPU experiment, deterministic split checks, causal masking, frozen-base gradients, zero-initialized equivalence, checkpoint validation, and merge/save/reload checks.
- Full local GPU experiment: seed 17, 200 base steps, 200 adaptation steps per method; fixed default hyperparameters.
- Base model: **26,315 parameters**. LoRA trains **1,708 parameters** (6.49% of the base parameter count), while all base parameters stay unchanged.
- Maximum merged/unmerged logit difference: 2.74181366e-06; adapter reload equality check passed.

| Model | Copy-last held-out accuracy | Copy-first held-out accuracy |
|---|---:|---:|
| Frozen base | 100% | 10% |
| LoRA adaptation | 10% | 100% |
| Full fine-tuning | 10% | 100% |

Each test partition contains 100 synthetic triples. The source/target tasks intentionally request different output positions, so adaptation changes behavior; source-task degradation is explicitly reported. These results are **not** a real-language benchmark, proof of improved LLM capability, or a multi-seed estimate. No hyperparameter changes were made after inspecting test scores. Different fixed learning rates mean this is not a controlled efficiency comparison.

Exact machine-generated results and split IDs are committed in `examples/local_run_report.json` and `examples/local_split_ids.json`. Local binary checkpoints are excluded from Git.

## Reproduce

```powershell
$env:PYTHONPATH='src'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
python -m unittest discover -s tests -v
python -m tiny_adapter_lab --out runs/new-local-cuda-run --device cuda --seed 17
```

Use a new output directory. CPU is supported without the CUDA flag. Cross-version/device bitwise reproducibility is not guaranteed. No external model/data downloads, hosted execution, or GitHub Actions were used.
