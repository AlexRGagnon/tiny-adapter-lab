"""Small causal transformer and explicit, mergeable LoRA linear layers."""
from __future__ import annotations
from dataclasses import dataclass
import copy
import hashlib
import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 11
    context: int = 4
    width: int = 32
    heads: int = 4
    layers: int = 2

    def __post_init__(self):
        for name, value in vars(self).items():
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.width % self.heads:
            raise ValueError('width must be divisible by heads')

class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.heads = cfg.heads
        self.qkv = nn.Linear(cfg.width, 3 * cfg.width)
        self.proj = nn.Linear(cfg.width, cfg.width)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (a.reshape(b, t, self.heads, d // self.heads).transpose(1, 2) for a in (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        return self.proj(out.transpose(1, 2).contiguous().reshape(b, t, d))

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(cfg.width), nn.LayerNorm(cfg.width)
        self.attn = Attention(cfg)
        self.mlp = nn.Sequential(nn.Linear(cfg.width, 4 * cfg.width), nn.GELU(), nn.Linear(4 * cfg.width, cfg.width))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))

class TinyLM(nn.Module):
    def __init__(self, cfg: ModelConfig = ModelConfig()):
        super().__init__()
        self.config = cfg
        self.token = nn.Embedding(cfg.vocab_size, cfg.width)
        self.position = nn.Embedding(cfg.context, cfg.width)
        self.blocks = nn.Sequential(*(Block(cfg) for _ in range(cfg.layers)))
        self.norm = nn.LayerNorm(cfg.width)
        self.head = nn.Linear(cfg.width, cfg.vocab_size)

    def forward(self, ids):
        if ids.ndim != 2 or not 0 < ids.shape[1] <= self.config.context:
            raise ValueError('ids must be [batch, 1..context]')
        if ids.dtype != torch.long:
            raise ValueError('ids must be int64 token IDs')
        x = self.token(ids) + self.position(torch.arange(ids.shape[1], device=ids.device))
        return self.head(self.norm(self.blocks(x)))

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        if type(rank) is not int or rank < 1 or rank > min(base.in_features, base.out_features):
            raise ValueError('rank outside linear dimensions')
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError('alpha must be finite and positive')
        self.base = base.requires_grad_(False)
        self.rank, self.alpha, self.scale = rank, alpha, alpha / rank
        self.lora_A = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.lora_B = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scale

    @torch.no_grad()
    def merged(self):
        result = copy.deepcopy(self.base)
        result.weight.add_((self.lora_B @ self.lora_A) * self.scale)
        return result

def inject_lora(model: nn.Module, rank: int = 4, alpha: float = 8.0) -> list[str]:
    """Freeze all base weights; wrap attention projections and the prediction head."""
    if any(isinstance(m, LoRALinear) for m in model.modules()):
        raise ValueError('model already contains adapters')
    selected = [(name, module) for name, module in model.named_modules()
                if isinstance(module, nn.Linear) and name.split('.')[-1] in {'qkv', 'proj', 'head'}]
    if not selected:
        raise ValueError('no supported target layers')
    # Validate every replacement before mutating the model.
    if type(rank) is not int or rank < 1 or any(rank > min(m.in_features, m.out_features) for _, m in selected):
        raise ValueError('invalid adapter rank')
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError('invalid alpha')
    model.requires_grad_(False)
    for name, module in selected:
        parent_name, _, attr = name.rpartition('.')
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, attr, LoRALinear(module, rank, alpha))
    return [name for name, _ in selected]

def merged_model(model: nn.Module) -> nn.Module:
    result = copy.deepcopy(model)
    for name, module in list(result.named_modules()):
        if isinstance(module, LoRALinear):
            parent_name, _, attr = name.rpartition('.')
            parent = result.get_submodule(parent_name) if parent_name else result
            setattr(parent, attr, module.merged())
    return result

def base_fingerprint(model: nn.Module) -> str:
    h = hashlib.sha256()
    items = []
    for name, param in model.named_parameters():
        if name.endswith(('.lora_A', '.lora_B')):
            continue
        items.append((name.replace('.base.', '.'), param))
    for name, param in sorted(items):
        h.update(f'{name}|{param.dtype}|{tuple(param.shape)}\n'.encode())
        raw = param.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        h.update(bytes(raw.tolist()))
    return h.hexdigest()

def adapter_tensors(model):
    return {name: param for name, param in model.named_parameters() if name.endswith(('.lora_A', '.lora_B'))}

def save_adapter(model: nn.Module, path: Path) -> None:
    adapters = adapter_tensors(model)
    if not adapters:
        raise ValueError('model has no adapters')
    modules = {n: {'rank': m.rank, 'alpha': m.alpha} for n, m in model.named_modules() if isinstance(m, LoRALinear)}
    torch.save({'format': 1, 'base_sha256': base_fingerprint(model), 'modules': modules,
                'tensors': {n: p.detach().cpu() for n, p in adapters.items()}}, path)

def load_adapter(model: nn.Module, path: Path) -> None:
    state = torch.load(path, map_location='cpu', weights_only=True)
    expected_modules = {n: {'rank': m.rank, 'alpha': m.alpha} for n, m in model.named_modules() if isinstance(m, LoRALinear)}
    if not expected_modules or not isinstance(state, dict) or state.get('format') != 1:
        raise ValueError('unsupported adapter format or target')
    if state.get('base_sha256') != base_fingerprint(model) or state.get('modules') != expected_modules:
        raise ValueError('adapter does not match base checkpoint/configuration')
    target, saved = adapter_tensors(model), state.get('tensors')
    if not isinstance(saved, dict) or target.keys() != saved.keys():
        raise ValueError('adapter keys mismatch')
    for name, p in target.items():
        s = saved[name]
        if not isinstance(s, torch.Tensor) or s.shape != p.shape or s.dtype != p.dtype or not torch.isfinite(s).all():
            raise ValueError(f'invalid adapter tensor: {name}')
    with torch.no_grad():
        for name, p in target.items():
            p.copy_(saved[name].to(p.device))
