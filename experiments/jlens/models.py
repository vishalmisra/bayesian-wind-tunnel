"""
Model loading for the J-lens experiments.

The production wind-tunnel checkpoints (logs/bijection_v20_repl/*) were
trained by the train_v256_ddp.py-family scripts, whose TinyGPT variant
differs from src/models/tinygpt.py:

  * separate wq/wk/wv/wo projections (not a fused qkv),
  * RMSNorm parameter named `weight` (not `scale`), block norms norm1/norm2,
  * MLP is nn.Sequential(Linear(d,4d), GELU, Linear(4d,d)) WITH biases,
  * blocks take (x, mask) and return a plain tensor,
  * a persistent tril `mask` buffer, weight-tied head,
  * vocab = 2V (keys 0..V-1, values V..2V-1), max_seq_len = 2L.

SepVocabTinyGPT reproduces that architecture exactly (state-dict compatible).
load_model() dispatches on state-dict keys and returns either variant
wrapped so downstream code sees one interface:

    model.blocks       -> ModuleList; block(i) output residual is TENSOR
    model.emb_module   -> module whose *input* pre-hook sees the embedding
                          residual (we root the autograd graph there)
    model.logits(x)    -> (B, T, vocab) logits
    model.dim, model.vocab_size, model.n_layers, model.n_heads
"""

import math
import re
import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class _MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        # Head mask for ablation parity with src/models/tinygpt.py.
        self.register_buffer("head_mask", torch.ones(n_heads))

    def forward(self, x, mask=None):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            attn = attn.masked_fill(mask[:T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = attn * self.head_mask.view(1, -1, 1, 1)
        return self.wo((attn @ v).transpose(1, 2).contiguous().view(B, T, C))


class _Block(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.norm1 = _RMSNorm(dim)
        self.attn = _MultiHeadAttention(dim, n_heads)
        self.norm2 = _RMSNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim)
        )

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x


class SepVocabTinyGPT(nn.Module):
    """State-dict-compatible reimplementation of the train_v256_ddp.py model."""

    def __init__(
        self,
        vocab_size: int,
        dim: int = 192,
        n_layers: int = 6,
        n_heads: int = 6,
        max_seq_len: int = 38,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.blocks = nn.ModuleList([_Block(dim, n_heads) for _ in range(n_layers)])
        self.norm = _RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        self.register_buffer("mask", torch.tril(torch.ones(max_seq_len, max_seq_len)))

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        h = self.tok_emb(x) + self.pos_emb(pos)[None]
        for block in self.blocks:
            h = block(h, self.mask)
        return self.head(self.norm(h))

    # ---- uniform J-lens interface -------------------------------------
    @property
    def emb_module(self) -> nn.Module:
        """Pre-hook target: blocks[0]'s input IS the embedding residual."""
        return self.blocks[0]

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class LegacyTinyGPTWrapper(nn.Module):
    """Uniform interface around src/models/tinygpt.py's TinyGPT
    (fused-qkv variant; blocks return (hidden, attn))."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    @property
    def blocks(self):
        return self.inner.blocks

    @property
    def emb_module(self) -> nn.Module:
        return self.inner.blocks[0]

    @property
    def dim(self):
        return self.inner.dim

    @property
    def vocab_size(self):
        return self.inner.vocab_size

    @property
    def n_layers(self):
        return self.inner.n_layers

    @property
    def n_heads(self):
        return self.inner.blocks[0].attn.n_heads

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.inner(x)
        return logits

    def forward(self, x):
        return self.logits(x)


class _MLPBlock(nn.Module):
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, mult * dim)
        self.fc2 = nn.Linear(mult * dim, dim)

    def forward(self, x):
        return x + self.fc2(F.gelu(self.fc1(self.ln(x))))


class MLPControl(nn.Module):
    """Attention-free control: a per-position residual MLP stack with the
    same embedding/unembedding scheme as SepVocabTinyGPT (spec section 3:
    "negative control: J-lens should find no low-dim reusable subspace").

    dim=192, n_layers=9 gives ~2.68M params, matching the 2.67M transformer.
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int = 192,
        n_layers: int = 9,
        max_seq_len: int = 38,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = 0
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.blocks = nn.ModuleList([_MLPBlock(dim) for _ in range(n_layers)])
        self.norm = _RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        h = self.tok_emb(x) + self.pos_emb(pos)[None]
        for block in self.blocks:
            h = block(h)
        return self.head(self.norm(h))

    @property
    def emb_module(self) -> nn.Module:
        return self.blocks[0]

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class _RecAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.head_dim = self.d_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.register_buffer("head_mask", torch.ones(n_heads))

    def forward(self, x, mask=None):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.d_head**-0.5
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        alpha = torch.softmax(attn, dim=-1)
        alpha = alpha * self.head_mask.view(1, -1, 1, 1)
        out = (alpha @ v).transpose(1, 2).reshape(B, T, D)
        return self.out_proj(out), alpha


class _RecBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.attn = _RecAttention(d_model, n_heads)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(d_ff, d_model),
            nn.Dropout(0.0),
        )

    def forward(self, x, mask=None):
        attn_out, alpha = self.attn(self.ln1(x), mask)
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x, alpha


class RecurrenceTransformerLens(nn.Module):
    """State-dict-compatible reimplementation of the recurrence /
    loss-horizon models (Model-Selection-BWT recurrence_extrapolation.py,
    learned-PE variant): token_embed (vocab+1, padding), pos_embed(512),
    ln1/ln2 LayerNorm blocks with fused-qkv (biased) attention, GELU ff,
    ln_final + output_proj(n_tokens).

    vocab_size on the uniform interface is n_tokens (the prediction
    classes, Z_p) -- the natural cotangent dims for the J-lens.
    """

    def __init__(
        self,
        vocab_size: int,
        n_tokens: int,
        d_model: int = 192,
        n_layers: int = 6,
        n_heads: int = 6,
        d_ff: int = 768,
    ):
        super().__init__()
        self.n_tokens = n_tokens
        self.dim = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.token_embed = nn.Embedding(vocab_size + 1, d_model, padding_idx=vocab_size)
        self.pos_embed = nn.Embedding(512, d_model)
        self.layers = nn.ModuleList(
            [_RecBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )
        self.ln_final = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, n_tokens)

    def forward(self, tokens):
        B, T = tokens.shape
        mask = torch.triu(
            torch.ones(T, T, device=tokens.device), diagonal=1
        ).bool()
        pos = torch.arange(T, device=tokens.device)
        x = self.token_embed(tokens) + self.pos_embed(pos)[None]
        for layer in self.layers:
            x, _ = layer(x, mask)
        return self.output_proj(self.ln_final(x))

    # ---- uniform J-lens interface -------------------------------------
    @property
    def blocks(self):
        return self.layers

    @property
    def emb_module(self) -> nn.Module:
        return self.layers[0]

    @property
    def vocab_size(self) -> int:
        return self.n_tokens

    @property
    def tok_emb(self) -> nn.Embedding:
        return self.token_embed

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class LSTMLens(nn.Module):
    """State-dict-compatible reimplementation of LSTMBijection
    (experiments/paper1_bijection/train_lstm.py): tok/pos embeddings, a
    stack of LSTM layers, LayerNorm, untied head. The single
    nn.LSTM(num_layers=N) is unrolled into N single-layer modules so the
    per-(layer, position) hidden sequence is hookable -- the J-lens
    capture points. Inter-layer dropout is inference-irrelevant.
    """

    def __init__(self, vocab_size: int, dim: int = 192, n_layers: int = 6,
                 max_seq_len: int = 38):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = 0
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.layers = nn.ModuleList(
            [nn.LSTM(dim, dim, 1, batch_first=True) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, x):
        # cuDNN cannot backprop through RNNs in eval mode; the J-lens
        # differentiates through the forward, so use the native kernel.
        with torch.backends.cudnn.flags(enabled=False):
            pos = torch.arange(x.shape[1], device=x.device)
            h = self.tok_emb(x) + self.pos_emb(pos)[None]
            for layer in self.layers:
                h, _ = layer(h)
            return self.head(self.norm(h))

    @property
    def blocks(self):
        return self.layers

    @property
    def emb_module(self) -> nn.Module:
        return self.layers[0]

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def load_flat_lstm_state(self, sd: dict) -> None:
        """Remap lstm.weight_ih_l{k} etc. onto layers[k].*_l0."""
        remapped = {}
        for k, v in sd.items():
            if k.startswith("lstm."):
                name, layer = k.rsplit("_l", 1)
                name = name.removeprefix("lstm.")
                remapped[f"layers.{layer}.{name}_l0"] = v
            else:
                remapped[k] = v
        missing, unexpected = self.load_state_dict(remapped, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"LSTM state dict mismatch: missing={missing} "
                f"unexpected={unexpected}"
            )


class RecurrentWrapper(nn.Module):
    """Uniform interface around repo model classes whose forward returns
    (logits, loss) and that already expose .blocks (e.g. MambaBijection)."""

    def __init__(self, inner, dim: int, vocab_size: int):
        super().__init__()
        self.inner = inner
        self.dim = dim
        self.vocab_size = vocab_size
        self.n_heads = 0

    @property
    def blocks(self):
        return self.inner.blocks

    @property
    def emb_module(self) -> nn.Module:
        return self.inner.blocks[0]

    @property
    def n_layers(self):
        return len(self.inner.blocks)

    @property
    def tok_emb(self):
        return self.inner.tok_emb

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        out = self.inner(x)
        return out[0] if isinstance(out, tuple) else out

    def forward(self, x):
        return self.logits(x)


def load_model(checkpoint_path: str, device: str = "cpu"):
    """Load either TinyGPT variant from a checkpoint, uniform interface."""
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict") or ck.get("model") or ck
    config = ck.get("config", {}) or {}
    sd = {k.removeprefix("module."): v for k, v in sd.items()}  # DDP

    if "layers.0.weight_ih_l0" in sd:  # LSTMLens-native checkpoint
        vocab_size, dim = sd["tok_emb.weight"].shape
        max_seq_len = sd["pos_emb.weight"].shape[0]
        n_layers = sum(1 for k in sd if k.endswith(".weight_ih_l0"))
        model = LSTMLens(vocab_size=vocab_size, dim=dim, n_layers=n_layers,
                         max_seq_len=max_seq_len)
        model.load_state_dict(sd)
        model.to(device).eval()
        return model

    if "lstm.weight_ih_l0" in sd:
        vocab_size, dim = sd["tok_emb.weight"].shape
        max_seq_len = sd["pos_emb.weight"].shape[0]
        n_layers = sum(1 for k in sd if re.match(r"lstm\.weight_ih_l\d+$", k))
        model = LSTMLens(vocab_size=vocab_size, dim=dim, n_layers=n_layers,
                         max_seq_len=max_seq_len)
        model.load_flat_lstm_state(sd)
        model.to(device).eval()
        return model

    if "blocks.0.A_log" in sd:  # MambaBijection (train_mamba.py, pure torch)
        import importlib.util

        src = PROJECT_ROOT / "experiments" / "paper1_bijection" / "train_mamba.py"
        train_dir = str(src.parent)
        if train_dir not in sys.path:
            sys.path.insert(0, train_dir)
        spec = importlib.util.spec_from_file_location("wt_train_mamba", src)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        vocab_size, dim = sd["tok_emb.weight"].shape
        n_layers = len({k.split(".")[1] for k in sd if k.startswith("blocks.")})
        inner = mod.MambaBijection(
            vocab_size, d_model=dim, n_layers=n_layers,
            max_seq_len=sd["pos_emb.weight"].shape[0],
        )
        inner.load_state_dict(sd)
        inner.to(device).eval()
        return RecurrentWrapper(inner, dim=dim, vocab_size=vocab_size).to(device)

    if "output_proj.weight" in sd and "token_embed.weight" in sd:
        # recurrence_bwt.py names the attention output projection 'out';
        # recurrence_extrapolation.py names it 'out_proj'. Normalize.
        sd = {k.replace(".attn.out.", ".attn.out_proj."): v for k, v in sd.items()}
        n_tokens, d_model = sd["output_proj.weight"].shape
        vocab_size = sd["token_embed.weight"].shape[0] - 1  # padding row
        n_layers = sum(1 for k in sd if k.endswith(".ln1.weight"))
        d_ff = sd["layers.0.ff.0.weight"].shape[0]
        model = RecurrenceTransformerLens(
            vocab_size=vocab_size,
            n_tokens=n_tokens,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=int(config.get("n_heads", 6)),
            d_ff=d_ff,
        )
        missing, unexpected = model.load_state_dict(sd, strict=False)
        real_missing = [k for k in missing if "head_mask" not in k]
        if real_missing or unexpected:
            raise RuntimeError(
                f"state dict mismatch: missing={real_missing} unexpected={unexpected}"
            )
        model.to(device).eval()
        return model

    if any(".fc1." in k for k in sd) and not any("attn" in k for k in sd):
        vocab_size, dim = sd["tok_emb.weight"].shape
        max_seq_len = sd["pos_emb.weight"].shape[0]
        n_layers = sum(1 for k in sd if k.endswith(".fc1.weight"))
        model = MLPControl(
            vocab_size=vocab_size, dim=dim, n_layers=n_layers, max_seq_len=max_seq_len
        )
        model.load_state_dict(sd, strict=False)
        model.to(device).eval()
        return model

    if any(".attn.wq." in k for k in sd):
        vocab_size, dim = sd["tok_emb.weight"].shape
        max_seq_len = sd["pos_emb.weight"].shape[0]
        n_layers = sum(1 for k in sd if k.endswith(".norm1.weight"))
        n_heads = int(config.get("n_heads", 6))
        model = SepVocabTinyGPT(
            vocab_size=vocab_size,
            dim=dim,
            n_layers=n_layers,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
        )
        # Untied checkpoints (DA-1 control) carry a head distinct from the
        # embedding; break the tie before loading or the shared parameter
        # gets overwritten twice.
        if "head.weight" in sd and not torch.equal(
            sd["head.weight"], sd["tok_emb.weight"]
        ):
            model.head = nn.Linear(dim, vocab_size, bias=False)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        # head.weight is tied; head_mask buffers are ours. Anything else
        # missing means the architecture guess is wrong -- fail loudly.
        real_missing = [
            k for k in missing if k != "head.weight" and "head_mask" not in k
        ]
        if real_missing or unexpected:
            raise RuntimeError(
                f"state dict mismatch: missing={real_missing} unexpected={unexpected}"
            )
        model.to(device).eval()
        return model

    from src.models.tinygpt import load_tinygpt

    return LegacyTinyGPTWrapper(load_tinygpt(checkpoint_path, device=device))
