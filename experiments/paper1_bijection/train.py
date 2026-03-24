#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bijection ICL experiments with a compact GPT (PyTorch), optional DDP.

Datasets:
  - FixedDict
  - ChangingDict
  - MixedDict
  - MixedSeparateVocabs

Analyses:
  - §3 entropy experiments: --entropy_eval {with_replacement,without_replacement}
  - Per-layer posterior KL(model||Bayes): --kl_eval {with_replacement,without_replacement}
  - Optional raw attention dump: --dump_attn N

Training niceties:
  - DDP support
  - Linear warmup + cosine LR decay
  - Gradient clipping

Semantics flag:
  - --p_is_changing (default True): in mixed datasets, p == probability of ChangingDict
    If set False, p == probability of FixedDict (legacy behavior).

Examples
--------
# Train (single GPU)
python bijection_minGPT.py --exp mixed --p 0.1 --p_is_changing True --V 10 --L 10 \
  --n_train 200000 --n_val 10000 --batch_size 128 --layers 6 --max_steps 30000 --lr 1e-3

# Train (8 GPUs)
torchrun --standalone --nproc_per_node=8 bijection_minGPT.py --exp mixed --p 0.1 --p_is_changing True \
  --V 10 --L 10 --n_train 200000 --n_val 10000 --batch_size 128 --layers 6 --max_steps 30000 --lr 1e-3

# Evaluate
python bijection_minGPT.py --eval_only --ckpt ckpt.pt --V 10 --L 10

# Entropy experiments
python bijection_minGPT.py --V 10 --L 10 --ckpt ckpt.pt --entropy_eval with_replacement \
  --entropy_samples 10000 --entropy_csv sawtooth.csv
python bijection_minGPT.py --V 10 --L 10 --ckpt ckpt.pt --entropy_eval without_replacement \
  --entropy_samples 10000 --entropy_csv smooth.csv

# Per-layer KL
python bijection_minGPT.py --V 10 --L 10 --ckpt ckpt.pt --kl_eval with_replacement \
  --kl_samples 2000 --kl_csv kl_layers_sawtooth.csv

# Attention dump (small batch)
python bijection_minGPT.py --V 10 --L 10 --ckpt ckpt.pt --dump_attn 32 --attn_npz attn_dump.npz
"""

import os, math, random, json, argparse, csv, time
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
except Exception:  # pragma: no cover
    dist, DDP = None, None

IGNORE_INDEX = -100
LOG2E = 1.0 / math.log(2.0)

# --------------------------
# Data generation utilities
# --------------------------

def sample_perm(V: int) -> List[int]:
    arr = list(range(V))
    random.shuffle(arr)
    return arr

def build_sequence_from_perm(perm: List[int], L: int, vocab_offset: int = 0, query_from_context: bool = True, with_replacement: bool = True, separate_key_value_vocab: bool = False, predict_all_values: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Input:  [x1, y(x1), x2, y(x2), ..., xL, y(xL), x_{L+1}]  (len 2L+1)
    
    Target modes:
    - predict_all_values=False (default): Only last position supervised
      Target: [-,   -,   -,   -,   ...,  -,    -,   y(x_{L+1})]
    - predict_all_values=True (Naman's approach): ALL value positions supervised
      Target: [-,y(x1),-,y(x2),...,-,y(xL),-,   y(x_{L+1})]
    
    If with_replacement=False, sample keys without replacement (each key appears at most once).
    If separate_key_value_vocab=True, keys use [0..V-1] and values use [V..2V-1] token IDs.
    """
    V = len(perm)
    if with_replacement:
        xs = [random.randrange(V) for _ in range(L)]
    else:
        # Sample without replacement: choose L distinct keys
        xs = random.sample(range(V), min(L, V))
    if query_from_context and len(xs) > 0:
        q = random.choice(xs)
    else:
        q = random.randrange(V)
    seq = []
    for i in range(L):
        key_token = xs[i] + vocab_offset
        if separate_key_value_vocab:
            value_token = perm[xs[i]] + vocab_offset + V  # Values in [V..2V-1]
        else:
            value_token = perm[xs[i]] + vocab_offset
        seq.append(key_token)
        seq.append(value_token)
    seq.append(q + vocab_offset)  # query key (from context if enabled)
    x = torch.tensor(seq, dtype=torch.long)
    y = torch.full((2*L+1,), IGNORE_INDEX, dtype=torch.long)
    
    # NEW: Optionally supervise ALL value positions
    if predict_all_values:
        for i in range(L):
            value_pos = 2*i + 1  # Position of value in sequence
            if separate_key_value_vocab:
                y[value_pos] = perm[xs[i]] + vocab_offset + V
            else:
                y[value_pos] = perm[xs[i]] + vocab_offset
    
    # Always supervise the query position
    if separate_key_value_vocab:
        y[-1] = perm[q] + vocab_offset + V  # Predict value token
    else:
        y[-1] = perm[q] + vocab_offset
    return x, y

class FixedDictDataset(Dataset):
    def __init__(self, V: int, L: int, n_samples: int, vocab_offset: int = 0, seed: int = 1337, query_from_context: bool = True, with_replacement: bool = True, separate_key_value_vocab: bool = False, predict_all_values: bool = False):
        self.V, self.L, self.n = V, L, n_samples
        self.vocab_offset = vocab_offset
        self.query_from_context = query_from_context
        self.with_replacement = with_replacement
        self.separate_key_value_vocab = separate_key_value_vocab
        self.predict_all_values = predict_all_values
        random.seed(seed)
        self.perm = sample_perm(V)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return build_sequence_from_perm(self.perm, self.L, self.vocab_offset, query_from_context=self.query_from_context, with_replacement=self.with_replacement, separate_key_value_vocab=self.separate_key_value_vocab, predict_all_values=self.predict_all_values)

class ChangingDictDataset(Dataset):
    def __init__(self, V: int, L: int, n_samples: int, vocab_offset: int = 0, query_from_context: bool = True, with_replacement: bool = True, separate_key_value_vocab: bool = False, predict_all_values: bool = False):
        self.V, self.L, self.n = V, L, n_samples
        self.vocab_offset = vocab_offset
        self.query_from_context = query_from_context
        self.with_replacement = with_replacement
        self.separate_key_value_vocab = separate_key_value_vocab
        self.predict_all_values = predict_all_values
    def __len__(self): return self.n
    def __getitem__(self, idx):
        return build_sequence_from_perm(sample_perm(self.V), self.L, self.vocab_offset, query_from_context=self.query_from_context, with_replacement=self.with_replacement, separate_key_value_vocab=self.separate_key_value_vocab, predict_all_values=self.predict_all_values)

class MixedDictDataset(Dataset):
    """
    Mixed over the SAME vocab [0..V-1].
    If p_is_changing=True: with prob p -> ChangingDict; else FixedDict.
    If p_is_changing=False: with prob p -> FixedDict; else ChangingDict. (legacy)
    """
    def __init__(self, V: int, L: int, n_samples: int, p: float, p_is_changing: bool = True, seed: int = 1337, query_from_context: bool = True, with_replacement: bool = True, separate_key_value_vocab: bool = False, predict_all_values: bool = False):
        self.V, self.L, self.n, self.p, self.p_is_changing = V, L, n_samples, p, p_is_changing
        self.query_from_context = query_from_context
        self.with_replacement = with_replacement
        self.separate_key_value_vocab = separate_key_value_vocab
        self.predict_all_values = predict_all_values
        rnd = random.Random(seed)
        self.fixed_perm = list(range(V)); rnd.shuffle(self.fixed_perm)
    def __len__(self): return self.n
    def __getitem__(self, idx):
        r = random.random()
        choose_changing = (r < self.p) if self.p_is_changing else (r >= self.p)
        if choose_changing:
            return build_sequence_from_perm(sample_perm(self.V), self.L, 0, query_from_context=self.query_from_context, with_replacement=self.with_replacement, separate_key_value_vocab=self.separate_key_value_vocab, predict_all_values=self.predict_all_values)
        else:
            return build_sequence_from_perm(self.fixed_perm, self.L, 0, query_from_context=self.query_from_context, with_replacement=self.with_replacement, separate_key_value_vocab=self.separate_key_value_vocab, predict_all_values=self.predict_all_values)

class MixedSeparateVocabsDataset(Dataset):
    """
    Two disjoint vocabs: [0..V-1] and [V..2V-1].
    Low-ID block: keys [0..V-1] map to values [0..V-1] (fixed via fixed_perm)
    High-ID block: keys [V..2V-1] map to values [V..2V-1] (changing, via random perms)
    
    If p_is_changing=True: with prob p -> sample from Changing (high block); else Fixed (low block).
    If p_is_changing=False: with prob p -> Fixed (low block); else Changing (high block).
    """
    def __init__(self, V: int, L: int, n_samples: int, p: float, p_is_changing: bool = True, seed: int = 1337, query_from_context: bool = True, with_replacement: bool = True):
        self.V, self.L, self.n, self.p, self.p_is_changing = V, L, n_samples, p, p_is_changing
        self.query_from_context = query_from_context
        self.with_replacement = with_replacement
        rnd = random.Random(seed)
        self.fixed_perm = list(range(V)); rnd.shuffle(self.fixed_perm)
    def __len__(self): return self.n
    def __getitem__(self, idx):
        r = random.random()
        choose_changing = (r < self.p) if self.p_is_changing else (r >= self.p)
        if choose_changing:
            # Changing on high IDs [V..2V-1]
            # Sample random perm of [0..V-1], then offset by V for both keys and values
            perm = sample_perm(self.V)
            return self._build_sequence_disjoint(perm, high_block=True)
        else:
            # Fixed on low IDs [0..V-1]
            return self._build_sequence_disjoint(self.fixed_perm, high_block=False)
    
    def _build_sequence_disjoint(self, perm: List[int], high_block: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build sequence with disjoint key-value blocks."""
        V = self.V
        L = self.L
        
        # Sample keys from appropriate block
        if self.with_replacement:
            keys_in_block = [random.randrange(V) for _ in range(L)]
        else:
            keys_in_block = random.sample(range(V), min(L, V))
        
        # Query key from context if enabled
        if self.query_from_context and len(keys_in_block) > 0:
            q_in_block = random.choice(keys_in_block)
        else:
            q_in_block = random.randrange(V)
        
        # Offset keys/values if high_block
        offset = V if high_block else 0
        
        # Build sequence: [k1, v1, k2, v2, ..., kL, vL, query_k]
        seq = []
        for i in range(L):
            key_token = keys_in_block[i] + offset
            value_token = perm[keys_in_block[i]] + offset  # perm maps [0..V-1] to [0..V-1], then offset
            seq.append(key_token)
            seq.append(value_token)
        seq.append(q_in_block + offset)  # query key
        
        x = torch.tensor(seq, dtype=torch.long)
        y = torch.full((2*L+1,), IGNORE_INDEX, dtype=torch.long)
        # Supervise only last position (query prediction)
        y[-1] = perm[q_in_block] + offset
        
        return x, y

def collate(batch):
    xs = torch.stack([b[0] for b in batch], dim=0)
    ys = torch.stack([b[1] for b in batch], dim=0)
    return xs, ys

# --------------------------
# Compact GPT
# --------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.scale

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, n_heads, dropout=0.0, non_causal: bool = False):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3*dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        self.non_causal = non_causal
        # Per-head multiplicative mask for ablations (1 = active, 0 = ablated)
        self.register_buffer("head_mask", torch.ones(self.n_heads))

    def forward(self, x, mask=None, return_attn=False):
        # x: (B, T, C)
        B, T, C = x.shape
        H, D = self.n_heads, self.head_dim

        # ---- robust reshape to (B, H, T, D) for q,k,v ----
        qkv = self.qkv(x)                                   # (B, T, 3*C)
        qkv = qkv.view(B, T, 3, H, D).permute(0, 3, 1, 2, 4)  # (B, H, T, 3, D)
        q, k, v = qkv[..., 0, :], qkv[..., 1, :], qkv[..., 2, :]  # each (B, H, T, D)

        # ---- scaled dot-product attention: att has shape (B, H, T, T) ----
        att = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)  # (B, H, T, T)

        # ---- attention mask ----
        # If non_causal: allow full bidirectional attention (no causal masking)
        # Else: use causal mask but unmask last row (query can see all)
        if not self.non_causal:
            causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
            if T > 0:
                causal[-1, :] = False
            att = att.masked_fill(causal, float("-inf"))

        if mask is not None:
            # if you ever pass an extra mask, it must be broadcastable to (B, H, T, T)
            att = att + mask

        att = F.softmax(att, dim=-1)
        att_do = self.attn_drop(att)

        y = torch.matmul(att_do, v)                          # (B, H, T, D)
        # Apply head mask before merging heads
        if self.head_mask is not None:
            y = y * self.head_mask.view(1, H, 1, 1)
        y = y.transpose(1, 2).contiguous().view(B, T, C)     # (B, T, C)
        y = self.proj_drop(self.proj(y))

        # Optional sanity check (turn on with: export DEBUG_ATTN=1)
        if os.environ.get("DEBUG_ATTN", "0") == "1":
            assert y.shape == (B, T, C)
            assert att.shape == (B, H, T, T), f"att shape {att.shape} != (B,H,T,T)"

        if return_attn:
            return y, att
        return y, None

class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.drop(self.fc2(x))
        return x

class Block(nn.Module):
    def __init__(self, dim, n_heads, dropout=0.0, non_causal: bool = False):
        super().__init__()
        self.ln1 = RMSNorm(dim)
        self.attn = MultiHeadAttention(dim, n_heads, dropout, non_causal=non_causal)
        self.ln2 = RMSNorm(dim)
        self.mlp = MLP(dim, 4.0, dropout)
        # If True, skip this layer (used for layer ablations)
        self.bypass = False
    def forward(self, x, return_attn=False):
        if self.bypass:
            return x, None
        a, att = self.attn(self.ln1(x), return_attn=return_attn)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x, att

class TinyGPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_layers=6, n_heads=6, dim=384, dropout=0.0, tie_weights=True,
                 non_causal: bool = False,
                 lookup_head: bool = False, lookup_weight: float = 1.0, lookup_mode: str = "add",
                 pointer_lookup: bool = False, pointer_mode: str = "replace"):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(block_size, dim)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(dim, n_heads, dropout, non_causal=non_causal) for _ in range(n_layers)])
        self.ln_f = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        if tie_weights:
            self.head.weight = self.tok_emb.weight
        self.lookup_head = lookup_head
        self.lookup_weight = lookup_weight
        self.lookup_mode = lookup_mode  # 'add' or 'replace'
        if self.lookup_head:
            self.lookup_proj = nn.Linear(dim, vocab_size, bias=False)
        self.pointer_lookup = pointer_lookup
        self.pointer_mode = pointer_mode  # 'add' or 'replace'

    def forward(self, idx, targets=None, return_hiddens=False, return_attn=False, disable_pointer: bool = False):
        B, T = idx.shape
        assert T <= self.block_size
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        hiddens = []
        attn_list = [] if return_attn else None
        for blk in self.blocks:
            x, att = blk(x, return_attn=return_attn)
            if return_hiddens:
                hiddens.append(x)
            if return_attn:
                attn_list.append(att)  # (B,H,T,T)
        x = self.ln_f(x)
        logits = self.head(x)          # (B, T, V)

        # Optional explicit lookup head at final query position (vector regression to values)
        if getattr(self, "lookup_head", False) and T >= 3:
            C = x.size(-1)
            keys = x[:, 0:T-1:2, :]    # (B, L, C)
            vals = x[:, 1:T-1:2, :]    # (B, L, C)
            q = x[:, -1, :]            # (B, C)
            Lctx = keys.size(1)
            if Lctx > 0:
                sim = torch.matmul(q.view(B, 1, C), keys.transpose(1, 2)) / math.sqrt(C)  # (B,1,L)
                w = F.softmax(sim, dim=-1)  # (B,1,L)
                agg = torch.matmul(w, vals).squeeze(1)  # (B, C)
                logits_lookup = self.lookup_proj(agg)   # (B, V)
                if self.lookup_mode == "replace":
                    logits[:, -1, :] = logits_lookup
                else:
                    logits[:, -1, :] = logits[:, -1, :] + self.lookup_weight * logits_lookup

        # Optional pointer-style lookup: attend keys, scatter weights onto their paired value token IDs
        if getattr(self, "pointer_lookup", False) and not disable_pointer and T >= 3:
            C = x.size(-1)
            keys_h = x[:, 0:T-1:2, :]          # (B, L, C)
            vals_ids = idx[:, 1:T-1:2]         # (B, L) token IDs for values
            q_h = x[:, -1, :]                  # (B, C)
            Lctx = keys_h.size(1)
            if Lctx > 0:
                sim = torch.matmul(q_h.view(B, 1, C), keys_h.transpose(1, 2)) / math.sqrt(C)  # (B,1,L)
                w = F.softmax(sim, dim=-1).squeeze(1)  # (B, L)
                eps = 1e-9
                pointer_logits = torch.full((B, self.vocab_size), -1e9, device=idx.device)
                pointer_logits.scatter_(1, vals_ids, torch.log(w + eps))
                if self.pointer_mode == "replace":
                    logits[:, -1, :] = pointer_logits
                else:
                    logits[:, -1, :] = logits[:, -1, :] + pointer_logits
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=IGNORE_INDEX
            )
        if return_hiddens or return_attn:
            return logits, loss, hiddens, attn_list
        return logits, loss

    # --------------------------
    # Ablation helpers
    # --------------------------
    def clear_head_ablation(self):
        """Set all head masks to 1 (no head ablated)."""
        for blk in self.blocks:
            blk.attn.head_mask.fill_(1.0)

    def set_head_ablation(self, heads_by_layer: Dict[int, List[int]]):
        """Zero out specified heads per layer.
        heads_by_layer: mapping layer_idx -> list of head indices to ablate
        """
        self.clear_head_ablation()
        for layer_idx, heads in heads_by_layer.items():
            if 0 <= layer_idx < len(self.blocks):
                mask = self.blocks[layer_idx].attn.head_mask
                for h in heads:
                    if 0 <= h < mask.numel():
                        mask[h] = 0.0

    def clear_layer_bypass(self):
        """Disable bypass on all layers."""
        for blk in self.blocks:
            blk.bypass = False

    def set_layer_bypass(self, layers: List[int]):
        """Enable bypass on specified layers (skip attention + MLP)."""
        self.clear_layer_bypass()
        for li in layers:
            if 0 <= li < len(self.blocks):
                self.blocks[li].bypass = True

# --------------------------
# DDP helpers
# --------------------------

def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"]); world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(rank % torch.cuda.device_count())
        return rank, world_size, True
    else:
        return 0, 1, False

def cleanup_ddp():
    if dist is not None and dist.is_initialized():
        dist.destroy_process_group()

# --------------------------
# Eval helpers
# --------------------------

@torch.no_grad()
def evaluate_model(model, loader, device, debug=False):
    model.eval()
    total, correct = 0, 0
    for batch_idx, (xb, yb) in enumerate(loader):
        xb = xb.to(device); yb = yb.to(device)
        logits, _ = model(xb, targets=None)
        preds = logits[:, -1, :].argmax(dim=-1)
        gold = yb[:, -1]
        mask = gold != IGNORE_INDEX
        
        # DEBUG: Print first 5 examples from first batch
        if debug and batch_idx == 0:
            print("\n=== DEBUG: Predictions vs Targets (first 5 examples) ===")
            for i in range(min(5, xb.size(0))):
                if mask[i]:
                    seq = xb[i].tolist()
                    pred_id = preds[i].item()
                    gold_id = gold[i].item()
                    # Get top-5 predictions
                    top5_probs, top5_ids = logits[i, -1, :].softmax(dim=-1).topk(5)
                    print(f"Example {i}:")
                    print(f"  Sequence: {seq}")
                    print(f"  Predicted: {pred_id}, Target: {gold_id}, Match: {pred_id == gold_id}")
                    print(f"  Top-5 predictions: {[(id.item(), f'{prob.item():.4f}') for id, prob in zip(top5_ids, top5_probs)]}")
        
        total += mask.sum().item()
        correct += (preds[mask] == gold[mask]).sum().item()
    return 100.0 * correct / max(1, total)

def entropy_bits_from_logits(logits: torch.Tensor) -> torch.Tensor:
    log_probs = logits.float().log_softmax(dim=-1)
    probs = log_probs.exp()
    ent_nats = -(probs * log_probs).sum(dim=-1)
    return ent_nats * LOG2E

def bayes_entropy_with_replacement(keys: torch.Tensor, K: int) -> torch.Tensor:
    L = keys.numel()
    H = []
    seen = set()
    for i in range(L):
        x_i = int(keys[i].item())
        if x_i in seen:
            H.append(0.0)
        else:
            H.append(math.log(K - len(seen), 2))
            seen.add(x_i)
    return torch.tensor(H, dtype=torch.float32)

def bayes_entropy_without_replacement(K: int, L: int) -> torch.Tensor:
    return torch.tensor([math.log(K - t, 2) for t in range(0, L)], dtype=torch.float32)

def build_bayes_posterior(V: int, obs_pairs: List[Tuple[int,int]], key: int) -> torch.Tensor:
    """
    Posterior over values for this key given observed pairs so far.
    If key seen -> delta; else uniform over remaining values.
    """
    mapping: Dict[int,int] = {}
    used_vals = set()
    for x,y in obs_pairs:
        if x not in mapping:
            mapping[x] = y
            used_vals.add(y)
    post = torch.zeros(V, dtype=torch.float32)
    if key in mapping:
        post[mapping[key]] = 1.0
        return post
    rem = [v for v in range(V) if v not in used_vals]
    p = 1.0 / len(rem)
    for v in rem:
        post[v] = p
    return post

@torch.no_grad()
def evaluate_entropy_experiment(model, V: int, L: int, device, mode: str = "with_replacement",
                                n_samples: int = 10000, vocab_offset: int = 0, csv_path: Optional[str] = None,
                                target: str = "query", query_source: str = "context", disable_pointer_for_eval: bool = True):
    """
    Measure entropy at VALUE prediction positions along the context (t = 1..L):
      - At step t, we present the prefix ending with key x_t (no y(x_t) yet),
        read logits at the last position (which predicts y(x_t)), and compute
        the entropy over the value vocabulary.
      - This matches §3 trajectories: with_replacement → sawtooth; without → smooth decay.
    """
    model.eval()
    K = V
    ent_model_sum = torch.zeros(L, dtype=torch.float64, device=device)
    ent_bayes_sum = torch.zeros(L, dtype=torch.float64, device=device)

    # Detect if model uses separate key/value vocabularies (values in [V..2V-1])
    uses_separate_vocabs = False
    inner = getattr(model, "module", model)
    if getattr(inner, "vocab_size", V) == 2 * V:
        uses_separate_vocabs = True

    for _ in range(n_samples):
        perm = sample_perm(V)
        if mode == "with_replacement":
            xs = [random.randrange(V) for _ in range(L)]
        elif mode == "without_replacement":
            xs = list(range(V)); random.shuffle(xs); xs = xs[:L]
        else:
            raise ValueError("mode must be with_replacement or without_replacement")

        # Running observations for Bayes posterior
        obs_pairs: List[Tuple[int,int]] = []

        for t in range(1, L+1):
            if target == "value":
                # Build prefix with first t-1 full pairs, then key x_t (measure entropy over y(x_t))
                seq_t = []
                for i in range(t-1):
                    key_tok = xs[i]
                    val_tok = perm[xs[i]] + (V if uses_separate_vocabs else 0)
                    seq_t.append(key_tok)
                    seq_t.append(val_tok)
                seq_t.append(xs[t-1])
                x = torch.tensor([tok + vocab_offset for tok in seq_t], dtype=torch.long, device=device)[None, :]
                logits, _ = model(x, targets=None, disable_pointer=disable_pointer_for_eval)
                logits_last = logits[0, -1, :]
                if logits_last.shape[-1] == 2 * V:
                    logits_last = logits_last[V:2*V]
                ent = entropy_bits_from_logits(logits_last[None, :])[0]
                ent_model_sum[t-1] += ent.double()
                bayes_post = build_bayes_posterior(V, obs_pairs, xs[t-1])
                hb = -(bayes_post[bayes_post > 0] * bayes_post[bayes_post > 0].log2()).sum().item() if bayes_post.sum() > 0 else 0.0
                ent_bayes_sum[t-1] += hb
                # update obs with (x_t,y_t)
                obs_pairs.append((xs[t-1], perm[xs[t-1]]))
            else:
                # target == "query": present first t pairs fully, then independent query q
                seq_t = []
                for i in range(t):
                    key_tok = xs[i]
                    val_tok = perm[xs[i]] + (V if uses_separate_vocabs else 0)
                    seq_t.append(key_tok)
                    seq_t.append(val_tok)
                # choose query key q
                if query_source == "context":
                    q = random.choice(xs[:t])
                else:
                    q = random.randrange(V)
                seq_t.append(q)
                x = torch.tensor([tok + vocab_offset for tok in seq_t], dtype=torch.long, device=device)[None, :]
                logits, _ = model(x, targets=None, disable_pointer=disable_pointer_for_eval)
                logits_q = logits[0, -1, :]
                if logits_q.shape[-1] == 2 * V:
                    logits_q = logits_q[V:2*V]
                ent = entropy_bits_from_logits(logits_q[None, :])[0]
                ent_model_sum[t-1] += ent.double()
                # Bayes entropy for y(q) given obs_pairs up to t
                # Build obs up to t
                obs_now = [(xs[i], perm[xs[i]]) for i in range(t)]
                bayes_post = build_bayes_posterior(V, obs_now, q)
                hb = -(bayes_post[bayes_post > 0] * bayes_post[bayes_post > 0].log2()).sum().item() if bayes_post.sum() > 0 else 0.0
                ent_bayes_sum[t-1] += hb

    ent_model_mean = (ent_model_sum / n_samples).float().cpu()
    ent_bayes_mean = (ent_bayes_sum / n_samples).float().cpu()
    mae_bits = torch.mean(torch.abs(ent_model_mean - ent_bayes_mean)).item()

    if csv_path is not None:
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "H_model_bits", "H_bayes_bits"])
            for t in range(1, L+1):
                w.writerow([t, float(ent_model_mean[t-1]), float(ent_bayes_mean[t-1])])

    return {"MAE_bits": mae_bits, "H_model_bits": ent_model_mean.tolist(), "H_bayes_bits": ent_bayes_mean.tolist()}

@torch.no_grad()
def evaluate_kl_layers(model: TinyGPT, V: int, L: int, device,
                       mode: str = "with_replacement",
                       n_samples: int = 2000,
                       vocab_offset: int = 0,
                       csv_path: Optional[str] = None) -> Dict:
    """
    Compute KL(model_layer || Bayes) in bits at each layer and value-position.
    We produce per-layer logits by (ln_f -> head) applied to layer hiddens.
    """
    model.eval()
    n_layers = len(model.blocks)
    kl_sum = torch.zeros(n_layers+1, L, dtype=torch.float64, device=device)
    cnt = 0

    for _ in range(n_samples):
        if mode == "with_replacement":
            perm = sample_perm(V)
            xs = [random.randrange(V) for _ in range(L+1)]
            seq = []
            obs_pairs = []
            for i in range(L):
                seq.append(xs[i]); seq.append(perm[xs[i]])
                obs_pairs.append((xs[i], perm[xs[i]]))
            seq.append(xs[L])
        elif mode == "without_replacement":
            perm = sample_perm(V)
            keys = list(range(V)); random.shuffle(keys)
            seq = []
            obs_pairs = []
            for i in range(L):
                seq.append(keys[i]); seq.append(perm[keys[i]])
                obs_pairs.append((keys[i], perm[keys[i]]))
            seq.append(keys[L % V])
        else:
            raise ValueError("mode must be with_replacement or without_replacement")

        x = torch.tensor([t + vocab_offset for t in seq], dtype=torch.long, device=device)[None, :]
        logits, _, hiddens, _ = model(x, targets=None, return_hiddens=True, return_attn=False)

        # Build Bayes posteriors for each value-position
        bayes_vecs = []
        for j in range(L):
            key_j = seq[2*j]
            post = build_bayes_posterior(V, obs_pairs[:j], key_j)
            bayes_vecs.append(post)
        bayes = torch.stack(bayes_vecs, dim=0).to(device)  # (L, V)

        def layer_kl_from_hidden(h: torch.Tensor) -> torch.Tensor:
            h_norm = model.ln_f(h)
            layer_logits = model.head(h_norm)                 # (1, T, V)
            lp = layer_logits[0, 1:2*L:2, :].log_softmax(-1)  # (L, V)
            p = lp.exp()
            q = (bayes + 1e-12)
            kl = (p * (lp - q.log())).sum(dim=-1) * LOG2E
            return kl  # (L,)

        for li, h in enumerate(hiddens):
            kl_sum[li, :] += layer_kl_from_hidden(h)

        final_lp = logits[0, 1:2*L:2, :].log_softmax(dim=-1)
        p_final = final_lp.exp()
        q = (bayes + 1e-12)
        kl_final = (p_final * (final_lp - q.log())).sum(dim=-1) * LOG2E
        kl_sum[n_layers, :] += kl_final

        cnt += 1

    kl_mean = (kl_sum / max(cnt,1)).float().cpu()  # (layers+1, L)
    if csv_path is not None:
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            header = ["t"] + [f"layer_{i}_KL_bits" for i in range(n_layers)] + ["final_KL_bits"]
            w.writerow(header)
            for t in range(1, L+1):
                row = [t] + [float(kl_mean[i, t-1]) for i in range(n_layers)] + [float(kl_mean[n_layers, t-1])]
                w.writerow(row)

    return {
        "layers": n_layers,
        "L": L,
        "KL_bits_per_layer_per_t": kl_mean.tolist(),
        "KL_bits_layer_means": kl_mean.mean(dim=1).tolist(),
    }

@torch.no_grad()
def dump_attention_sample(model: TinyGPT, V: int, L: int, device,
                          n_batch: int = 32,
                          mode: str = "with_replacement",
                          vocab_offset: int = 0,
                          out_npz: str = "attn_dump.npz"):
    import numpy as np
    model.eval()
    xs = []
    seqs = []
    for _ in range(n_batch):
        if mode == "with_replacement":
            perm = sample_perm(V)
            xs_r = [random.randrange(V) for _ in range(L+1)]
            seq = []
            for i in range(L):
                seq.append(xs_r[i]); seq.append(perm[xs_r[i]])
            seq.append(xs_r[L])
        else:
            perm = sample_perm(V)
            keys = list(range(V)); random.shuffle(keys)
            seq = []
            for i in range(L):
                seq.append(keys[i]); seq.append(perm[keys[i]])
            seq.append(keys[L % V])
        seqs.append(seq)
        xs.append(torch.tensor([t + vocab_offset for t in seq], dtype=torch.long))
    x = torch.stack(xs, dim=0).to(device)
    logits, _, _, attn_list = model(x, targets=None, return_hiddens=False, return_attn=True)
    to_save = {f"attn_layer_{i}": att.cpu().numpy() for i, att in enumerate(attn_list)}
    to_save["seq"] = np.array(seqs, dtype=int)
    np.savez(out_npz, **to_save)
    return {"saved": out_npz, "layers": len(attn_list), "B": n_batch}

# --------------------------
# Main
# --------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--const_lr", action="store_true", help="keep learning rate fixed at --lr (no scheduler)")
    parser.add_argument("--no_scheduler", action="store_true", help="disable warmup+cosine scheduling; keep LR constant")
    parser.add_argument("--train_all_values", action="store_true", help="add auxiliary loss on value tokens 1,3,5,...")
    parser.add_argument("--aux_value_weight", type=float, default=1.0, help="weight for auxiliary value-token loss")
    parser.add_argument("--log_grad_norm", action="store_true", help="log grad_norm in eval prints")
    parser.add_argument("--ckpt_every", type=int, default=0, help="save checkpoint every N steps (0 disables)")
    parser.add_argument("--overfit_one_batch", action="store_true", help="reuse a single training batch repeatedly")
    parser.add_argument("--exp", type=str, default="changing",
                        choices=["fixed", "changing", "mixed", "mixed_sep", "mixed_separate"],
                        help="training dataset choice")
    parser.add_argument("--p", type=float, default=0.1, help="mixing prob")
    parser.add_argument("--p_is_changing", type=lambda s: s.lower() in {"1","true","t","yes","y"}, default=True,
                        help="If True, p == prob of Changing; else p == prob of Fixed.")
    parser.add_argument("--V", type=int, default=10, help="vocab size per block (mixed_sep uses 2V total)")
    parser.add_argument("--L", type=int, default=10, help="dictionary length L (sequence len is 2L+1)")
    parser.add_argument("--n_train", type=int, default=200000)
    parser.add_argument("--n_val", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_steps", type=int, default=30000)
    parser.add_argument("--eval_every", type=int, default=250)
    parser.add_argument("--ckpt", type=str, default="ckpt.pt")
    parser.add_argument("--init_from", type=str, default="", help="If provided, initialize model weights from this checkpoint (weights only)")
    parser.add_argument("--resume", action="store_true", help="resume training from --ckpt if it exists (loads model/opt/scheduler/step)")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--eval_only", action="store_true")

    # Entropy eval
    parser.add_argument("--entropy_eval", type=str, default="",
                        choices=["", "with_replacement", "without_replacement"],
                        help="Run §3 entropy experiment (no training): compute MAE vs Bayes and dump CSV.")
    parser.add_argument("--entropy_csv", type=str, default="entropy_curve.csv")
    parser.add_argument("--entropy_samples", type=int, default=10000)
    parser.add_argument("--entropy_target", type=str, default="query",
                        choices=["query", "value"],
                        help="Measure entropy at query (y(q) at last position) or at value positions (y(x_t)) for debugging")
    parser.add_argument("--entropy_query_source", type=str, default="context",
                        choices=["context", "uniform"],
                        help="When target=query: choose q from observed context keys (context) or uniform over all keys (uniform)")
    parser.add_argument("--entropy_disable_pointer", type=lambda s: s.lower() in {"1","true","t","yes","y"}, default=True,
                        help="If True, disable pointer head during entropy eval to read base LM head logits")

    # Per-layer posterior KL
    parser.add_argument("--kl_eval", type=str, default="",
                        choices=["", "with_replacement", "without_replacement"],
                        help="Compute per-layer KL(model||Bayes) at value positions; dump CSV.")
    parser.add_argument("--kl_csv", type=str, default="kl_layers.csv")
    parser.add_argument("--kl_samples", type=int, default=2000)

    # Optional attention dump
    parser.add_argument("--dump_attn", type=int, default=0, help="If >0, dump a batch of raw attention maps.")
    parser.add_argument("--attn_npz", type=str, default="attn_dump.npz")
    parser.add_argument("--attn_mode", type=str, default="with_replacement",
                        choices=["with_replacement", "without_replacement"])

    # in argparse:
    parser.add_argument("--self_test", type=int, default=0)
    parser.add_argument("--query_from_context", type=lambda s: s.lower() in {"1","true","t","yes","y"}, default=True,
                        help="If True, sample query key from observed context keys")
    parser.add_argument("--with_replacement", type=lambda s: s.lower() in {"1","true","t","yes","y"}, default=True,
                        help="If True, sample keys with replacement (default); False = without replacement")
    parser.add_argument("--separate_key_value_vocab", type=lambda s: s.lower() in {"1","true","t","yes","y"}, default=True,
                        help="If True, use separate vocabularies for keys [0..V-1] and values [V..2V-1] (paper setup)")
    parser.add_argument("--non_causal", type=lambda s: s.lower() in {"1","true","t","yes","y"}, default=False,
                        help="If True, use non-causal bidirectional attention everywhere (no causal mask)")


    args = parser.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)
    rank, world_size, use_ddp = setup_ddp()
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}" if torch.cuda.is_available() else "cpu")
    is_master = (rank == 0)

    # early in main() after building device:
    if args.self_test:
        _self_test_changingdict(V=args.V, L=args.L, device=device if torch.cuda.is_available() else "cpu")
        return

    # Build training dataset
    if args.exp == "fixed":
        train_ds = FixedDictDataset(args.V, args.L, args.n_train, vocab_offset=0, seed=args.seed, query_from_context=args.query_from_context, with_replacement=args.with_replacement, separate_key_value_vocab=args.separate_key_value_vocab, predict_all_values=args.train_all_values)
    elif args.exp == "changing":
        train_ds = ChangingDictDataset(args.V, args.L, args.n_train, vocab_offset=0, query_from_context=args.query_from_context, with_replacement=args.with_replacement, separate_key_value_vocab=args.separate_key_value_vocab, predict_all_values=args.train_all_values)
    elif args.exp == "mixed":
        train_ds = MixedDictDataset(args.V, args.L, args.n_train, p=args.p, p_is_changing=args.p_is_changing, seed=args.seed, query_from_context=args.query_from_context, with_replacement=args.with_replacement, separate_key_value_vocab=args.separate_key_value_vocab, predict_all_values=args.train_all_values)
    elif args.exp == "mixed_sep" or args.exp == "mixed_separate":
        train_ds = MixedSeparateVocabsDataset(args.V, args.L, args.n_train, p=args.p, p_is_changing=args.p_is_changing, seed=args.seed, query_from_context=args.query_from_context, with_replacement=args.with_replacement)
    else:
        raise ValueError("unknown --exp")

    # Validation loaders (same-vocab)
    # Use same seed as training so FixedDict uses the same fixed permutation in train/val
    val_fixed = DataLoader(FixedDictDataset(args.V, args.L, args.n_val, vocab_offset=0, seed=args.seed, query_from_context=args.query_from_context, with_replacement=args.with_replacement, separate_key_value_vocab=args.separate_key_value_vocab, predict_all_values=args.train_all_values),
                           batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate)
    val_changing = DataLoader(ChangingDictDataset(args.V, args.L, args.n_val, vocab_offset=0, query_from_context=args.query_from_context, with_replacement=args.with_replacement, separate_key_value_vocab=args.separate_key_value_vocab, predict_all_values=args.train_all_values),
                              batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate)
    # Ensure fixed component in MixedDict shares the same fixed permutation via the same seed
    val_mixed = DataLoader(MixedDictDataset(args.V, args.L, args.n_val, p=0.1, p_is_changing=True, seed=args.seed, query_from_context=args.query_from_context, with_replacement=args.with_replacement, separate_key_value_vocab=args.separate_key_value_vocab, predict_all_values=args.train_all_values),
                           batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate)

    # Extra eval for MixedSeparateVocabs transfer (low IDs vs high IDs)
    val_changing_low = DataLoader(ChangingDictDataset(args.V, args.L, args.n_val, vocab_offset=0, query_from_context=args.query_from_context, with_replacement=args.with_replacement, predict_all_values=args.train_all_values),
                                  batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate)
    val_changing_high = DataLoader(ChangingDictDataset(args.V, args.L, args.n_val, vocab_offset=args.V, query_from_context=args.query_from_context, with_replacement=args.with_replacement, predict_all_values=args.train_all_values),
                                   batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate)

    if use_ddp:
        sampler = torch.utils.data.distributed.DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=2, pin_memory=True, collate_fn=collate)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True, collate_fn=collate)

    # Vocab size: if separate_key_value_vocab is True, we need 2V tokens (keys + values)
    if args.separate_key_value_vocab or args.exp == "mixed_sep":
        vocab_size = 2 * args.V
    else:
        vocab_size = args.V
    block_size = 2*args.L + 1

    model = TinyGPT(vocab_size=vocab_size, block_size=block_size, n_layers=args.layers,
                    n_heads=args.heads, dim=args.dim, dropout=args.dropout, non_causal=args.non_causal,
                    lookup_head=False, lookup_weight=1.0, lookup_mode="add",
                    pointer_lookup=True, pointer_mode="replace").to(device)
    if use_ddp:
        model = DDP(model, device_ids=[device.index])

    # Optional: initialize from a prior checkpoint (weights-only)
    if args.init_from:
        if os.path.exists(args.init_from):
            state = torch.load(args.init_from, map_location=device)
            src_sd = state.get("model", state)
            tgt = (model.module if isinstance(model, DDP) else model)
            tgt_sd = tgt.state_dict()
            # Filter only shape-compatible tensors to avoid size mismatch (e.g., vocab size changes 20->40)
            filtered = {}
            skipped = []
            for k, v in src_sd.items():
                if k in tgt_sd and tgt_sd[k].shape == v.shape:
                    filtered[k] = v
                else:
                    skipped.append(k)
            missing_before = len([k for k in tgt_sd.keys() if k not in filtered])
            tgt.load_state_dict(filtered, strict=False)
            if is_master:
                print(f"[init_from] Loaded {len(filtered)} tensors; skipped {len(skipped)} due to shape mismatch; missing_on_target={missing_before}")
        else:
            if is_master:
                print(f"[init_from] WARNING: file not found: {args.init_from}")

    # --- Analysis-only modes (no training) ---
    if args.entropy_eval:
        if os.path.exists(args.ckpt):
            state = torch.load(args.ckpt, map_location=device)
            (model.module if isinstance(model, DDP) else model).load_state_dict(state["model"])
        res = evaluate_entropy_experiment(
            model, V=args.V, L=args.L, device=device,
            mode=args.entropy_eval, n_samples=args.entropy_samples,
            vocab_offset=0, csv_path=args.entropy_csv,
            target=args.entropy_target, query_source=args.entropy_query_source,
            disable_pointer_for_eval=args.entropy_disable_pointer
        )
        if is_master:
            print(json.dumps({"entropy_eval": args.entropy_eval, **res}, indent=2))
            print(f"Wrote CSV to {args.entropy_csv}")
        cleanup_ddp()
        return

    if args.kl_eval:
        if os.path.exists(args.ckpt):
            state = torch.load(args.ckpt, map_location=device)
            (model.module if isinstance(model, DDP) else model).load_state_dict(state["model"])
        res = evaluate_kl_layers(
            model, V=args.V, L=args.L, device=device,
            mode=args.kl_eval, n_samples=args.kl_samples,
            vocab_offset=0, csv_path=args.kl_csv
        )
        if is_master:
            print(json.dumps({"kl_eval": args.kl_eval, **res}, indent=2))
            print(f"Wrote CSV to {args.kl_csv}")
        cleanup_ddp()
        return

    if args.dump_attn > 0:
        if os.path.exists(args.ckpt):
            state = torch.load(args.ckpt, map_location=device)
            (model.module if isinstance(model, DDP) else model).load_state_dict(state["model"])
        res = dump_attention_sample(model, V=args.V, L=args.L, device=device,
                                    n_batch=args.dump_attn, mode=args.attn_mode,
                                    vocab_offset=0, out_npz=args.attn_npz)
        if is_master:
            print(json.dumps({"dump_attn": res}, indent=2))
        cleanup_ddp()
        return

    # --------------------------
    # Training
    # --------------------------
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())  # disabled for FP32
    scaler = None  # AMP disabled: full FP32 training
    scheduler = None
    if not args.const_lr and not args.no_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.max_steps)
    warmup_steps = max(100, args.max_steps // 50)
    max_grad_norm = 1.0

    def save_ckpt():
        if is_master:
            # Ensure parent directory exists for checkpoint path
            try:
                os.makedirs(os.path.dirname(args.ckpt) or '.', exist_ok=True)
            except Exception:
                pass
            to_save = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
            payload = {
                "model": to_save,
                "args": vars(args),
                "step": step,
            }
            try:
                payload["opt"] = opt.state_dict()
            except Exception:
                pass
            try:
                if scheduler is not None:
                    payload["scheduler"] = scheduler.state_dict()
            except Exception:
                pass
            # Ensure parent directory for checkpoint exists
            ckpt_dir = os.path.dirname(args.ckpt)
            if ckpt_dir:
                os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(payload, args.ckpt)

    # eval-only
    if args.eval_only:
        assert os.path.exists(args.ckpt), f"Checkpoint not found: {args.ckpt}"
        state = torch.load(args.ckpt, map_location=device)
        (model.module if isinstance(model, DDP) else model).load_state_dict(state["model"])
        results = {
            "FixedDict": evaluate_model(model, val_fixed, device),
            "ChangingDict": evaluate_model(model, val_changing, device),
            "MixedDict_p0.1": evaluate_model(model, val_mixed, device),
        }
        if is_master:
            print(json.dumps(results, indent=2))
        cleanup_ddp()
        return

    model.train()
    step = 0

    # resume training if requested and checkpoint exists
    if args.resume and os.path.exists(args.ckpt):
        state = torch.load(args.ckpt, map_location=device)
        (model.module if isinstance(model, DDP) else model).load_state_dict(state.get("model", {}))
        if "opt" in state:
            try:
                opt.load_state_dict(state["opt"])
            except Exception:
                pass
        if scheduler is not None and "scheduler" in state:
            try:
                scheduler.load_state_dict(state["scheduler"])
            except Exception:
                pass
        step = int(state.get("step", step))
    # prepare one batch for overfit mode if requested
    overfit_batch = None
    if args.overfit_one_batch:
        xb1, yb1 = next(iter(train_loader))
        overfit_batch = (xb1.to(device), yb1.to(device))

    # DEBUG: Print first 3 training examples to verify data generation
    if is_master and step == 0:
        print("\n=== DEBUG: First 3 training examples ===")
        debug_loader = iter(train_loader)
        xb_debug, yb_debug = next(debug_loader)
        for i in range(min(3, xb_debug.size(0))):
            x_seq = xb_debug[i].tolist()
            y_seq = yb_debug[i].tolist()
            print(f"\nExample {i+1}:")
            print(f"  Input sequence (len={len(x_seq)}): {x_seq}")
            print(f"  Target sequence: {y_seq}")
            # Parse as key-value pairs
            pairs = []
            for j in range(0, len(x_seq) - 1, 2):
                if j + 1 < len(x_seq) - 1:
                    pairs.append(f"{x_seq[j]}->{x_seq[j+1]}")
            query_key = x_seq[-1]
            expected_value = y_seq[-1]
            print(f"  Pairs: {', '.join(pairs)}")
            print(f"  Query key: {query_key}, Expected value: {expected_value}")
        print("=" * 50 + "\n")

    while step < args.max_steps:
        for xb, yb in train_loader:
            if args.overfit_one_batch and overfit_batch is not None:
                xb, yb = overfit_batch
            else:
                xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            # keep LR constant if requested
            if args.const_lr or args.no_scheduler:
                for g in opt.param_groups:
                    g["lr"] = args.lr

            from contextlib import nullcontext
            autocast_ctx = nullcontext()
            with autocast_ctx:
                logits, _ = model(xb, targets=None)
                # main loss (original targets: only last position supervised)
                Vv = logits.size(-1)
                main_loss = F.cross_entropy(
                    logits.view(-1, Vv),
                    yb.view(-1),
                    ignore_index=IGNORE_INDEX
                )
                
                # DEBUG: Compute query-only loss separately for logging
                B, Tt, _ = logits.shape
                query_logits = logits[:, -1, :]  # last position (query)
                query_labels = yb[:, -1]  # last target
                query_mask = query_labels != IGNORE_INDEX
                if query_mask.any():
                    query_loss = F.cross_entropy(query_logits[query_mask], query_labels[query_mask])
                else:
                    query_loss = torch.tensor(0.0, device=logits.device)
                
                # auxiliary supervision on value tokens (predict value after key positions)
                aux_loss = torch.tensor(0.0, device=logits.device)
                if args.train_all_values:
                    if Tt > 1:
                        logits_t = logits[:, :-1, :]  # predict next token
                        labels_next = xb[:, 1:]
                        idx = torch.arange(Tt - 1, device=xb.device)
                        even_pos = (idx % 2 == 0)  # positions 0,2,4,... (keys)
                        aux_mask = even_pos.unsqueeze(0).expand(B, -1)
                        flat_mask = aux_mask.reshape(-1)
                        aux_logits = logits_t.reshape(-1, Vv)[flat_mask]
                        aux_labels = labels_next.reshape(-1)[flat_mask]
                        if aux_logits.numel() > 0:
                            aux_loss = F.cross_entropy(aux_logits, aux_labels)
                            loss = main_loss + args.aux_value_weight * aux_loss
                        else:
                            loss = main_loss
                    else:
                        loss = main_loss
                else:
                    loss = main_loss
            loss.backward()

            # grad clip
            if scaler is not None:
                scaler.unscale_(opt)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            opt.step()

            # LR warmup then cosine (if scheduler enabled)
            if scheduler is not None:
                if step < warmup_steps:
                    scale = float(step + 1) / warmup_steps
                    for g in opt.param_groups:
                        g["lr"] = args.lr * scale
                else:
                    scheduler.step()

            step += 1

            # periodic checkpoint
            if args.ckpt_every and (step % args.ckpt_every == 0):
                save_ckpt()

            if is_master and (step % args.eval_every == 0 or step == 1):
                model.eval()
                with torch.no_grad():
                    r_fixed = evaluate_model(model, val_fixed, device)
                    # Enable debug for ChangingDict to see predictions
                    debug_mode = (step <= 500) or (step % 5000 == 0)  # Debug at start and periodically
                    r_changing = evaluate_model(model, val_changing, device, debug=debug_mode)
                    r_mixed = evaluate_model(model, val_mixed, device)
                    if args.exp == "mixed_sep":
                        r_change_low = evaluate_model(model, val_changing_low, device)
                        r_change_high = evaluate_model(model, val_changing_high, device)
                    else:
                        r_change_low = None; r_change_high = None
                msg = {
                    "step": step,
                    "loss": float(loss.item()),
                    "query_loss": float(query_loss.item()),
                    "aux_loss": float(aux_loss.item()),
                    "FixedDict": r_fixed,
                    "ChangingDict": r_changing,
                    "MixedDict_p0.1": r_mixed,
                    "lr": opt.param_groups[0]["lr"] if len(opt.param_groups) else None,
                }
                if args.log_grad_norm:
                    try:
                        msg["grad_norm"] = float(grad_norm.item() if hasattr(grad_norm, "item") else grad_norm)
                    except Exception:
                        pass
                if r_change_low is not None:
                    msg["Changing_lowIDs"] = r_change_low
                    msg["Changing_highIDs"] = r_change_high
                print(json.dumps(msg))
                model.train()

            if step >= args.max_steps:
                break

    save_ckpt()

    if is_master:
        model.eval()
        results = {
            "FixedDict": evaluate_model(model, val_fixed, device),
            "ChangingDict": evaluate_model(model, val_changing, device),
            "MixedDict_p0.1": evaluate_model(model, val_mixed, device),
        }
        if args.exp == "mixed_sep":
            results["Changing_lowIDs"] = evaluate_model(model, val_changing_low, device)
            results["Changing_highIDs"] = evaluate_model(model, val_changing_high, device)
        with open("results.json", "w") as f:
            json.dump(results, f, indent=2)
        print("Final results:", json.dumps(results, indent=2))

    cleanup_ddp()
# ---- add near the top-level helpers ----
def _self_test_changingdict(V=10, L=10, batch=32, iters=400, device="cuda"):
    ds = ChangingDictDataset(V, L, n_samples=batch)
    xb, yb = collate([ds[i] for i in range(batch)])
    # verify label correctness
    for i in range(batch):
        seq = xb[i].tolist()
        gold = yb[i, -1].item()
        # rebuild perm from the first 2L tokens of seq
        obs = seq[:2*L]
        mapping = {}
        for j in range(0, 2*L, 2):
            xj, yj = obs[j], obs[j+1]
            mapping[xj] = yj
        query = seq[-1]
        assert gold == mapping.get(query, gold), f"Label mismatch: got {gold}, expected {mapping.get(query)}"
    # overfit a single batch
    model = TinyGPT(vocab_size=V, block_size=2*L+1, n_layers=4, n_heads=4, dim=192).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=0.0)
    xb, yb = xb.to(device), yb.to(device)
    for t in range(iters):
        opt.zero_grad(set_to_none=True)
        _, loss = model(xb, targets=yb)
        loss.backward()
        opt.step()
        if (t+1) % 50 == 0:
            print(f"[self_test] iter {t+1} loss {loss.item():.4f}")
    # final acc
    with torch.no_grad():
        logits, _ = model(xb, targets=None)
        preds = logits[:, -1, :].argmax(-1)
        gold = yb[:, -1]
        acc = (preds == gold).float().mean().item() * 100
        print(f"[self_test] final batch acc: {acc:.1f}%")

if __name__ == "__main__":
    main()


# Compatibility aliases for dependent scripts
make_batch = build_sequence_from_perm
evaluate_entropy_mae = evaluate_entropy_experiment
evaluate_kl_divergence = evaluate_kl_layers
