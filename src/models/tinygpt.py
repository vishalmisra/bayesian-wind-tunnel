"""
TinyGPT: Minimal transformer for Bayesian wind tunnel experiments.

This is a simplified GPT architecture used for bijection learning and 
HMM state tracking experiments, as described in:
  "The Bayesian Geometry of Transformer Attention" (Paper I)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.scale


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with optional causal masking."""
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0, non_causal: bool = False):
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.non_causal = non_causal
        
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # Head mask for ablation experiments
        self.register_buffer("head_mask", torch.ones(n_heads))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)  # Each: (B, T, n_heads, head_dim)
        
        q = q.transpose(1, 2)  # (B, n_heads, T, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        # Causal mask
        if not self.non_causal:
            mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
            attn = attn.masked_fill(mask, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply head mask for ablation
        attn = attn * self.head_mask.view(1, -1, 1, 1)
        
        out = torch.matmul(attn, v)  # (B, n_heads, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)
        
        return out, attn


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation."""
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden = dim * mult
        self.fc1 = nn.Linear(dim, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm architecture."""
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0, non_causal: bool = False):
        super().__init__()
        self.ln1 = RMSNorm(dim)
        self.attn = MultiHeadAttention(dim, n_heads, dropout, non_causal)
        self.ln2 = RMSNorm(dim)
        self.mlp = FeedForward(dim, dropout=dropout)
        self.bypass = False  # For layer ablation experiments

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.bypass:
            return x, None
        attn_out, attn_weights = self.attn(self.ln1(x))
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, attn_weights


class TinyGPT(nn.Module):
    """
    Minimal GPT for Bayesian wind tunnel experiments.
    
    Args:
        vocab_size: Size of the vocabulary
        dim: Embedding dimension
        n_layers: Number of transformer blocks
        n_heads: Number of attention heads
        max_seq_len: Maximum sequence length
        dropout: Dropout rate
    """
    def __init__(
        self,
        vocab_size: int,
        dim: int = 192,
        n_layers: int = 6,
        n_heads: int = 6,
        max_seq_len: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers
        
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.drop = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, n_heads, dropout) for _ in range(n_layers)
        ])
        
        self.ln_f = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        
        # Weight tying
        self.head.weight = self.tok_emb.weight
        
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, 
        x: torch.Tensor, 
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input token IDs of shape (B, T)
            targets: Optional target IDs for computing loss
            
        Returns:
            logits: Output logits of shape (B, T, vocab_size)
            loss: Cross-entropy loss if targets provided, else None
        """
        B, T = x.shape
        device = x.device
        
        pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(pos)
        h = self.drop(h)
        
        for block in self.blocks:
            h, _ = block(h)
        
        h = self.ln_f(h)
        logits = self.head(h)
        
        loss = None
        if targets is not None:
            # Flatten for cross-entropy, ignoring -100 indices
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-100
            )
        
        return logits, loss

    def get_attention_maps(self, x: torch.Tensor) -> list:
        """Get attention maps from all layers."""
        B, T = x.shape
        device = x.device
        
        pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(pos)
        h = self.drop(h)
        
        attention_maps = []
        for block in self.blocks:
            h, attn = block(h)
            attention_maps.append(attn)
        
        return attention_maps


def load_tinygpt(checkpoint_path: str, device: str = "cuda") -> TinyGPT:
    """Load a TinyGPT model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract config from checkpoint
    config = ckpt.get("config", {})
    state_dict = ckpt.get("model", ckpt)
    
    # Try to get vocab_size from config (may use different keys)
    vocab_size = config.get("vocab_size") or config.get("V")
    dim = config.get("dim")
    n_layers = config.get("n_layers")
    n_heads = config.get("n_heads")
    
    # Infer max_seq_len from pos_emb in state dict
    max_seq_len = state_dict.get("pos_emb.weight", torch.zeros(128, 1)).shape[0]
    
    # If not in config, infer from state dict
    if vocab_size is None or dim is None or n_layers is None:
        vocab_size = state_dict["tok_emb.weight"].shape[0]
        dim = state_dict["tok_emb.weight"].shape[1]
        n_layers = sum(1 for k in state_dict if k.startswith("blocks.") and k.endswith(".ln1.scale"))
    
    if n_heads is None:
        n_heads = 6  # Default
    
    model = TinyGPT(
        vocab_size=vocab_size,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        max_seq_len=max_seq_len,
    )
    
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    
    return model
