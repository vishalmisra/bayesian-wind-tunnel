"""
GPT-Mini Model for HMM Wind Tunnel

A minimal GPT-style transformer for learning Bayesian inference in HMMs.
Produces posterior distributions over hidden states at each observation position.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class GPTMiniConfig:
    """Configuration for GPT-Mini model."""
    vocab_size: int = 105       # HMM tokenizer vocab size
    d_model: int = 256          # Hidden dimension
    n_heads: int = 8            # Number of attention heads
    n_layers: int = 9           # Number of transformer layers
    num_states: int = 5         # Number of HMM states (output dimension)
    max_seq_len: int = 1024     # Maximum sequence length
    dropout: float = 0.1


def _generate_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Generate causal attention mask."""
    mask = torch.triu(torch.ones((seq_len, seq_len), device=device, dtype=torch.bool), diagonal=1)
    return mask


class GPTMini(nn.Module):
    """
    Minimal GPT-style Transformer with causal masking.
    
    Produces per-token hidden states and a head projecting to 
    num_states logits per token (for posterior prediction).
    """
    
    def __init__(self, cfg: GPTMiniConfig):
        super().__init__()
        self.cfg = cfg
        
        # Embeddings
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * 4,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)
        
        # Output head
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.num_states)
        
        self.dropout = nn.Dropout(cfg.dropout)
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)
        if isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: (B, T) token IDs
            
        Returns:
            logits: (B, T, num_states) posterior logits
            hidden: (B, T, d_model) hidden states
        """
        device = input_ids.device
        B, T = input_ids.shape
        
        # Embeddings
        pos = torch.arange(0, T, device=device).unsqueeze(0).expand(B, T)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        x = self.dropout(x)
        
        # Causal attention
        attn_mask = _generate_causal_mask(T, device=device)
        h = self.encoder(x, mask=attn_mask)
        
        # Output
        h = self.ln_f(h)
        logits = self.head(h)
        
        return logits, h
    
    @torch.no_grad()
    def predict_posteriors(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get posterior probabilities."""
        logits, _ = self.forward(input_ids)
        return torch.softmax(logits, dim=-1)


def load_gpt_mini(checkpoint_path: str, device: str = "cuda") -> GPTMini:
    """Load a GPT-Mini model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    # Extract config
    if "config" in ckpt:
        cfg_dict = ckpt["config"]
        cfg = GPTMiniConfig(**cfg_dict)
    else:
        # Infer from state dict
        state_dict = ckpt.get("model", ckpt)
        vocab_size = state_dict["tok_emb.weight"].shape[0]
        d_model = state_dict["tok_emb.weight"].shape[1]
        # Count layers
        n_layers = sum(1 for k in state_dict if "encoder.layers" in k and "self_attn.in_proj_weight" in k)
        if n_layers == 0:
            n_layers = 9  # default
        cfg = GPTMiniConfig(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers)
    
    model = GPTMini(cfg)
    
    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    
    return model
