"""
Shared HF activation extraction utilities for ICL / SULA experiments.

All helpers here are meant to be the single source of truth for how we:
  - register hooks to capture per-head value/key projections at each layer
  - interpret the hook outputs into numpy arrays

Conventions:
  - Values / keys are captured as [batch, seq, n_heads, d_head]
  - For geometry, we typically take the final token: tensor[0, final_pos]
    giving [n_heads, d_head] per layer.
  - Attention is taken from model outputs.attentions, which is already
    [n_layers, batch, n_heads, seq, seq]; we slice to the final query
    position outside this module.

This module also supports optional causal interventions on values/keys
at a single transformer layer via an InterventionConfig passed to HFExtractor.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class InterventionConfig:
    """
    Configuration for optional value/key interventions at one or more layers.

    value_mode:
      - "none":        no value intervention
      - "axis_cut":    remove/attenuate component along a 1D axis
      - "axis_only":   keep only component along the axis
      - "axis_shift":  shift activations along the axis

    key_mode:
      - "none":        no key intervention
      - "key_rotate":  apply a random orthogonal rotation in key space

    axis_source:
      - "true":   load axis from disk (e.g., SULA entropy axis)
      - "random": sample a random unit axis per run
    """

    value_mode: str = "none"
    key_mode: str = "none"
    # Either a single target layer (layer_idx) or a list of target layers
    layer_idx: Optional[int] = None
    layer_indices: Optional[List[int]] = None
    axis_source: str = "true"
    axis_path: Optional[str] = None
    lambda_scale: float = 0.0
    delta_sigma: float = 0.0
    seed: int = 0


class InterventionRuntime:
    """
    Runtime helper that applies configured interventions to values/keys.

    This object is owned by HFExtractor and used inside hooks.
    """

    def __init__(self, cfg: InterventionConfig | None) -> None:
        self.cfg = cfg
        # Per-layer caches for axes and scales
        self._axis_vec_per_layer: Dict[int, np.ndarray] = {}
        self._sigma_pc1_per_layer: Dict[int, float] = {}
        self._random_axis_vec_per_layer: Dict[int, np.ndarray] = {}
        self._key_rotation: Optional[np.ndarray] = None

    @property
    def enabled(self) -> bool:
        return self.cfg is not None and (
            self.cfg.value_mode != "none" or self.cfg.key_mode != "none"
        )

    def _target_layers(self) -> Optional[List[int]]:
        """Return the list of layers this config should apply to."""
        if self.cfg is None:
            return None
        if self.cfg.layer_indices:
            return self.cfg.layer_indices
        if self.cfg.layer_idx is not None:
            return [self.cfg.layer_idx]
        return None

    def _load_true_axis_for_layer(self, layer_idx: int, dim: int) -> None:
        if layer_idx in self._axis_vec_per_layer:
            return
        if self.cfg is None or self.cfg.axis_path is None:
            raise RuntimeError("InterventionConfig.axis_path must be set for axis_source='true'")

        base_path = Path(self.cfg.axis_path)
        if base_path.is_file():
            axis_path = base_path
        else:
            axis_path = base_path / f"entropy_axis_L{layer_idx}_model.npz"

        data = np.load(axis_path)
        if "u_ent" not in data or "sigma_pc1" not in data:
            raise RuntimeError(f"Axis file {axis_path} missing 'u_ent' or 'sigma_pc1'")
        u_ent = np.asarray(data["u_ent"], dtype=np.float32).reshape(-1)
        sigma_pc1 = float(data["sigma_pc1"])
        if not np.isfinite(sigma_pc1) or sigma_pc1 <= 0.0:
            sigma_pc1 = float(1.0)
        norm = float(np.linalg.norm(u_ent))
        if not np.isfinite(norm) or norm == 0.0:
            raise RuntimeError(f"Degenerate u_ent loaded from {axis_path}")
        u_norm = (u_ent / norm).astype(np.float32)
        if u_norm.shape[0] != dim:
            raise RuntimeError(
                f"Loaded axis dim {u_norm.shape[0]} does not match value dim {dim} for layer {layer_idx}"
            )
        self._axis_vec_per_layer[layer_idx] = u_norm
        self._sigma_pc1_per_layer[layer_idx] = sigma_pc1

    def _get_axis(self, dim: int, layer_idx: int) -> Tuple[np.ndarray, float]:
        """
        Return a unit axis vector (np.ndarray) and sigma_pc1 scalar.
        """
        if self.cfg is None:
            raise RuntimeError("InterventionRuntime used without config")
        if self.cfg.axis_source == "true":
            self._load_true_axis_for_layer(layer_idx, dim)
            u = self._axis_vec_per_layer[layer_idx]
            sigma_pc1 = self._sigma_pc1_per_layer.get(layer_idx, 1.0)
            return u, sigma_pc1

        # axis_source == "random"
        if layer_idx not in self._random_axis_vec_per_layer or self._random_axis_vec_per_layer[
            layer_idx
        ].shape[0] != dim:
            # Offset seed by layer to decorrelate per-layer random axes.
            base_seed = self.cfg.seed if self.cfg is not None else 0
            rng = np.random.default_rng(base_seed + layer_idx)
            g = rng.normal(size=(dim,)).astype(np.float32)
            norm = float(np.linalg.norm(g))
            if norm == 0.0 or not np.isfinite(norm):
                g = np.zeros_like(g)
                g[0] = 1.0
                norm = 1.0
            self._random_axis_vec_per_layer[layer_idx] = g / norm
        u = self._random_axis_vec_per_layer[layer_idx]

        sigma_pc1 = 1.0
        # Optionally reuse sigma from true-axis files if available.
        if self.cfg.axis_path is not None and layer_idx not in self._sigma_pc1_per_layer:
            base_path = Path(self.cfg.axis_path)
            if base_path.is_file():
                axis_path = base_path
            else:
                axis_path = base_path / f"entropy_axis_L{layer_idx}_model.npz"
            try:
                data = np.load(axis_path)
                sigma_pc1_loaded = float(data["sigma_pc1"])
                if np.isfinite(sigma_pc1_loaded) and sigma_pc1_loaded > 0.0:
                    self._sigma_pc1_per_layer[layer_idx] = sigma_pc1_loaded
            except Exception:
                pass
        if layer_idx in self._sigma_pc1_per_layer:
            sigma_pc1 = self._sigma_pc1_per_layer[layer_idx]
        return u, sigma_pc1

    def _get_key_rotation(self, dim: int) -> np.ndarray:
        """
        Lazily create a random orthogonal matrix R ∈ R^{dim×dim}.
        """
        if self._key_rotation is not None and self._key_rotation.shape == (dim, dim):
            return self._key_rotation
        rng = np.random.default_rng(self.cfg.seed if self.cfg is not None else None)
        A = rng.normal(size=(dim, dim)).astype(np.float32)
        Q, _ = np.linalg.qr(A)
        self._key_rotation = Q.astype(np.float32)
        return self._key_rotation

    def apply_value(self, layer_idx: int, value_tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply the configured value intervention to a tensor of shape [B, T, H, d_head].
        """
        target_layers = self._target_layers()
        if (
            not self.enabled
            or self.cfg is None
            or target_layers is None
            or layer_idx not in target_layers
            or self.cfg.value_mode == "none"
        ):
            return value_tensor

        B, T, H, D = value_tensor.shape
        dim = H * D
        u_np, sigma_pc1 = self._get_axis(dim, layer_idx)
        u = torch.from_numpy(u_np).to(device=value_tensor.device, dtype=value_tensor.dtype)

        v_flat = value_tensor.reshape(B * T, dim)
        coeff = v_flat @ u  # [B*T]

        if self.cfg.value_mode == "axis_cut":
            v_par = coeff.unsqueeze(1) * u.unsqueeze(0)
            v_perp = v_flat - v_par
            v_new = v_perp + self.cfg.lambda_scale * v_par
        elif self.cfg.value_mode == "axis_only":
            v_new = coeff.unsqueeze(1) * u.unsqueeze(0)
        elif self.cfg.value_mode == "axis_shift":
            shift = self.cfg.delta_sigma * float(sigma_pc1)
            v_par = (coeff + shift).unsqueeze(1) * u.unsqueeze(0)
            v_perp = v_flat - coeff.unsqueeze(1) * u.unsqueeze(0)
            v_new = v_perp + v_par
        else:
            return value_tensor

        return v_new.reshape(B, T, H, D)

    def apply_keys(self, layer_idx: int, key_tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply the configured key intervention to a tensor of shape [B, T, H, d_head].
        """
        target_layers = self._target_layers()
        if (
            not self.enabled
            or self.cfg is None
            or target_layers is None
            or layer_idx not in target_layers
            or self.cfg.key_mode == "none"
        ):
            return key_tensor

        if self.cfg.key_mode == "key_rotate":
            B, T, H, D = key_tensor.shape
            R_np = self._get_key_rotation(D)
            R = torch.from_numpy(R_np).to(device=key_tensor.device, dtype=key_tensor.dtype)
            k_flat = key_tensor.reshape(B * T * H, D)
            k_rot = k_flat @ R
            return k_rot.reshape(B, T, H, D)

        return key_tensor


def load_hf_causal_lm(model_name: str) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a HF causal LM and tokenizer with standard settings for these experiments.

    This centralizes dtype / device_map choices so all experiments are consistent.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model.eval()
    if hasattr(model, "set_attn_implementation"):
        try:
            model.set_attn_implementation("eager")
        except Exception:
            # Not all models support this, so we fail soft.
            pass
    return model, tokenizer


def register_hooks(
    model: AutoModelForCausalLM,
    model_name: str,
    intervention: Optional[InterventionRuntime] = None,
) -> Dict[str, torch.Tensor]:
    """
    Register forward hooks to capture per-head key/value projections at each layer.

    Supported architectures:
      - GPT-NeoX style (e.g., Pythia): model.gpt_neox.layers[*].attention.query_key_value
      - LLaMA / Phi-2 style: model.model.layers[*].self_attn.k_proj / v_proj
      - GPT-J / GPT-2 style with transformer.h[*].attn.query_key_value
    """
    hooks: Dict[str, torch.Tensor] = {}
    num_heads = getattr(model.config, "num_attention_heads", None) or getattr(
        model.config, "n_head", None
    )
    if num_heads is None:
        raise RuntimeError(f"Cannot determine num_heads for model {model_name}")

    def flatten_attn(tensor: torch.Tensor) -> torch.Tensor:
        """
        Reshape a [B, T, hidden] projection into [B, T, n_heads, d_head].
        """
        hidden_dim = tensor.shape[-1]
        if hidden_dim % num_heads != 0:
            raise RuntimeError(
                f"Hidden dim ({hidden_dim}) not divisible by num_heads ({num_heads})"
            )
        head_dim_local = hidden_dim // num_heads
        return (
            tensor.reshape(tensor.shape[0], tensor.shape[1], num_heads, head_dim_local)
            .detach()
        )

    def qkv_hook(layer_idx: int):
        def hook(module, input, output):
            # output: [B, T, 3 * hidden] or already [B, T, 3, n_heads * d_head]
            tensor = output
            hidden_dim = tensor.shape[-1]
            if hidden_dim % (3 * num_heads) != 0:
                # Unexpected shape; skip capturing/intervening for this module.
                return
            head_dim_local = hidden_dim // (3 * num_heads)
            tensor_reshaped = tensor.reshape(
                tensor.shape[0], tensor.shape[1], 3, num_heads, head_dim_local
            )
            # index 1 = keys, 2 = values
            keys = tensor_reshaped[:, :, 1]
            values = tensor_reshaped[:, :, 2]

            # Apply optional interventions
            if intervention is not None and intervention.enabled:
                keys = intervention.apply_keys(layer_idx, keys)
                values = intervention.apply_value(layer_idx, values)
                tensor_reshaped[:, :, 1] = keys
                tensor_reshaped[:, :, 2] = values

            # Record hooks for downstream geometry
            hooks[f"{layer_idx}_k"] = keys.detach()
            hooks[f"{layer_idx}_v"] = values.detach()

            # Return possibly modified tensor to continue the forward pass
            return tensor_reshaped.reshape(
                tensor.shape[0], tensor.shape[1], 3 * num_heads * head_dim_local
            )

        return hook

    def projection_hook(layer_idx: int, proj_name: str):
        def hook(module, input, output):
            tensor = output
            proj_flat = flatten_attn(tensor)

            if intervention is not None and intervention.enabled:
                if proj_name == "k":
                    proj_flat = intervention.apply_keys(layer_idx, proj_flat)
                elif proj_name == "v":
                    proj_flat = intervention.apply_value(layer_idx, proj_flat)

            hooks[f"{layer_idx}_{proj_name}"] = proj_flat.detach()

            # Recombine heads back to [B, T, hidden] for the model
            B, T, n_heads_local, d_head_local = proj_flat.shape
            hidden_dim = n_heads_local * d_head_local
            return proj_flat.reshape(B, T, hidden_dim)

        return hook

    for layer_idx in range(model.config.num_hidden_layers):
        if hasattr(model, "gpt_neox"):
            # Pythia-style GPT-NeoX
            layer = model.gpt_neox.layers[layer_idx]
            layer.attention.query_key_value.register_forward_hook(qkv_hook(layer_idx))
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            # LLaMA / Phi-2 style
            layer = model.model.layers[layer_idx]
            layer.self_attn.k_proj.register_forward_hook(
                projection_hook(layer_idx, "k")
            )
            layer.self_attn.v_proj.register_forward_hook(
                projection_hook(layer_idx, "v")
            )
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            # GPT-J / GPT-2 style
            layer = model.transformer.h[layer_idx]
            layer.attn.query_key_value.register_forward_hook(qkv_hook(layer_idx))
        else:
            # If we hit an unsupported architecture, log via print and stop registering.
            print(f"[hf_activations] Unsupported model architecture for hooks: {model_name}")
            break
    return hooks


def extract_value_key(
    hooks: Dict[str, torch.Tensor],
    layer_idx: int,
    proj: str,
    final_pos: int,
) -> np.ndarray:
    """
    Extract a single layer's value or key at the final token position.

    Returns:
      np.ndarray with shape [n_heads, d_head] in float32.
    """
    key = f"{layer_idx}_{proj}"
    tensor = hooks.get(key)
    if tensor is None:
        raise RuntimeError(f"Missing hook data for {key}")
    # hooks store [B, T, n_heads, d_head]; we take batch 0, position final_pos
    tensor = tensor[0, final_pos].to(torch.float32)
    return tensor.cpu().numpy()


class HFExtractor:
    """
    Unified wrapper around a HF causal LM exposing a common activation interface.

    This is the main entry point that other scripts should use instead of rolling
    their own hooks. It provides:
      - tokenize(prompt) -> input_ids, attention_mask tensors on the right device
      - extract_values(tokens)  -> [layers, heads, head_dim] at final token
      - extract_keys(tokens)    -> [layers, heads, head_dim] at final token
      - extract_attention(tokens) -> [layers, heads, seq, seq]
      - extract_next_token_distribution(tokens) -> [vocab]
    """

    def __init__(self, model_name: str, intervention: InterventionConfig | None = None) -> None:
        self.model_name = model_name
        self.model, self.tokenizer = load_hf_causal_lm(model_name)
        self.device = next(self.model.parameters()).device
        # Single shared hook registry reused across prompts
        self.intervention_runtime = InterventionRuntime(intervention)
        self.hooks: Dict[str, torch.Tensor] = register_hooks(
            self.model, model_name, intervention=self.intervention_runtime
        )

    def tokenize(self, prompt: str) -> Dict[str, torch.Tensor]:
        """Tokenize a single prompt and move tensors to the model device."""
        tokens = self.tokenizer(prompt, return_tensors="pt")
        return {k: v.to(self.device) for k, v in tokens.items()}

    # ---------------------------
    # Low-level run helper
    # ---------------------------
    def _run_and_collect(
        self, tokens: Dict[str, torch.Tensor]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Run the model once and collect values, keys, attention, and next-token probs.

        Returns:
          values: [layers, heads, d_head]
          keys:   [layers, heads, d_head]
          attn:   [layers, heads, seq, seq]
          probs:  [vocab]
        """
        with torch.no_grad():
            outputs = self.model(
                **tokens,
                output_attentions=True,
                output_hidden_states=False,
                use_cache=False,
            )
        input_ids = tokens["input_ids"]
        final_pos = input_ids.shape[1] - 1

        # Collect per-layer values / keys at final token
        n_layers = self.model.config.num_hidden_layers
        layer_values: List[np.ndarray] = []
        layer_keys: List[np.ndarray] = []
        for layer_idx in range(n_layers):
            v = extract_value_key(self.hooks, layer_idx, "v", final_pos)
            layer_values.append(v)
            try:
                k = extract_value_key(self.hooks, layer_idx, "k", final_pos)
            except RuntimeError:
                # Some architectures only expose v-projections via hooks; fall back to zeros.
                k = np.zeros_like(v)
            layer_keys.append(k)
        values_arr = np.stack(layer_values)  # [layers, heads, d_head]
        keys_arr = np.stack(layer_keys)      # [layers, heads, d_head]

        # Attention: list[n_layers] each [batch, heads, seq, seq]
        if outputs.attentions is None:
            # Fallback: zeros with minimal shape
            n_heads = getattr(self.model.config, "num_attention_heads", 1)
            seq_len = int(input_ids.shape[1])
            attn = np.zeros((n_layers, n_heads, seq_len, seq_len), dtype=np.float32)
        else:
            attn = np.stack(
                [attn_t[0].float().cpu().numpy() for attn_t in outputs.attentions]
            )  # [layers, heads, seq, seq]

        # Next-token distribution at final position
        logits = outputs.logits[:, -1, :]  # [batch, vocab]
        probs = torch.softmax(logits, dim=-1)[0].to(torch.float32).cpu().numpy()

        return values_arr, keys_arr, attn, probs

    # ---------------------------
    # Public API
    # ---------------------------
    def extract_values(self, tokens: Dict[str, torch.Tensor]) -> np.ndarray:
        values, _, _, _ = self._run_and_collect(tokens)
        return values

    def extract_keys(self, tokens: Dict[str, torch.Tensor]) -> np.ndarray:
        _, keys, _, _ = self._run_and_collect(tokens)
        return keys

    def extract_attention(self, tokens: Dict[str, torch.Tensor]) -> np.ndarray:
        _, _, attn, _ = self._run_and_collect(tokens)
        return attn

    def extract_next_token_distribution(self, tokens: Dict[str, torch.Tensor]) -> np.ndarray:
        _, _, _, probs = self._run_and_collect(tokens)
        return probs



