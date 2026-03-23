#!/usr/bin/env python3
"""
Mechanistic analysis of Mamba on HMM task.
Analyzes layer-wise representations, SSM parameters, and ablations.
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.stats import pearsonr
import seaborn as sns

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv, padding=d_conv-1, groups=self.d_inner)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        A = torch.arange(1, d_state + 1, dtype=torch.float32) * 0.1
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(self.d_inner)

    def forward(self, x, return_intermediates=False):
        B, L, _ = x.shape
        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)
        x_branch = x_branch.transpose(1, 2)
        x_branch = self.conv1d(x_branch)[:, :, :L]
        x_branch = x_branch.transpose(1, 2)
        x_branch = F.silu(x_branch)

        x_dbl = self.x_proj(x_branch)
        dt, B_param, C_param = x_dbl.split([1, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt)).clamp(min=1e-4, max=10.0)
        A = -torch.exp(self.A_log.clamp(max=5.0))

        y, states = self.selective_scan_with_states(x_branch, dt, A, B_param, C_param)
        y = self.norm(y)
        y = y + self.D * x_branch
        y = y * F.silu(z)
        out = self.out_proj(y)

        if return_intermediates:
            return out, {'dt': dt, 'B': B_param, 'C': C_param, 'states': states}
        return out

    def selective_scan_with_states(self, u, dt, A, B, C):
        B_batch, L, d_inner = u.shape
        d_state = self.d_state
        h = torch.zeros(B_batch, d_inner, d_state, device=u.device, dtype=u.dtype)
        ys = []
        states = []
        for i in range(L):
            dA = torch.exp((dt[:, i, :, None] * A).clamp(min=-20, max=0))
            dB_u = (dt[:, i, :, None] * B[:, i, None, :] * u[:, i, :, None]).clamp(-10, 10)
            h = h * dA + dB_u
            h = h.clamp(-100, 100)
            y = (h * C[:, i, None, :]).sum(-1)
            ys.append(y)
            states.append(h.clone())
        return torch.stack(ys, dim=1), states


class MambaHMM(nn.Module):
    def __init__(self, vocab_size, num_states, d_model=256, n_layers=9, d_state=16):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([MambaBlock(d_model, d_state=d_state) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_states)
        self.n_layers = n_layers

    def forward(self, x, return_all_layers=False):
        h = self.tok_emb(x)
        layer_outputs = [h.clone()]
        for block in self.blocks:
            h = h + block(h)
            if torch.isnan(h).any():
                h = torch.nan_to_num(h, nan=0.0)
            layer_outputs.append(h.clone())
        h = self.norm(h)
        logits = self.head(h)
        if return_all_layers:
            return logits, layer_outputs
        return logits

    def forward_with_intermediates(self, x):
        """Forward pass returning all intermediate SSM states."""
        h = self.tok_emb(x)
        all_intermediates = []
        for block in self.blocks:
            residual = h
            out, intermediates = block(h, return_intermediates=True)
            h = residual + out
            all_intermediates.append(intermediates)
        h = self.norm(h)
        logits = self.head(h)
        return logits, all_intermediates

    def forward_ablate_layer(self, x, ablate_idx):
        """Forward pass with one layer ablated (zeroed out)."""
        h = self.tok_emb(x)
        for i, block in enumerate(self.blocks):
            if i == ablate_idx:
                continue  # Skip this layer
            h = h + block(h)
            if torch.isnan(h).any():
                h = torch.nan_to_num(h, nan=0.0)
        h = self.norm(h)
        return self.head(h)


def compute_entropy(probs):
    """Compute entropy from probability distribution."""
    probs = probs.clamp(min=1e-10)
    return -(probs * torch.log2(probs)).sum(dim=-1)


def analyze_layerwise_geometry(model, dataloader, device, num_batches=50):
    """Analyze how representations evolve through layers."""
    print("Analyzing layer-wise geometry...")
    model.eval()

    all_layer_reps = {i: [] for i in range(model.n_layers + 1)}
    all_entropies = []
    all_true_states = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break

            # Unpack tuple batch (input_ids, targets)
            input_ids, targets = batch
            input_ids = input_ids.to(device)
            targets = targets.to(device)  # targets are posterior probabilities

            logits, layer_outputs = model(input_ids, return_all_layers=True)

            # Use targets as posterior probabilities
            # Targets shape is (B, L_target, S) where L_target = sequence_length (observations only)
            B, L_target, S = targets.shape

            # Get representations only at positions corresponding to targets
            # For HMM task: positions 1 to L_target+1 are the observation positions
            # (position 0 is BOS token)
            for i, layer_rep in enumerate(layer_outputs):
                # Take positions 1 to L_target+1 to match target positions
                # layer_rep shape: (B, L_input, d_model)
                reps = layer_rep[:, 1:L_target+1, :].reshape(-1, layer_rep.shape[-1]).cpu().numpy()
                all_layer_reps[i].append(reps)

            # Collect entropies and true states
            posterior_flat = targets.view(-1, S)
            ent = compute_entropy(posterior_flat).cpu().numpy()
            states = posterior_flat.argmax(dim=-1).cpu().numpy()
            all_entropies.append(ent)
            all_true_states.append(states)

    # Concatenate
    for i in all_layer_reps:
        all_layer_reps[i] = np.concatenate(all_layer_reps[i], axis=0)
    all_entropies = np.concatenate(all_entropies)
    all_true_states = np.concatenate(all_true_states)

    return all_layer_reps, all_entropies, all_true_states


def analyze_ssm_parameters(model, dataloader, device, num_batches=20):
    """Analyze the input-dependent SSM parameters (dt, B, C)."""
    print("Analyzing SSM parameters...")
    model.eval()

    layer_params = {i: {'dt': [], 'B': [], 'C': []} for i in range(model.n_layers)}
    all_entropies = []
    all_true_states = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break

            # Unpack tuple batch
            input_ids, targets = batch
            input_ids = input_ids.to(device)
            targets = targets.to(device)

            logits, all_intermediates = model.forward_with_intermediates(input_ids)

            B, L_target, S = targets.shape
            for i, intermediates in enumerate(all_intermediates):
                # Take positions 1 to L_target+1 to match target positions
                dt_slice = intermediates['dt'][:, 1:L_target+1, :].reshape(-1, intermediates['dt'].shape[-1]).cpu().numpy()
                B_slice = intermediates['B'][:, 1:L_target+1, :].reshape(-1, intermediates['B'].shape[-1]).cpu().numpy()
                C_slice = intermediates['C'][:, 1:L_target+1, :].reshape(-1, intermediates['C'].shape[-1]).cpu().numpy()
                layer_params[i]['dt'].append(dt_slice)
                layer_params[i]['B'].append(B_slice)
                layer_params[i]['C'].append(C_slice)

            # Collect entropies and states
            posterior_flat = targets.view(-1, S)
            all_entropies.append(compute_entropy(posterior_flat).cpu().numpy())
            all_true_states.append(posterior_flat.argmax(dim=-1).cpu().numpy())

    # Concatenate
    for i in layer_params:
        for k in layer_params[i]:
            layer_params[i][k] = np.concatenate(layer_params[i][k], axis=0)
    all_entropies = np.concatenate(all_entropies)
    all_true_states = np.concatenate(all_true_states)

    return layer_params, all_entropies, all_true_states


def run_layer_ablations(model, dataloader, device, num_batches=50):
    """Run ablation study removing each layer."""
    print("Running layer ablations...")
    model.eval()

    # Baseline performance
    baseline_mae = 0
    baseline_count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break

            # Unpack tuple batch
            input_ids, targets = batch
            input_ids = input_ids.to(device)
            targets = targets.to(device)  # Posterior probabilities

            B, L_target, S = targets.shape

            logits = model(input_ids)
            # Take only positions 1 to L_target+1 from logits to match targets
            pred_probs = F.softmax(logits[:, 1:L_target+1, :], dim=-1)

            # Compute entropy MAE
            pred_ent = compute_entropy(pred_probs)
            true_ent = compute_entropy(targets)
            baseline_mae += torch.abs(pred_ent - true_ent).sum().item()
            baseline_count += pred_ent.numel()

    baseline_mae /= baseline_count
    print(f"Baseline MAE: {baseline_mae:.6f} bits")

    # Ablate each layer
    ablation_results = []
    for layer_idx in range(model.n_layers):
        layer_mae = 0
        layer_count = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= num_batches:
                    break

                input_ids, targets = batch
                input_ids = input_ids.to(device)
                targets = targets.to(device)

                B, L_target, S = targets.shape

                logits = model.forward_ablate_layer(input_ids, layer_idx)
                pred_probs = F.softmax(logits[:, 1:L_target+1, :], dim=-1)

                pred_ent = compute_entropy(pred_probs)
                true_ent = compute_entropy(targets)
                layer_mae += torch.abs(pred_ent - true_ent).sum().item()
                layer_count += pred_ent.numel()

        layer_mae /= layer_count
        ablation_results.append(layer_mae)
        print(f"Layer {layer_idx} ablated: MAE = {layer_mae:.6f} bits (ratio: {layer_mae/baseline_mae:.2f}x)")

    return baseline_mae, ablation_results


def plot_layerwise_pca(layer_reps, entropies, true_states, output_dir):
    """Plot PCA of representations at each layer."""
    print("Plotting layer-wise PCA...")
    n_layers = len(layer_reps)

    # Select subset for visualization
    n_samples = min(5000, len(entropies))
    idx = np.random.choice(len(entropies), n_samples, replace=False)

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i >= n_layers:
            ax.axis('off')
            continue

        reps = layer_reps[i][idx]
        pca = PCA(n_components=2)
        coords = pca.fit_transform(reps)

        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=true_states[idx],
                           cmap='tab10', alpha=0.5, s=10)
        ax.set_title(f'Layer {i}\nVar: {pca.explained_variance_ratio_[:2].sum():.2%}')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')

    plt.suptitle('Mamba Layer-wise Representations (colored by true HMM state)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'mamba_layerwise_pca_states.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Also plot colored by entropy
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i >= n_layers:
            ax.axis('off')
            continue

        reps = layer_reps[i][idx]
        pca = PCA(n_components=2)
        coords = pca.fit_transform(reps)

        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=entropies[idx],
                           cmap='RdYlBu_r', alpha=0.5, s=10)
        ax.set_title(f'Layer {i}')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')

    plt.colorbar(scatter, ax=axes, label='Posterior Entropy (bits)')
    plt.suptitle('Mamba Layer-wise Representations (colored by posterior entropy)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'mamba_layerwise_pca_entropy.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_entropy_correlation_by_layer(layer_reps, entropies, output_dir):
    """Plot correlation between PC1 and entropy at each layer."""
    print("Computing entropy correlations...")
    n_layers = len(layer_reps)
    correlations = []

    for i in range(n_layers):
        pca = PCA(n_components=1)
        pc1 = pca.fit_transform(layer_reps[i]).flatten()
        corr, _ = pearsonr(pc1, entropies)
        correlations.append(abs(corr))

    plt.figure(figsize=(10, 5))
    plt.bar(range(n_layers), correlations, color='steelblue', edgecolor='black')
    plt.xlabel('Layer')
    plt.ylabel('|Correlation| between PC1 and Entropy')
    plt.title('Mamba: Entropy Encoding Strength by Layer')
    plt.xticks(range(n_layers))
    plt.ylim(0, 1)
    for i, c in enumerate(correlations):
        plt.text(i, c + 0.02, f'{c:.2f}', ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(output_dir / 'mamba_entropy_correlation_by_layer.png', dpi=150)
    plt.close()

    return correlations


def plot_cluster_separation_by_layer(layer_reps, true_states, output_dir):
    """Plot how well clusters separate by layer (silhouette-like metric)."""
    print("Computing cluster separation...")
    from sklearn.metrics import silhouette_score

    n_layers = len(layer_reps)
    silhouettes = []

    # Subsample for speed
    n_samples = min(3000, len(true_states))
    idx = np.random.choice(len(true_states), n_samples, replace=False)

    for i in range(n_layers):
        reps = layer_reps[i][idx]
        states = true_states[idx]

        # Use PCA to reduce dimensionality for silhouette
        pca = PCA(n_components=min(10, reps.shape[1]))
        reps_pca = pca.fit_transform(reps)

        try:
            score = silhouette_score(reps_pca, states)
        except:
            score = 0
        silhouettes.append(score)

    plt.figure(figsize=(10, 5))
    plt.bar(range(n_layers), silhouettes, color='coral', edgecolor='black')
    plt.xlabel('Layer')
    plt.ylabel('Silhouette Score')
    plt.title('Mamba: HMM State Cluster Separation by Layer')
    plt.xticks(range(n_layers))
    for i, s in enumerate(silhouettes):
        plt.text(i, s + 0.01, f'{s:.2f}', ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(output_dir / 'mamba_cluster_separation_by_layer.png', dpi=150)
    plt.close()

    return silhouettes


def plot_ssm_parameter_analysis(layer_params, entropies, true_states, output_dir):
    """Analyze SSM parameters by layer."""
    print("Analyzing SSM parameters...")
    n_layers = len(layer_params)

    # Plot dt (discretization step) statistics by layer
    dt_means = [layer_params[i]['dt'].mean() for i in range(n_layers)]
    dt_stds = [layer_params[i]['dt'].std() for i in range(n_layers)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # dt magnitude by layer
    axes[0].bar(range(n_layers), dt_means, yerr=dt_stds, color='steelblue',
                edgecolor='black', capsize=3)
    axes[0].set_xlabel('Layer')
    axes[0].set_ylabel('Mean Δt')
    axes[0].set_title('Discretization Step (Δt) by Layer')
    axes[0].set_xticks(range(n_layers))

    # B norm by layer (input selectivity)
    B_norms = [np.linalg.norm(layer_params[i]['B'], axis=1).mean() for i in range(n_layers)]
    axes[1].bar(range(n_layers), B_norms, color='coral', edgecolor='black')
    axes[1].set_xlabel('Layer')
    axes[1].set_ylabel('Mean ||B||')
    axes[1].set_title('Input Projection (B) Magnitude by Layer')
    axes[1].set_xticks(range(n_layers))

    # C norm by layer (output selectivity)
    C_norms = [np.linalg.norm(layer_params[i]['C'], axis=1).mean() for i in range(n_layers)]
    axes[2].bar(range(n_layers), C_norms, color='forestgreen', edgecolor='black')
    axes[2].set_xlabel('Layer')
    axes[2].set_ylabel('Mean ||C||')
    axes[2].set_title('Output Projection (C) Magnitude by Layer')
    axes[2].set_xticks(range(n_layers))

    plt.tight_layout()
    plt.savefig(output_dir / 'mamba_ssm_params_by_layer.png', dpi=150)
    plt.close()

    # Correlation of dt with entropy (does the model use dt to control forgetting?)
    dt_entropy_corrs = []
    for i in range(n_layers):
        dt_mean_per_sample = layer_params[i]['dt'].mean(axis=1)  # Average over d_inner
        corr, _ = pearsonr(dt_mean_per_sample, entropies)
        dt_entropy_corrs.append(corr)

    plt.figure(figsize=(10, 5))
    colors = ['steelblue' if c >= 0 else 'coral' for c in dt_entropy_corrs]
    plt.bar(range(n_layers), dt_entropy_corrs, color=colors, edgecolor='black')
    plt.xlabel('Layer')
    plt.ylabel('Correlation(Δt, Entropy)')
    plt.title('Does Mamba Modulate Δt Based on Uncertainty?')
    plt.xticks(range(n_layers))
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    for i, c in enumerate(dt_entropy_corrs):
        plt.text(i, c + 0.02 * np.sign(c), f'{c:.2f}', ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(output_dir / 'mamba_dt_entropy_correlation.png', dpi=150)
    plt.close()

    return dt_entropy_corrs


def plot_ablation_results(baseline_mae, ablation_results, output_dir):
    """Plot ablation study results."""
    print("Plotting ablation results...")
    n_layers = len(ablation_results)
    ratios = [mae / baseline_mae for mae in ablation_results]

    plt.figure(figsize=(10, 5))
    colors = ['coral' if r > 2 else 'steelblue' for r in ratios]
    plt.bar(range(n_layers), ratios, color=colors, edgecolor='black')
    plt.axhline(y=1, color='black', linestyle='--', linewidth=1, label='Baseline')
    plt.xlabel('Layer Ablated')
    plt.ylabel('MAE Ratio (ablated / baseline)')
    plt.title('Mamba Layer Ablation Study: Which Layers Are Critical?')
    plt.xticks(range(n_layers))
    for i, r in enumerate(ratios):
        plt.text(i, r + 0.1, f'{r:.1f}x', ha='center', fontsize=9)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'mamba_layer_ablation.png', dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str, default='./mamba_analysis', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-batches', type=int, default=50)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Import HMM dataset - add project root to path
    # Checkpoint at logs/mamba_hmm/ckpt_best.pt -> project root is 3 levels up
    checkpoint_path = Path(args.checkpoint).resolve()
    project_root = checkpoint_path.parent.parent.parent
    print(f"Checkpoint path: {checkpoint_path}")
    print(f"Project root: {project_root}")
    sys.path.insert(0, str(project_root))

    # Also try absolute path for safety
    import os
    alt_root = os.environ.get('PROJECT_ROOT', str(project_root))
    if alt_root not in sys.path:
        sys.path.insert(0, alt_root)

    from src.data.hmm import HMMConfig, HMMTokenizer, HMMDataset, collate_hmm_batch

    # Setup
    config = HMMConfig(n_states=5, n_observations=5, sequence_length=20)
    tokenizer = HMMTokenizer(config)

    # Load model
    print(f"Loading checkpoint from {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)

    model = MambaHMM(
        vocab_size=tokenizer.vocab_size,
        num_states=config.n_states,
        d_model=256,
        n_layers=9,
        d_state=16
    )
    model.load_state_dict(ckpt['model'])
    model = model.to(args.device)
    model.eval()
    print(f"Loaded model with {sum(p.numel() for p in model.parameters())} parameters")

    # Create dataset
    print("Creating validation dataset...")
    val_dataset = HMMDataset(n_samples=2000, cfg=config, tokenizer=tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           collate_fn=collate_hmm_batch, shuffle=False)

    # Run analyses
    print("\n" + "="*60)
    print("MAMBA MECHANISTIC ANALYSIS")
    print("="*60 + "\n")

    # 1. Layer-wise geometry
    layer_reps, entropies, true_states = analyze_layerwise_geometry(
        model, val_loader, args.device, args.num_batches)

    plot_layerwise_pca(layer_reps, entropies, true_states, output_dir)
    correlations = plot_entropy_correlation_by_layer(layer_reps, entropies, output_dir)
    silhouettes = plot_cluster_separation_by_layer(layer_reps, true_states, output_dir)

    print(f"\nEntropy correlation by layer: {[f'{c:.2f}' for c in correlations]}")
    print(f"Cluster separation by layer: {[f'{s:.2f}' for s in silhouettes]}")

    # 2. SSM parameter analysis
    layer_params, ssm_entropies, ssm_states = analyze_ssm_parameters(model, val_loader, args.device, num_batches=20)
    dt_corrs = plot_ssm_parameter_analysis(layer_params, ssm_entropies, ssm_states, output_dir)
    print(f"\nΔt-entropy correlation by layer: {[f'{c:.2f}' for c in dt_corrs]}")

    # 3. Layer ablations
    baseline_mae, ablation_results = run_layer_ablations(model, val_loader, args.device, args.num_batches)
    plot_ablation_results(baseline_mae, ablation_results, output_dir)

    print(f"\nBaseline MAE: {baseline_mae:.6f} bits")
    print(f"Ablation ratios: {[f'{r/baseline_mae:.1f}x' for r in ablation_results]}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Output saved to: {output_dir}")
    print(f"\nKey findings:")

    # Find critical layers
    critical_layers = [i for i, r in enumerate(ablation_results) if r/baseline_mae > 2]
    print(f"- Critical layers (>2x MAE when ablated): {critical_layers}")

    # Find layer with best cluster separation
    best_cluster_layer = np.argmax(silhouettes)
    print(f"- Best cluster separation at layer {best_cluster_layer} (silhouette={silhouettes[best_cluster_layer]:.2f})")

    # Find layer with strongest entropy encoding
    best_entropy_layer = np.argmax(correlations)
    print(f"- Strongest entropy encoding at layer {best_entropy_layer} (|r|={correlations[best_entropy_layer]:.2f})")


if __name__ == '__main__':
    main()
