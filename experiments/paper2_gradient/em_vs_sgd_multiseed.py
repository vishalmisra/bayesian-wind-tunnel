"""
Paper II: EM-like vs SGD multi-seed experiment
Sticky Markov-chain task from Section 5.2

Run with: python em_vs_sgd_multiseed.py [--seeds 5] [--device cuda]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import json
from pathlib import Path

# ============================================================================
# Data Generation: Sticky Markov Chain
# ============================================================================

def make_transition_matrix(n_symbols=8, self_prob=0.3):
    """
    Create transition matrix with self-transition prob and circular distance weighting.
    """
    P = torch.zeros(n_symbols, n_symbols)
    for i in range(n_symbols):
        for j in range(n_symbols):
            if i == j:
                P[i, j] = self_prob
            else:
                # Circular distance
                dist = min(abs(i - j), n_symbols - abs(i - j))
                P[i, j] = 1.0 / dist
        # Normalize non-self transitions
        P[i, :] = P[i, :] / P[i, :].sum() * (1 - self_prob) + (torch.arange(n_symbols) == i).float() * self_prob
        P[i, :] = P[i, :] / P[i, :].sum()  # Ensure normalization
    return P

def generate_markov_sequence(P, T, start=None):
    """Generate sequence from Markov chain."""
    n_symbols = P.shape[0]
    if start is None:
        start = torch.randint(n_symbols, (1,)).item()

    seq = [start]
    for _ in range(T - 1):
        probs = P[seq[-1]]
        next_sym = torch.multinomial(probs, 1).item()
        seq.append(next_sym)
    return torch.tensor(seq)

def generate_data(T=2000, n_symbols=8, d_embed=20, self_prob=0.3, device='cpu', embed_seed=0):
    """
    Generate sticky Markov chain data.
    Input x_t = μ_{y_{t-1}} + ε_t
    """
    P = make_transition_matrix(n_symbols, self_prob)

    # Symbol embeddings (fixed means) - save and restore random state
    rng_state = torch.get_rng_state()
    torch.manual_seed(embed_seed)  # Fixed embeddings
    mu = torch.randn(n_symbols, d_embed, device=device)
    torch.set_rng_state(rng_state)  # Restore state for sequence generation

    # Generate sequence
    seq = generate_markov_sequence(P, T)
    targets = seq.to(device)  # y_t

    # Inputs: x_t = μ_{y_{t-1}} + noise
    inputs = mu[targets[:-1]] + torch.randn(T-1, d_embed, device=device)
    targets = targets[1:]  # Predict next symbol

    # Theoretical minimum entropy H(Y_t | Y_{t-1})
    H_min = -(P * torch.log(P + 1e-10)).sum(dim=1).mean().item()

    return inputs, targets, P, H_min

# ============================================================================
# Single-Head Attention Model
# ============================================================================

class SingleHeadAttention(nn.Module):
    def __init__(self, d_x=20, d_k=10, d_v=15, n_classes=8):
        super().__init__()
        self.d_k = d_k
        self.W_Q = nn.Parameter(torch.randn(d_x, d_k) * 0.1)
        self.W_K = nn.Parameter(torch.randn(d_x, d_k) * 0.1)
        self.W_V = nn.Parameter(torch.randn(d_x, d_v) * 0.1)
        self.W_O = nn.Parameter(torch.randn(d_v, n_classes) * 0.1)

    def forward(self, x):
        """
        x: (T, d_x)
        Returns: logits (T, n_classes), attention weights (T, T)
        """
        Q = x @ self.W_Q  # (T, d_k)
        K = x @ self.W_K  # (T, d_k)
        V = x @ self.W_V  # (T, d_v)

        # Attention scores (causal mask for autoregressive)
        scores = Q @ K.T / (self.d_k ** 0.5)  # (T, T)

        # Causal mask
        T = x.shape[0]
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))

        alpha = F.softmax(scores, dim=-1)  # (T, T)

        # Attended values
        out = alpha @ V  # (T, d_v)
        logits = out @ self.W_O  # (T, n_classes)

        return logits, alpha


# ============================================================================
# Multi-Layer Transformer Model
# ============================================================================

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        T, d = x.shape

        # Project and reshape for multi-head
        Q = self.W_Q(x).view(T, self.n_heads, self.d_k).transpose(0, 1)  # (H, T, d_k)
        K = self.W_K(x).view(T, self.n_heads, self.d_k).transpose(0, 1)
        V = self.W_V(x).view(T, self.n_heads, self.d_k).transpose(0, 1)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)  # (H, T, T)

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(0), float('-inf'))

        alpha = F.softmax(scores, dim=-1)
        alpha = self.dropout(alpha)

        # Attended values
        out = torch.matmul(alpha, V)  # (H, T, d_k)
        out = out.transpose(0, 1).contiguous().view(T, self.d_model)  # (T, d_model)
        out = self.W_O(out)

        return out, alpha.mean(dim=0)  # Return mean attention across heads


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model

        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        # Pre-norm architecture
        attn_out, alpha = self.attn(self.ln1(x), mask)
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x, alpha


class MultiLayerTransformer(nn.Module):
    def __init__(self, d_x=20, d_model=64, n_layers=4, n_heads=4, n_classes=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Linear(d_x, d_model)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])

        # Output
        self.ln_final = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, n_classes)

    def forward(self, x):
        T = x.shape[0]

        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()

        # Project input
        x = self.input_proj(x)

        # Transformer layers
        alphas = []
        for layer in self.layers:
            x, alpha = layer(x, mask)
            alphas.append(alpha)

        # Output
        x = self.ln_final(x)
        logits = self.output_proj(x)

        return logits, alphas[-1]  # Return last layer attention

# ============================================================================
# Training Protocols
# ============================================================================

def train_sgd(model, inputs, targets, n_steps=1000, lr=0.01):
    """Standard SGD training."""
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    losses = []
    for step in range(n_steps):
        optimizer.zero_grad()
        logits, _ = model(inputs)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses

def train_em_like(model, inputs, targets, n_steps=1000, lr_routing=0.01, lr_value=0.1):
    """
    EM-like training: alternate between routing updates (Q,K,O) and value updates (V).
    """
    losses = []

    for step in range(n_steps):
        # Forward pass
        logits, alpha = model(inputs)
        loss = F.cross_entropy(logits, targets)
        losses.append(loss.item())

        # Compute gradients
        loss.backward()

        with torch.no_grad():
            # M-step: larger update for values (responsibility-weighted)
            if model.W_V.grad is not None:
                model.W_V -= lr_value * model.W_V.grad

            # E-step: smaller update for routing (Q, K, O)
            if model.W_Q.grad is not None:
                model.W_Q -= lr_routing * model.W_Q.grad
            if model.W_K.grad is not None:
                model.W_K -= lr_routing * model.W_K.grad
            if model.W_O.grad is not None:
                model.W_O -= lr_routing * model.W_O.grad

        # Zero gradients
        model.W_Q.grad = None
        model.W_K.grad = None
        model.W_V.grad = None
        model.W_O.grad = None

    return losses

def evaluate(model, inputs, targets):
    """Compute final metrics."""
    with torch.no_grad():
        logits, _ = model(inputs)
        loss = F.cross_entropy(logits, targets).item()

        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean().item()

        preds = logits.argmax(dim=-1)
        acc = (preds == targets).float().mean().item()

    return {'loss': loss, 'entropy': entropy, 'accuracy': acc}


def train_deep_transformer(model, inputs, targets, n_steps=2000, lr=3e-4, device='cpu', inputs_test=None, targets_test=None):
    """Train multi-layer transformer with Adam."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps)

    losses = []
    for step in range(n_steps):
        optimizer.zero_grad()
        logits, _ = model(inputs)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())

        if (step + 1) % 500 == 0:
            # Report both train and test metrics
            train_metrics = evaluate(model, inputs, targets)
            if inputs_test is not None:
                test_metrics = evaluate(model, inputs_test, targets_test)
                print(f"    Step {step+1}: train_loss={train_metrics['loss']:.4f}, test_loss={test_metrics['loss']:.4f}, test_entropy={test_metrics['entropy']:.4f}, test_acc={test_metrics['accuracy']:.4f}")
            else:
                print(f"    Step {step+1}: loss={train_metrics['loss']:.4f}, entropy={train_metrics['entropy']:.4f}, acc={train_metrics['accuracy']:.4f}")

    return losses


def run_deep_experiment(seed, device='cpu', n_steps=2000, n_layers=4, n_heads=4, d_model=64):
    """Run experiment with multi-layer transformer."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Generate TRAINING data
    inputs_train, targets_train, P, H_min = generate_data(T=2000, device=device)

    # Generate separate TEST data (fresh sequence from same Markov chain)
    torch.manual_seed(seed + 10000)  # Different seed for test
    inputs_test, targets_test, _, _ = generate_data(T=2000, device=device)

    # Create and train model
    model = MultiLayerTransformer(
        d_x=20, d_model=d_model, n_layers=n_layers,
        n_heads=n_heads, n_classes=8, dropout=0.1
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_layers} layers, {n_heads} heads, d_model={d_model}, params={n_params:,}")

    losses = train_deep_transformer(model, inputs_train, targets_train, n_steps=n_steps, device=device,
                                     inputs_test=inputs_test, targets_test=targets_test)

    # Evaluate on HELD-OUT test data
    metrics = evaluate(model, inputs_test, targets_test)

    return {
        'seed': seed,
        'H_min': H_min,
        'metrics': metrics,
        'losses': losses,
        'n_params': n_params,
        'config': {'n_layers': n_layers, 'n_heads': n_heads, 'd_model': d_model}
    }

# ============================================================================
# Multi-seed Experiment
# ============================================================================

def run_single_seed(seed, device='cpu', n_steps=1000):
    """Run one seed of EM vs SGD comparison."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Generate data (embeddings fixed, sequence varies by seed)
    inputs, targets, P, H_min = generate_data(T=2000, device=device)

    # Train with SGD
    model_sgd = SingleHeadAttention().to(device)
    torch.manual_seed(seed)  # Same init
    model_sgd.W_Q.data = torch.randn_like(model_sgd.W_Q) * 0.1
    model_sgd.W_K.data = torch.randn_like(model_sgd.W_K) * 0.1
    model_sgd.W_V.data = torch.randn_like(model_sgd.W_V) * 0.1
    model_sgd.W_O.data = torch.randn_like(model_sgd.W_O) * 0.1

    losses_sgd = train_sgd(model_sgd, inputs, targets, n_steps=n_steps)
    metrics_sgd = evaluate(model_sgd, inputs, targets)

    # Train with EM-like
    model_em = SingleHeadAttention().to(device)
    torch.manual_seed(seed)  # Same init
    model_em.W_Q.data = torch.randn_like(model_em.W_Q) * 0.1
    model_em.W_K.data = torch.randn_like(model_em.W_K) * 0.1
    model_em.W_V.data = torch.randn_like(model_em.W_V) * 0.1
    model_em.W_O.data = torch.randn_like(model_em.W_O) * 0.1

    losses_em = train_em_like(model_em, inputs, targets, n_steps=n_steps)
    metrics_em = evaluate(model_em, inputs, targets)

    return {
        'seed': seed,
        'H_min': H_min,
        'sgd': metrics_sgd,
        'em': metrics_em,
        'losses_sgd_full': losses_sgd,
        'losses_em_full': losses_em,
    }

def compute_steps_to_threshold(losses, threshold):
    """Find first step where loss drops below threshold."""
    for i, loss in enumerate(losses):
        if loss < threshold:
            return i
    return len(losses)  # Never reached

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=5, help='Number of seeds')
    parser.add_argument('--n_steps', type=int, default=1000, help='Training steps')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', type=str, default='em_vs_sgd_results.json')
    parser.add_argument('--save_curves', action='store_true', help='Save full loss curves')
    parser.add_argument('--deep', action='store_true', help='Run deep transformer experiment')
    parser.add_argument('--n_layers', type=int, default=4, help='Number of transformer layers (deep mode)')
    parser.add_argument('--n_heads', type=int, default=4, help='Number of attention heads (deep mode)')
    parser.add_argument('--d_model', type=int, default=64, help='Model dimension (deep mode)')
    args = parser.parse_args()

    # Run deep transformer experiment if requested
    if args.deep:
        print(f"Running DEEP TRANSFORMER experiment on {args.device}")
        print(f"Config: {args.n_layers} layers, {args.n_heads} heads, d_model={args.d_model}")
        print(f"Training for {args.n_steps} steps, {args.seeds} seeds")
        print("=" * 60)

        seeds = [42, 2024, 1337, 7777, 9999][:args.seeds]
        results = []

        for seed in seeds:
            print(f"\nSeed {seed}...")
            result = run_deep_experiment(
                seed, device=args.device, n_steps=args.n_steps,
                n_layers=args.n_layers, n_heads=args.n_heads, d_model=args.d_model
            )
            results.append(result)
            print(f"  Final: loss={result['metrics']['loss']:.4f}, entropy={result['metrics']['entropy']:.4f}, acc={result['metrics']['accuracy']:.4f}")
            print(f"  H_min={result['H_min']:.4f}, gap={result['metrics']['entropy'] - result['H_min']:.4f} bits")

        # Aggregate
        print("\n" + "=" * 60)
        print("DEEP TRANSFORMER RESULTS (mean ± std)")
        print("=" * 60)

        H_min = results[0]['H_min']
        losses = [r['metrics']['loss'] for r in results]
        entropies = [r['metrics']['entropy'] for r in results]
        accs = [r['metrics']['accuracy'] for r in results]
        gaps = [r['metrics']['entropy'] - H_min for r in results]

        print(f"\nTheoretical Min Entropy: {H_min:.4f}")
        print(f"Loss:     {np.mean(losses):.4f} ± {np.std(losses):.4f}")
        print(f"Entropy:  {np.mean(entropies):.4f} ± {np.std(entropies):.4f}")
        print(f"Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
        print(f"Gap to optimum: {np.mean(gaps):.4f} ± {np.std(gaps):.4f} bits")

        # Save results
        output = {
            'mode': 'deep',
            'config': results[0]['config'],
            'n_params': results[0]['n_params'],
            'seeds': seeds,
            'n_steps': args.n_steps,
            'H_min': H_min,
            'loss_mean': np.mean(losses),
            'loss_std': np.std(losses),
            'entropy_mean': np.mean(entropies),
            'entropy_std': np.std(entropies),
            'accuracy_mean': np.mean(accs),
            'accuracy_std': np.std(accs),
            'gap_mean': np.mean(gaps),
            'gap_std': np.std(gaps),
            'per_seed': results,
        }

        output_file = args.output.replace('.json', '_deep.json')
        Path(output_file).write_text(json.dumps(output, indent=2, default=float))
        print(f"\nResults saved to {output_file}")
        return

    print(f"Running {args.seeds} seeds on {args.device}")
    print(f"Training for {args.n_steps} steps each")
    print("=" * 60)

    seeds = [42, 2024, 1337, 7777, 9999][:args.seeds]
    results = []

    for seed in seeds:
        print(f"\nSeed {seed}...")
        result = run_single_seed(seed, device=args.device, n_steps=args.n_steps)
        results.append(result)
        print(f"  SGD:  loss={result['sgd']['loss']:.4f}, entropy={result['sgd']['entropy']:.4f}, acc={result['sgd']['accuracy']:.4f}")
        print(f"  EM:   loss={result['em']['loss']:.4f}, entropy={result['em']['entropy']:.4f}, acc={result['em']['accuracy']:.4f}")
        print(f"  H_min={result['H_min']:.4f}")

    # Aggregate statistics
    print("\n" + "=" * 60)
    print("AGGREGATE RESULTS (mean ± std)")
    print("=" * 60)

    H_min = results[0]['H_min']  # Same for all seeds

    sgd_losses = [r['sgd']['loss'] for r in results]
    sgd_entropies = [r['sgd']['entropy'] for r in results]
    sgd_accs = [r['sgd']['accuracy'] for r in results]

    em_losses = [r['em']['loss'] for r in results]
    em_entropies = [r['em']['entropy'] for r in results]
    em_accs = [r['em']['accuracy'] for r in results]

    print(f"\nTheoretical Min Entropy: {H_min:.4f}")
    print(f"\nSGD:")
    print(f"  Loss:     {np.mean(sgd_losses):.4f} ± {np.std(sgd_losses):.4f}")
    print(f"  Entropy:  {np.mean(sgd_entropies):.4f} ± {np.std(sgd_entropies):.4f}")
    print(f"  Accuracy: {np.mean(sgd_accs):.4f} ± {np.std(sgd_accs):.4f}")

    print(f"\nEM-like:")
    print(f"  Loss:     {np.mean(em_losses):.4f} ± {np.std(em_losses):.4f}")
    print(f"  Entropy:  {np.mean(em_entropies):.4f} ± {np.std(em_entropies):.4f}")
    print(f"  Accuracy: {np.mean(em_accs):.4f} ± {np.std(em_accs):.4f}")

    # Save results
    output = {
        'seeds': seeds,
        'n_steps': args.n_steps,
        'H_min': H_min,
        'sgd': {
            'loss_mean': np.mean(sgd_losses),
            'loss_std': np.std(sgd_losses),
            'entropy_mean': np.mean(sgd_entropies),
            'entropy_std': np.std(sgd_entropies),
            'accuracy_mean': np.mean(sgd_accs),
            'accuracy_std': np.std(sgd_accs),
        },
        'em': {
            'loss_mean': np.mean(em_losses),
            'loss_std': np.std(em_losses),
            'entropy_mean': np.mean(em_entropies),
            'entropy_std': np.std(em_entropies),
            'accuracy_mean': np.mean(em_accs),
            'accuracy_std': np.std(em_accs),
        },
        'per_seed': results,
    }

    Path(args.output).write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {args.output}")

    # Convergence analysis: steps to reach SGD's final loss
    print("\n" + "=" * 60)
    print("CONVERGENCE ANALYSIS")
    print("=" * 60)

    sgd_final_mean = np.mean(sgd_losses)
    steps_to_sgd_level = []
    for r in results:
        steps = compute_steps_to_threshold(r['losses_em_full'], sgd_final_mean)
        steps_to_sgd_level.append(steps)

    print(f"SGD final loss (mean): {sgd_final_mean:.4f}")
    print(f"EM steps to reach SGD level: {np.mean(steps_to_sgd_level):.0f} ± {np.std(steps_to_sgd_level):.0f}")
    print(f"  (SGD needs {args.n_steps} steps)")
    if np.mean(steps_to_sgd_level) < args.n_steps:
        speedup = args.n_steps / np.mean(steps_to_sgd_level)
        print(f"  EM is {speedup:.1f}x faster to reach same loss")

    # Statistical test (paired t-test)
    from scipy import stats
    t_stat, p_value = stats.ttest_rel(em_losses, sgd_losses)
    print(f"\nPaired t-test (EM vs SGD loss):")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  p-value: {p_value:.4f}")
    if p_value < 0.05:
        print(f"  EM significantly better than SGD (p < 0.05)")

    # Print LaTeX table format
    print("\n" + "=" * 60)
    print("LATEX TABLE FORMAT")
    print("=" * 60)
    print(f"final_loss_EM & {np.mean(em_losses):.4f} $\\pm$ {np.std(em_losses):.4f} \\\\")
    print(f"final_loss_SGD & {np.mean(sgd_losses):.4f} $\\pm$ {np.std(sgd_losses):.4f} \\\\")
    print(f"final_acc_EM & {np.mean(em_accs):.4f} $\\pm$ {np.std(em_accs):.4f} \\\\")
    print(f"final_acc_SGD & {np.mean(sgd_accs):.4f} $\\pm$ {np.std(sgd_accs):.4f} \\\\")
    print(f"final_entropy_EM & {np.mean(em_entropies):.4f} $\\pm$ {np.std(em_entropies):.4f} \\\\")
    print(f"final_entropy_SGD & {np.mean(sgd_entropies):.4f} $\\pm$ {np.std(sgd_entropies):.4f} \\\\")
    print(f"Theoretical Min entropy & {H_min:.4f} \\\\")

    # Save convergence curves if requested
    if args.save_curves:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))

        # Plot mean ± std for both
        all_em = np.array([r['losses_em_full'] for r in results])
        all_sgd = np.array([r['losses_sgd_full'] for r in results])

        steps = np.arange(args.n_steps)
        ax.plot(steps, all_em.mean(axis=0), 'b-', label='EM-like', linewidth=2)
        ax.fill_between(steps, all_em.mean(axis=0) - all_em.std(axis=0),
                        all_em.mean(axis=0) + all_em.std(axis=0), alpha=0.2, color='blue')

        ax.plot(steps, all_sgd.mean(axis=0), 'r-', label='SGD', linewidth=2)
        ax.fill_between(steps, all_sgd.mean(axis=0) - all_sgd.std(axis=0),
                        all_sgd.mean(axis=0) + all_sgd.std(axis=0), alpha=0.2, color='red')

        ax.axhline(y=H_min, color='green', linestyle='--', label=f'Theoretical min ({H_min:.3f})')
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Cross-Entropy Loss')
        ax.set_title('EM-like vs SGD Convergence (5 seeds)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('em_vs_sgd_convergence.png', dpi=150)
        print(f"\nConvergence plot saved to em_vs_sgd_convergence.png")

if __name__ == '__main__':
    main()
