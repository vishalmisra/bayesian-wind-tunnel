"""
HMM Wind Tunnel Data Generation

Generates Hidden Markov Model instances with:
- Transition matrix T (S×S)
- Emission matrix E (S×O)  
- Observation sequence o[0:K-1]
- Ground-truth posteriors p(s_t | o[0:t])

Key: Ground truth is computed using exact discretized parameters 
that the model receives as input (prevents precision ceiling).
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import torch


@dataclass
class HMMConfig:
    """Configuration for HMM data generation."""
    n_states: int = 5           # Number of hidden states (S)
    n_observations: int = 5     # Number of observation symbols (O)
    sequence_length: int = 15   # Default sequence length (K)
    min_prob: float = 0.05      # Minimum probability in matrices
    decimals: int = 2           # Decimal places for discretization
    seed: int = 42


@dataclass  
class HMMInstance:
    """Single HMM instance with ground-truth posteriors."""
    T_discrete: np.ndarray      # Transition matrix (S, S)
    E_discrete: np.ndarray      # Emission matrix (S, O)
    observations: List[int]     # Observation sequence (length K)
    posteriors: np.ndarray      # Ground-truth posteriors (K, S)


def _np_logsumexp(x: np.ndarray) -> float:
    """Numerically stable logsumexp."""
    m = np.max(x)
    return float(m + np.log(np.sum(np.exp(x - m))))


def sample_stochastic_matrix(
    n_rows: int, 
    n_cols: int, 
    min_prob: float = 0.05, 
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Sample a row-stochastic matrix with minimum probability constraint.
    Returns shape (n_rows, n_cols), rows sum to 1.
    """
    if rng is None:
        rng = np.random.default_rng()
    matrix = np.zeros((n_rows, n_cols), dtype=np.float64)
    for i in range(n_rows):
        raw = rng.dirichlet(alpha=np.ones(n_cols, dtype=np.float64))
        raw = np.maximum(raw, min_prob)
        matrix[i] = raw / raw.sum()
    return matrix


def discretize_and_renormalize(matrix: np.ndarray, decimals: int = 2) -> np.ndarray:
    """
    Round stochastic matrix to specified decimals and adjust rows 
    to sum exactly to 1.0 while respecting min/max constraints.
    """
    mat = np.round(matrix.astype(np.float64), decimals=decimals)
    S, C = mat.shape
    out = np.zeros_like(mat)
    
    min_cents = 5  # 0.05 minimum
    
    for i in range(S):
        # Convert to integer hundredths to operate on grid
        cents = np.rint(mat[i] * 100.0).astype(int)
        cents = np.maximum(cents, min_cents)
        total = int(cents.sum())
        diff = 100 - total
        
        if diff > 0:
            # Add to smallest entries first
            order = np.argsort(cents)
            idx = 0
            while diff > 0:
                j = order[idx % C]
                if cents[j] < 100:
                    cents[j] += 1
                    diff -= 1
                idx += 1
        elif diff < 0:
            # Subtract from largest entries first
            order = np.argsort(-cents)
            idx = 0
            while diff < 0:
                j = order[idx % C]
                if cents[j] > min_cents:
                    cents[j] -= 1
                    diff += 1
                idx += 1
                    
        out[i] = cents.astype(np.float64) / 100.0
    return out


def forward_algorithm(
    pi: np.ndarray, 
    T: np.ndarray, 
    E: np.ndarray, 
    observations: List[int]
) -> np.ndarray:
    """
    Compute posterior p(s_t | o[0:t]) for all t using log-space forward pass.
    
    Args:
        pi: Initial state distribution (S,)
        T: Transition matrix (S, S)
        E: Emission matrix (S, O)
        observations: Observation sequence (length K)
        
    Returns:
        posteriors: Shape (K, S), rows sum to 1
    """
    K = len(observations)
    S = pi.shape[0]
    
    log_pi = np.log(pi, dtype=np.float64)
    log_T = np.log(T, dtype=np.float64)
    log_E = np.log(E, dtype=np.float64)
    
    log_alpha = np.zeros((K, S), dtype=np.float64)
    
    # t = 0
    log_alpha[0] = log_pi + log_E[:, observations[0]]
    log_alpha[0] -= _np_logsumexp(log_alpha[0])
    
    # t >= 1
    for t in range(1, K):
        for j in range(S):
            log_alpha[t, j] = _np_logsumexp(log_alpha[t - 1] + log_T[:, j]) + log_E[j, observations[t]]
        log_alpha[t] -= _np_logsumexp(log_alpha[t])
        
    return np.exp(log_alpha, dtype=np.float64)


def generate_hmm_instance(
    cfg: HMMConfig,
    rng: Optional[np.random.Generator] = None,
    sequence_length: Optional[int] = None
) -> HMMInstance:
    """
    Generate a single HMM instance with ground-truth posteriors.
    
    Steps:
    1. Sample raw T, E matrices from Dirichlet
    2. Discretize to cfg.decimals decimal places
    3. Generate observation sequence from HMM
    4. Compute exact posteriors using forward algorithm
    """
    if rng is None:
        rng = np.random.default_rng(cfg.seed)
    
    K = sequence_length if sequence_length else cfg.sequence_length
    S = cfg.n_states
    O = cfg.n_observations
    
    # Sample and discretize matrices
    T_raw = sample_stochastic_matrix(S, S, cfg.min_prob, rng)
    E_raw = sample_stochastic_matrix(S, O, cfg.min_prob, rng)
    
    T_discrete = discretize_and_renormalize(T_raw, cfg.decimals)
    E_discrete = discretize_and_renormalize(E_raw, cfg.decimals)
    
    # Initial distribution (uniform)
    pi = np.ones(S, dtype=np.float64) / S
    
    # Generate sequence
    observations = []
    states = []
    
    # Initial state
    state = rng.choice(S, p=pi)
    states.append(state)
    obs = rng.choice(O, p=E_discrete[state])
    observations.append(int(obs))
    
    # Subsequent steps
    for _ in range(1, K):
        state = rng.choice(S, p=T_discrete[state])
        states.append(state)
        obs = rng.choice(O, p=E_discrete[state])
        observations.append(int(obs))
    
    # Compute ground-truth posteriors
    posteriors = forward_algorithm(pi, T_discrete, E_discrete, observations)
    
    return HMMInstance(
        T_discrete=T_discrete,
        E_discrete=E_discrete,
        observations=observations,
        posteriors=posteriors
    )


class HMMTokenizer:
    """
    Tokenizer for HMM sequences.
    
    Vocabulary:
    - Probability tokens: 0.05, 0.06, ..., 1.00 (96 tokens)
    - Special tokens: [TRANS], [EMIT], [SEP], [OBS]
    - Observation symbols: 0, 1, 2, 3, 4
    """
    
    def __init__(self, n_states: int = 5, n_obs: int = 5):
        self.n_states = n_states
        self.n_obs = n_obs
        
        # Probability tokens (0.05 to 1.00 in steps of 0.01)
        self.prob_tokens = [round(0.05 + i * 0.01, 2) for i in range(96)]
        self.prob_to_id = {p: i for i, p in enumerate(self.prob_tokens)}
        
        # Special tokens
        self.id_trans = 96
        self.id_emit = 97
        self.id_sep = 98
        self.id_obs = 99
        
        # Observation symbol tokens
        self.obs_offset = 100
        
        self.vocab_size = 100 + n_obs  # 105 for n_obs=5
        
    def encode_prob(self, p: float) -> int:
        """Encode probability to token ID."""
        p_rounded = round(p, 2)
        if p_rounded not in self.prob_to_id:
            # Find closest
            idx = min(range(len(self.prob_tokens)), 
                     key=lambda i: abs(self.prob_tokens[i] - p_rounded))
            return idx
        return self.prob_to_id[p_rounded]
    
    def encode_instance(self, instance: HMMInstance) -> List[int]:
        """
        Encode HMM instance to token sequence.
        
        Format:
        [TRANS] T[0,0] T[0,1] ... T[S-1,S-1]
        [EMIT] E[0,0] E[0,1] ... E[S-1,O-1]
        [SEP]
        [OBS] obs[0] [OBS] obs[1] ... [OBS] obs[K-1]
        """
        tokens = []
        
        # Encode transition matrix
        tokens.append(self.id_trans)
        for row in instance.T_discrete:
            for p in row:
                tokens.append(self.encode_prob(p))
        
        # Encode emission matrix
        tokens.append(self.id_emit)
        for row in instance.E_discrete:
            for p in row:
                tokens.append(self.encode_prob(p))
        
        # Separator
        tokens.append(self.id_sep)
        
        # Encode observations
        for obs in instance.observations:
            tokens.append(self.id_obs)
            tokens.append(self.obs_offset + obs)
        
        return tokens
    
    def get_observation_positions(self, tokens: List[int]) -> List[int]:
        """
        Get positions where observation symbols appear (for loss computation).
        """
        positions = []
        for i, tok in enumerate(tokens):
            if tok >= self.obs_offset and tok < self.obs_offset + self.n_obs:
                positions.append(i)
        return positions


class HMMDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for HMM wind tunnel."""
    
    def __init__(
        self, 
        n_samples: int, 
        cfg: HMMConfig,
        tokenizer: HMMTokenizer,
        seed: int = 42
    ):
        self.n_samples = n_samples
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.rng = np.random.default_rng(seed)
        
        # Pre-generate instances
        self.instances = []
        for _ in range(n_samples):
            instance = generate_hmm_instance(cfg, self.rng)
            self.instances.append(instance)
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        instance = self.instances[idx]
        tokens = self.tokenizer.encode_instance(instance)
        
        input_ids = torch.tensor(tokens, dtype=torch.long)
        targets = torch.tensor(instance.posteriors, dtype=torch.float32)
        
        return input_ids, targets


def collate_hmm_batch(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
    """Collate function for HMM batches."""
    input_ids = torch.stack([x[0] for x in batch])
    targets = torch.stack([x[1] for x in batch])
    return input_ids, targets
