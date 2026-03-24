"""
Belief Revision Wind Tunnel - Hard Version
N=16 candidates with Dirichlet(0.3) likelihoods

Tests true Bayesian inference over multiple hypotheses with soft evidence.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List
import torch
from torch.utils.data import Dataset


@dataclass
class BeliefRevisionHardConfig:
    """Configuration for hard belief revision task."""
    n_candidates: int = 16         # Number of candidate referents
    n_evidence: int = 5            # Number of evidence tokens
    m_filler: int = 4              # Filler tokens between intros
    dirichlet_alpha: float = 0.3   # Dirichlet concentration (sparse)
    seed: int = 42


class BeliefRevisionHardTokenizer:
    """Tokenizer for hard belief revision task."""
    
    # Special tokens
    PAD = 0
    INTRO = 1      # Marks introduction of an entity
    FILLER = 2     # Filler token  
    PRONOUN = 3    # The ambiguous pronoun
    EVIDENCE = 4   # Marks evidence token
    QUERY = 5      # Query position - predict posterior
    
    # Candidate IDs start at offset 10 (tokens 10 to 10+N-1)
    ID_OFFSET = 10
    
    def __init__(self, config: BeliefRevisionHardConfig):
        self.config = config
        self.n_candidates = config.n_candidates
        # Vocab size: special tokens (0-9) + N candidate IDs
        self.vocab_size = self.ID_OFFSET + config.n_candidates
    
    def encode_id(self, candidate_idx: int) -> int:
        """Convert candidate index (0 to N-1) to token."""
        return candidate_idx + self.ID_OFFSET
    
    def decode_id(self, token: int) -> int:
        """Convert token back to candidate index."""
        return token - self.ID_OFFSET


class BeliefRevisionHardTask:
    """
    Hard belief revision task with Dirichlet likelihoods.
    
    Structure:
    - N candidates introduced with unique position tokens
    - Latent referent Z sampled uniformly
    - K evidence tokens emitted according to P(evidence | Z)
    - Emission matrix rows sampled from Dirichlet(alpha)
    - Model must output full posterior P(Z | evidence)
    """
    
    def __init__(self, config: BeliefRevisionHardConfig, seed: int = 42):
        self.config = config
        self.N = config.n_candidates
        self.K = config.n_evidence
        self.alpha = config.dirichlet_alpha
        
        # Sample FIXED emission matrix (same for all episodes)
        rng = np.random.default_rng(seed)
        # E[z, i] = P(evidence token = candidate i | true referent = z)
        # Each row sampled from Dirichlet([alpha, alpha, ..., alpha])
        self.emission_matrix = rng.dirichlet(
            np.ones(self.N) * self.alpha, 
            size=self.N
        )  # Shape: (N, N)
        
        # For interpretability, bias toward diagonal (true referent emits its own ID more)
        # Add some diagonal boost then renormalize
        diagonal_boost = np.eye(self.N) * 2.0
        self.emission_matrix = self.emission_matrix + diagonal_boost
        self.emission_matrix = self.emission_matrix / self.emission_matrix.sum(axis=1, keepdims=True)
        
        print(f"Emission matrix (first 4 rows):")
        for z in range(min(4, self.N)):
            top3 = np.argsort(self.emission_matrix[z])[-3:][::-1]
            probs = self.emission_matrix[z, top3]
            print(f"  Z={z}: top emissions -> {list(zip(top3, probs.round(3)))}")
    
    def generate_episode(
        self, 
        tokenizer: BeliefRevisionHardTokenizer,
        rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray, int, float]:
        """
        Generate a single episode.
        
        Returns:
            input_tokens: The input sequence
            target_posterior: Full posterior over N candidates at query position  
            query_pos: Position where prediction is made
            bayes_entropy: Entropy of the Bayes-optimal posterior
        """
        N = self.N
        K = self.K
        m = self.config.m_filler
        
        # 1) Sample latent referent Z uniformly
        Z = rng.integers(0, N)
        
        # 2) Build token sequence
        # Structure: INTRO FILLER×m (×N) ... PRONOUN EVIDENCE e_1 ... EVIDENCE e_K QUERY
        tokens = []
        
        # Introduce each candidate (just position markers, model learns what they mean)
        for i in range(N):
            tokens.append(tokenizer.INTRO)
            tokens.append(tokenizer.encode_id(i))  # Candidate i's ID token
            for _ in range(m):
                tokens.append(tokenizer.FILLER)
        
        # Pronoun (ambiguous reference point)
        tokens.append(tokenizer.PRONOUN)
        
        # Evidence tokens: sample from emission distribution P(e | Z)
        evidence_indices = rng.choice(N, size=K, p=self.emission_matrix[Z])
        for e_idx in evidence_indices:
            tokens.append(tokenizer.EVIDENCE)
            tokens.append(tokenizer.encode_id(e_idx))
        
        # Query position
        tokens.append(tokenizer.QUERY)
        
        input_tokens = np.array(tokens, dtype=np.int64)
        query_pos = len(tokens) - 1
        
        # 3) Compute Bayes-optimal posterior
        # Prior: P(Z = z) = 1/N for all z
        # Likelihood: P(evidence | Z = z) = prod_k emission_matrix[z, e_k]
        # Posterior: P(Z = z | evidence) ∝ prior[z] * likelihood[z]
        
        log_prior = np.log(1.0 / N)
        log_likelihoods = np.zeros(N)
        for z in range(N):
            log_lik = 0.0
            for e_idx in evidence_indices:
                log_lik += np.log(self.emission_matrix[z, e_idx] + 1e-10)
            log_likelihoods[z] = log_lik
        
        log_posterior = log_prior + log_likelihoods
        log_posterior = log_posterior - np.max(log_posterior)  # Numerical stability
        posterior = np.exp(log_posterior)
        posterior = posterior / posterior.sum()
        
        # Target: full posterior over candidate IDs (vocab positions ID_OFFSET to ID_OFFSET+N-1)
        target_posterior = np.zeros(tokenizer.vocab_size, dtype=np.float32)
        for i in range(N):
            target_posterior[tokenizer.encode_id(i)] = posterior[i]
        
        # Entropy of Bayes-optimal posterior
        bayes_entropy = -np.sum(posterior * np.log2(posterior + 1e-10))
        
        return input_tokens, target_posterior, query_pos, bayes_entropy


class BeliefRevisionHardDataset(Dataset):
    """PyTorch dataset for hard belief revision task."""
    
    def __init__(
        self,
        n_samples: int,
        config: BeliefRevisionHardConfig,
        tokenizer: BeliefRevisionHardTokenizer,
        task: BeliefRevisionHardTask,
        seed: int = 42
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.task = task
        self.n_samples = n_samples
        self.rng = np.random.default_rng(seed)
        
        # Pre-generate all episodes
        self.episodes = []
        for _ in range(n_samples):
            input_tokens, target_posterior, query_pos, bayes_entropy = task.generate_episode(
                tokenizer, self.rng
            )
            self.episodes.append({
                'input_tokens': input_tokens,
                'target_posterior': target_posterior,
                'query_pos': query_pos,
                'bayes_entropy': bayes_entropy
            })
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        ep = self.episodes[idx]
        return {
            'input_ids': torch.tensor(ep['input_tokens'], dtype=torch.long),
            'target_dist': torch.tensor(ep['target_posterior'], dtype=torch.float),
            'query_pos': ep['query_pos'],
            'bayes_entropy': ep['bayes_entropy']
        }


def collate_belief_revision_hard(batch: List[dict]) -> dict:
    """Collate function for DataLoader."""
    max_len = max(item['input_ids'].shape[0] for item in batch)
    
    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    target_dists = torch.stack([item['target_dist'] for item in batch])
    query_positions = torch.tensor([item['query_pos'] for item in batch], dtype=torch.long)
    bayes_entropies = torch.tensor([item['bayes_entropy'] for item in batch], dtype=torch.float)
    
    for i, item in enumerate(batch):
        seq_len = item['input_ids'].shape[0]
        input_ids[i, :seq_len] = item['input_ids']
    
    return {
        'input_ids': input_ids,
        'target_dist': target_dists,
        'query_pos': query_positions,
        'bayes_entropy': bayes_entropies
    }


# Quick test
if __name__ == '__main__':
    config = BeliefRevisionHardConfig(n_candidates=16, n_evidence=5, dirichlet_alpha=0.3)
    tokenizer = BeliefRevisionHardTokenizer(config)
    task = BeliefRevisionHardTask(config, seed=42)
    
    print(f"\nVocab size: {tokenizer.vocab_size}")
    print(f"Candidate ID range: {tokenizer.ID_OFFSET} to {tokenizer.ID_OFFSET + config.n_candidates - 1}")

# Quick test
if __name__ == "__main__":
    config = BeliefRevisionHardConfig(n_candidates=16, n_evidence=5, dirichlet_alpha=0.3)
    tokenizer = BeliefRevisionHardTokenizer(config)
    task = BeliefRevisionHardTask(config, seed=42)
    
    print(f"\nVocab size: {tokenizer.vocab_size}")
    print(f"Candidate ID range: {tokenizer.ID_OFFSET} to {tokenizer.ID_OFFSET + config.n_candidates - 1}")
    
    dataset = BeliefRevisionHardDataset(5, config, tokenizer, task, seed=123)
    
    for i in range(3):
        ep = dataset[i]
        print(f"\nEpisode {i}:")
        print(f"  Sequence length: {len(ep["input_ids"])}")
        print(f"  Query position: {ep["query_pos"]}")
        print(f"  Bayes entropy: {ep["bayes_entropy"]:.4f} bits")
        
        # Show posterior
        posterior = ep["target_dist"][tokenizer.ID_OFFSET:tokenizer.ID_OFFSET + config.n_candidates]
        top3 = torch.argsort(posterior, descending=True)[:3]
        for idx in top3:
            prob = posterior[idx].item()
            print(f"    Candidate {idx.item()}: {prob:.3f}")
