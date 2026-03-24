"""
Belief Revision Wind Tunnel - Version A
Two-Candidate Antecedent with Delayed Disambiguation

Tests backward binding: model must retrieve an earlier token ID based on late evidence.
This is where transformers should excel and SSMs should struggle.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List
import torch
from torch.utils.data import Dataset


@dataclass
class BeliefRevisionConfig:
    """Configuration for belief revision task."""
    n_candidates: int = 2          # Number of candidate referents (Version A = 2)
    m_filler: int = 8              # Filler tokens between intros
    k_filler: int = 8              # Filler tokens before evidence
    n_entity_ids: int = 100        # Number of possible entity IDs (smaller = easier)
    seed: int = 42


class BeliefRevisionTokenizer:
    """Tokenizer for belief revision task."""
    
    # Special tokens
    PAD = 0
    INTRO = 1      # Marks introduction of an entity
    FILLER = 2     # Filler token
    PRONOUN = 3    # The ambiguous pronoun
    CUE = 4        # Marks the disambiguating cue
    QUERY = 5      # Query position - predict the antecedent ID
    
    # Entity IDs start at offset 10
    ID_OFFSET = 10
    
    def __init__(self, config: BeliefRevisionConfig):
        self.config = config
        self.n_entity_ids = config.n_entity_ids
        # Vocab size: special tokens (0-9) + entity IDs
        self.vocab_size = self.ID_OFFSET + config.n_entity_ids
    
    def encode_id(self, entity_id: int) -> int:
        """Convert entity index (0 to n_entity_ids-1) to token."""
        return entity_id + self.ID_OFFSET
    
    def decode_id(self, token: int) -> int:
        """Convert token back to entity index."""
        return token - self.ID_OFFSET


def generate_episode(
    config: BeliefRevisionConfig,
    tokenizer: BeliefRevisionTokenizer,
    rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """
    Generate a single belief revision episode.
    
    Returns:
        input_tokens: The input sequence
        target_distribution: Bayes-optimal distribution over vocab at query position
        query_pos: Position where prediction is made
        id_tokens: The candidate ID tokens (for analysis)
    """
    N = config.n_candidates
    m = config.m_filler
    k = config.k_filler
    
    # 1) Sample N unique random entity IDs from [0, n_entity_ids)
    entity_indices = rng.choice(config.n_entity_ids, size=N, replace=False)
    id_tokens = np.array([tokenizer.encode_id(idx) for idx in entity_indices])
    
    # 2) Sample latent antecedent Z uniformly
    Z = rng.integers(0, N)
    
    # 3) Build token sequence
    # Structure: INTRO ID_0 FILLER×m INTRO ID_1 FILLER×m ... PRONOUN FILLER×k CUE ID_Z QUERY
    tokens = []
    
    # Introduce each candidate
    for i in range(N):
        tokens.append(tokenizer.INTRO)
        tokens.append(id_tokens[i])
        for _ in range(m):
            tokens.append(tokenizer.FILLER)
    
    # Pronoun (ambiguous reference)
    tokens.append(tokenizer.PRONOUN)
    
    # Filler before evidence
    for _ in range(k):
        tokens.append(tokenizer.FILLER)
    
    # Cue: reveals which candidate is the antecedent
    tokens.append(tokenizer.CUE)
    tokens.append(id_tokens[Z])  # The cue IS the answer (deterministic disambiguation)
    
    # Query position
    tokens.append(tokenizer.QUERY)
    
    input_tokens = np.array(tokens, dtype=np.int64)
    query_pos = len(tokens) - 1
    
    # 4) Compute Bayes-optimal target distribution
    # After seeing CUE + ID_Z, the posterior is deterministic: P(Z) = 1, P(others) = 0
    # The target at QUERY is to predict ID_Z
    target_dist = np.zeros(tokenizer.vocab_size, dtype=np.float32)
    target_dist[id_tokens[Z]] = 1.0
    
    return input_tokens, target_dist, query_pos, id_tokens


def compute_entropy(p: np.ndarray) -> float:
    """Compute entropy of a distribution."""
    p = np.clip(p, 1e-10, 1.0)
    return -np.sum(p * np.log2(p))


class BeliefRevisionDataset(Dataset):
    """PyTorch dataset for belief revision task."""
    
    def __init__(
        self,
        n_samples: int,
        config: BeliefRevisionConfig,
        tokenizer: BeliefRevisionTokenizer,
        seed: int = 42
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.n_samples = n_samples
        self.rng = np.random.default_rng(seed)
        
        # Pre-generate all episodes
        self.episodes = []
        for _ in range(n_samples):
            input_tokens, target_dist, query_pos, id_tokens = generate_episode(
                config, tokenizer, self.rng
            )
            self.episodes.append({
                'input_tokens': input_tokens,
                'target_dist': target_dist,
                'query_pos': query_pos,
                'id_tokens': id_tokens,
                # Bayes entropy at query: 0 bits (deterministic after cue)
                'bayes_entropy': 0.0
            })
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        ep = self.episodes[idx]
        return {
            'input_ids': torch.tensor(ep['input_tokens'], dtype=torch.long),
            'target_dist': torch.tensor(ep['target_dist'], dtype=torch.float),
            'query_pos': ep['query_pos'],
            'id_tokens': torch.tensor(ep['id_tokens'], dtype=torch.long),
            'bayes_entropy': ep['bayes_entropy']
        }


def collate_belief_revision(batch: List[dict]) -> dict:
    """Collate function for DataLoader."""
    # All sequences should be same length in Version A
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
    config = BeliefRevisionConfig(n_candidates=2, m_filler=8, k_filler=8, n_entity_ids=100)
    tokenizer = BeliefRevisionTokenizer(config)
    
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Entity ID range: {tokenizer.ID_OFFSET} to {tokenizer.ID_OFFSET + config.n_entity_ids - 1}")
    
    dataset = BeliefRevisionDataset(5, config, tokenizer, seed=42)
    
    for i in range(3):
        ep = dataset[i]
        print(f"\nEpisode {i}:")
        print(f"  Sequence length: {ep['input_ids'].shape[0]}")
        print(f"  Query position: {ep['query_pos']}")
        print(f"  ID tokens: {ep['id_tokens'].tolist()}")
        print(f"  Target ID: {ep['target_dist'].argmax().item()}")
        print(f"  Bayes entropy: {ep['bayes_entropy']:.4f} bits")
        print(f"  Tokens: {ep['input_ids'].tolist()}")
