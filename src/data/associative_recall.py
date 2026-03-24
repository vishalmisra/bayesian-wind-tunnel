"""
Associative Recall Wind Tunnel

Tests random-access memory: store N key-value pairs, query with late key.
Transformers can attend directly to the key position.
SSMs must compress N pairs into fixed state before knowing which will be queried.

As N increases, SSMs hit information-theoretic limits.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
import torch
from torch.utils.data import Dataset


@dataclass 
class AssociativeRecallConfig:
    """Configuration for associative recall task."""
    n_pairs: int = 64           # Number of key-value pairs to store
    n_queries: int = 3          # Number of queries per sequence  
    n_filler: int = 32          # Filler tokens between sections
    key_vocab: int = 256        # Size of key vocabulary
    value_vocab: int = 256      # Size of value vocabulary
    seed: int = 42


class AssociativeRecallTokenizer:
    """Tokenizer for associative recall task."""
    
    # Special tokens
    PAD = 0
    STORE = 1       # Marks key-value storage
    FILLER = 2      # Filler token
    QUERY = 3       # Query marker
    SEP = 4         # Separator between key and value
    
    # Keys start at offset 10
    KEY_OFFSET = 10
    # Values start after keys
    
    def __init__(self, config: AssociativeRecallConfig):
        self.config = config
        self.key_vocab = config.key_vocab
        self.value_vocab = config.value_vocab
        self.VALUE_OFFSET = self.KEY_OFFSET + config.key_vocab
        self.vocab_size = self.VALUE_OFFSET + config.value_vocab
    
    def encode_key(self, k: int) -> int:
        return k + self.KEY_OFFSET
    
    def encode_value(self, v: int) -> int:
        return v + self.VALUE_OFFSET
    
    def decode_key(self, token: int) -> int:
        return token - self.KEY_OFFSET
    
    def decode_value(self, token: int) -> int:
        return token - self.VALUE_OFFSET


def generate_episode(
    config: AssociativeRecallConfig,
    tokenizer: AssociativeRecallTokenizer,
    rng: np.random.Generator
) -> Tuple[np.ndarray, List[Tuple[int, int]], List[int]]:
    """
    Generate a single associative recall episode.
    
    Returns:
        input_tokens: The input sequence
        queries: List of (query_position, target_value_token) pairs
        kv_pairs: The key-value mapping for analysis
    """
    N = config.n_pairs
    M = config.n_queries
    F = config.n_filler
    
    # Sample N unique keys and N unique values
    keys = rng.choice(config.key_vocab, size=N, replace=False)
    values = rng.choice(config.value_vocab, size=N, replace=False)
    
    # Create key-value mapping
    kv_map = {k: v for k, v in zip(keys, values)}
    
    # Sample M query keys (from the stored keys)
    query_indices = rng.choice(N, size=M, replace=False)
    query_keys = keys[query_indices]
    query_values = values[query_indices]
    
    # Build token sequence
    tokens = []
    
    # Storage section: STORE K SEP V for each pair
    for k, v in zip(keys, values):
        tokens.append(tokenizer.STORE)
        tokens.append(tokenizer.encode_key(k))
        tokens.append(tokenizer.SEP)
        tokens.append(tokenizer.encode_value(v))
    
    # Filler between storage and queries
    for _ in range(F):
        tokens.append(tokenizer.FILLER)
    
    # Query section
    queries = []  # (position, target_token) pairs
    for i, (qk, qv) in enumerate(zip(query_keys, query_values)):
        tokens.append(tokenizer.QUERY)
        tokens.append(tokenizer.encode_key(qk))
        # The next position is where we predict the value
        query_pos = len(tokens)
        target_token = tokenizer.encode_value(qv)
        queries.append((query_pos, target_token))
        # Add placeholder for the answer (model predicts here)
        tokens.append(tokenizer.PAD)  # Will be masked in loss
        
        # Filler between queries (except after last)
        if i < M - 1:
            for _ in range(F // 2):
                tokens.append(tokenizer.FILLER)
    
    return np.array(tokens, dtype=np.int64), queries, list(zip(keys, values))


class AssociativeRecallDataset(Dataset):
    """PyTorch dataset for associative recall task."""
    
    def __init__(
        self,
        n_samples: int,
        config: AssociativeRecallConfig,
        tokenizer: AssociativeRecallTokenizer,
        seed: int = 42
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.n_samples = n_samples
        self.rng = np.random.default_rng(seed)
        
        # Pre-generate all episodes
        self.episodes = []
        for _ in range(n_samples):
            input_tokens, queries, kv_pairs = generate_episode(
                config, tokenizer, self.rng
            )
            self.episodes.append({
                "input_tokens": input_tokens,
                "queries": queries,  # List of (pos, target_token)
                "kv_pairs": kv_pairs
            })
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        ep = self.episodes[idx]
        return {
            "input_ids": torch.tensor(ep["input_tokens"], dtype=torch.long),
            "queries": ep["queries"],
            "kv_pairs": ep["kv_pairs"]
        }


def collate_associative_recall(batch: List[dict]) -> dict:
    """Collate function for DataLoader."""
    max_len = max(item["input_ids"].shape[0] for item in batch)
    
    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    
    # Collect query info
    all_queries = []
    for i, item in enumerate(batch):
        seq_len = item["input_ids"].shape[0]
        input_ids[i, :seq_len] = item["input_ids"]
        # Adjust query positions for batch
        all_queries.append(item["queries"])
    
    return {
        "input_ids": input_ids,
        "queries": all_queries  # List of lists
    }


# Quick test
if __name__ == "__main__":
    config = AssociativeRecallConfig(n_pairs=16, n_queries=3, n_filler=16)
    tokenizer = AssociativeRecallTokenizer(config)
    
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Key range: {tokenizer.KEY_OFFSET} to {tokenizer.KEY_OFFSET + config.key_vocab - 1}")
    print(f"Value range: {tokenizer.VALUE_OFFSET} to {tokenizer.VALUE_OFFSET + config.value_vocab - 1}")
    
    dataset = AssociativeRecallDataset(5, config, tokenizer, seed=42)
    
    for i in range(2):
        ep = dataset[i]
        print(f"\nEpisode {i}:")
        print(f"  Sequence length: {len(ep["input_ids"])}")
        print(f"  Queries: {ep["queries"]}")
        print(f"  First 5 KV pairs: {ep["kv_pairs"][:5]}")
