"""Hero2Vec embeddings for Overwatch heroes."""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Dataset

from src.utils.heroes import get_hero_metadata


class HeroCooccurrenceDataset(Dataset):
    """Dataset for hero co-occurrence training."""

    def __init__(self, matches: List[Tuple[List[int], List[int]]]) -> None:
        """
        Initialize dataset from matches.

        Args:
            matches: List of (team1, team2) tuples
        """
        self.pairs: List[Tuple[int, int]] = []

        for team1, team2 in matches:
            # Create pairs within teams (positive samples)
            for i, hero1 in enumerate(team1):
                for hero2 in team1[i + 1 :]:
                    self.pairs.append((hero1, hero2))

            for i, hero1 in enumerate(team2):
                for hero2 in team2[i + 1 :]:
                    self.pairs.append((hero1, hero2))

    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[int, int]:
        """Get hero pair at index."""
        return self.pairs[idx]


class Hero2Vec(nn.Module):
    """Hero2Vec embedding model using skip-gram architecture."""

    def __init__(
        self, vocab_size: int, embedding_dim: int = 32, window_size: int = 5
    ) -> None:
        """
        Initialize Hero2Vec model.

        Args:
            vocab_size: Number of unique heroes
            embedding_dim: Dimension of hero embeddings
            window_size: Context window size
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # Embedding layers
        self.hero_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Initialize weights
        self.hero_embeddings.weight.data.uniform_(-0.5 / embedding_dim, 0.5 / embedding_dim)
        self.context_embeddings.weight.data.uniform_(-0.5 / embedding_dim, 0.5 / embedding_dim)

    def forward(self, hero_ids: torch.Tensor, context_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            hero_ids: Hero ID tensor (batch_size,)
            context_ids: Context hero ID tensor (batch_size,)

        Returns:
            Similarity scores
        """
        hero_emb = self.hero_embeddings(hero_ids)
        context_emb = self.context_embeddings(context_ids)

        # Dot product similarity
        scores = torch.sum(hero_emb * context_emb, dim=1)
        return scores

    def get_embeddings(self) -> NDArray[np.float32]:
        """
        Get hero embedding matrix.

        Returns:
            Embedding matrix of shape (vocab_size, embedding_dim)
        """
        self.eval()
        with torch.no_grad():
            embeddings = self.hero_embeddings.weight.cpu().numpy()
        return embeddings.astype(np.float32)


def train_hero2vec(
    matches: List[Tuple[List[int], List[int]]],
    embedding_dim: int = 32,
    batch_size: int = 64,
    epochs: int = 50,
    learning_rate: float = 0.001,
    device: str = "cpu",
) -> Hero2Vec:
    """
    Train Hero2Vec model on match data.

    Args:
        matches: List of (team1, team2) tuples
        embedding_dim: Dimension of embeddings
        batch_size: Training batch size
        epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on ('cpu' or 'cuda')

    Returns:
        Trained Hero2Vec model
    """
    metadata = get_hero_metadata()
    vocab_size = len(metadata.heroes)

    # Create dataset
    dataset = HeroCooccurrenceDataset(matches)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = Hero2Vec(vocab_size, embedding_dim)
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for hero_ids, context_ids in dataloader:
            hero_ids = hero_ids.to(device)
            context_ids = context_ids.to(device)

            # Forward pass
            scores = model(hero_ids, context_ids)
            # Positive samples (co-occurring heroes)
            labels = torch.ones_like(scores)

            loss = criterion(scores, labels)
            total_loss += loss.item()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    return model


def load_hero_embeddings(embeddings_path: Path) -> NDArray[np.float32]:
    """
    Load pre-trained hero embeddings from file.

    Args:
        embeddings_path: Path to embeddings file (.npy)

    Returns:
        Embedding matrix
    """
    embeddings = np.load(embeddings_path)
    return embeddings.astype(np.float32)


def save_hero_embeddings(embeddings: NDArray[np.float32], embeddings_path: Path) -> None:
    """
    Save hero embeddings to file.

    Args:
        embeddings: Embedding matrix
        embeddings_path: Path to save embeddings
    """
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_path, embeddings)

