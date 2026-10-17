"""Two-Tower Retrieval Model.

A dual-encoder architecture where user and item representations are computed
independently by separate neural network towers. At serving time, item embeddings
can be precomputed and indexed for fast approximate nearest neighbor retrieval.

This is the architecture used in production at scale (YouTube, Meta, etc.) for
candidate generation in multi-stage recommendation pipelines.

References:
    - Yi et al., "Sampling-Bias-Corrected Neural Modeling for Large Corpus
      Item Recommendations", RecSys 2019.
    - Huang et al., "Embedding-based Retrieval in Facebook Search", KDD 2020.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class Tower(nn.Module):
    """A single tower (user or item encoder).

    Processes an embedding (or concatenation of embeddings + features) through
    a stack of fully-connected layers, producing a fixed-size representation.

    Args:
        input_dim: Input feature dimension.
        hidden_layers: List of hidden layer sizes.
        output_dim: Final representation dimension.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: list[int],
        output_dim: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        curr_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(curr_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            curr_dim = hidden_dim

        layers.append(nn.Linear(curr_dim, output_dim))
        self.network = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features, shape (batch_size, input_dim).

        Returns:
            Encoded representation, shape (batch_size, output_dim).
        """
        return self.network(x)


class TwoTowerModel(nn.Module):
    """Two-Tower (Dual Encoder) Retrieval Model.

    Architecture:
        User features ──► User Tower ──► user_repr ──┐
                                                      ├─► similarity score
        Item features ──► Item Tower ──► item_repr ──┘

    Supports:
        - Dot product or cosine similarity scoring
        - Temperature-scaled softmax (in-batch negatives) loss
        - Auxiliary features: user age groups, item genre embeddings

    Args:
        num_users: Total number of users.
        num_items: Total number of items.
        user_embed_dim: User ID embedding dimension.
        item_embed_dim: Item ID embedding dimension.
        user_hidden_layers: Hidden layer sizes for user tower.
        item_hidden_layers: Hidden layer sizes for item tower.
        dropout: Dropout probability.
        temperature: Temperature for softmax loss scaling.
        similarity: Similarity function ('dot' or 'cosine').
        use_features: Whether to include auxiliary features.
        num_genres: Number of item genres (for multi-hot encoding).
        num_age_groups: Number of user age groups.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        user_embed_dim: int = 64,
        item_embed_dim: int = 64,
        user_hidden_layers: list[int] | None = None,
        item_hidden_layers: list[int] | None = None,
        dropout: float = 0.2,
        temperature: float = 0.07,
        similarity: str = "dot",
        use_features: bool = True,
        num_genres: int = 18,
        num_age_groups: int = 7,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.similarity = similarity
        self.use_features = use_features
        self.num_users = num_users
        self.num_items = num_items

        if user_hidden_layers is None:
            user_hidden_layers = [128, 64]
        if item_hidden_layers is None:
            item_hidden_layers = [128, 64]

        # Shared output dimension (must match for similarity computation)
        output_dim = 64

        # User embeddings and features
        self.user_id_embedding = nn.Embedding(num_users, user_embed_dim)
        user_input_dim = user_embed_dim

        if use_features:
            self.user_age_embedding = nn.Embedding(num_age_groups, 16)
            user_input_dim += 16

        # Item embeddings and features
        self.item_id_embedding = nn.Embedding(num_items, item_embed_dim)
        item_input_dim = item_embed_dim

        if use_features:
            self.genre_projection = nn.Linear(num_genres, 32)
            item_input_dim += 32

        # Build towers
        self.user_tower = Tower(user_input_dim, user_hidden_layers, output_dim, dropout)
        self.item_tower = Tower(item_input_dim, item_hidden_layers, output_dim, dropout)

        self._init_embeddings()

    def _init_embeddings(self) -> None:
        nn.init.normal_(self.user_id_embedding.weight, std=0.01)
        nn.init.normal_(self.item_id_embedding.weight, std=0.01)
        if self.use_features:
            nn.init.normal_(self.user_age_embedding.weight, std=0.01)
            nn.init.xavier_uniform_(self.genre_projection.weight)

    def encode_user(
        self,
        user_ids: torch.Tensor,
        user_ages: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode users into latent representations.

        Args:
            user_ids: User indices, shape (batch_size,).
            user_ages: User age group indices, shape (batch_size,).
                Required when use_features=True.

        Returns:
            User representations, shape (batch_size, output_dim).
        """
        user_emb = self.user_id_embedding(user_ids)  # (B, D_user)

        if self.use_features:
            if user_ages is not None:
                age_emb = self.user_age_embedding(user_ages)  # (B, 16)
            else:
                # Default to age group 0 if not provided
                age_emb = self.user_age_embedding(
                    torch.zeros(user_ids.size(0), dtype=torch.long, device=user_ids.device)
                )
            user_emb = torch.cat([user_emb, age_emb], dim=-1)

        user_repr = self.user_tower(user_emb)  # (B, output_dim)

        if self.similarity == "cosine":
            user_repr = F.normalize(user_repr, p=2, dim=-1)

        return user_repr

    def encode_item(
        self,
        item_ids: torch.Tensor,
        item_genres: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode items into latent representations.

        Args:
            item_ids: Item indices, shape (batch_size,).
            item_genres: Multi-hot genre vectors, shape (batch_size, num_genres).
                Required when use_features=True.

        Returns:
            Item representations, shape (batch_size, output_dim).
        """
        item_emb = self.item_id_embedding(item_ids)  # (B, D_item)

        if self.use_features:
            if item_genres is not None:
                genre_emb = self.genre_projection(item_genres.float())  # (B, 32)
            else:
                # Default to zero genres if not provided
                num_genres = self.genre_projection.in_features
                genre_emb = self.genre_projection(
                    torch.zeros(item_ids.size(0), num_genres, dtype=torch.float, device=item_ids.device)
                )
            item_emb = torch.cat([item_emb, genre_emb], dim=-1)

        item_repr = self.item_tower(item_emb)  # (B, output_dim)

        if self.similarity == "cosine":
            item_repr = F.normalize(item_repr, p=2, dim=-1)

        return item_repr

    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        user_ages: Optional[torch.Tensor] = None,
        item_genres: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute similarity scores for user-item pairs.

        Args:
            user_ids: User indices, shape (batch_size,).
            item_ids: Item indices, shape (batch_size,).
            user_ages: Optional user age group indices.
            item_genres: Optional item genre multi-hot vectors.

        Returns:
            Similarity scores, shape (batch_size,).
        """
        user_repr = self.encode_user(user_ids, user_ages)
        item_repr = self.encode_item(item_ids, item_genres)

        # Dot product similarity (cosine if representations are L2-normalized)
        scores = (user_repr * item_repr).sum(dim=-1)  # (B,)
        return scores

    def compute_softmax_loss(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        user_ages: Optional[torch.Tensor] = None,
        item_genres: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Temperature-scaled softmax loss with in-batch negatives.

        Each positive pair in the batch uses all other items in the batch
        as negatives, creating an efficient O(B^2) contrastive loss.

        Args:
            user_ids: User indices, shape (batch_size,).
            item_ids: Positive item indices, shape (batch_size,).
            user_ages: Optional user age group indices.
            item_genres: Optional item genre multi-hot vectors.

        Returns:
            Scalar loss value.
        """
        user_repr = self.encode_user(user_ids, user_ages)   # (B, D)
        item_repr = self.encode_item(item_ids, item_genres)  # (B, D)

        # All-pairs similarity matrix: (B, B)
        logits = torch.mm(user_repr, item_repr.t()) / self.temperature

        # Labels: each user's positive item is at the diagonal
        labels = torch.arange(logits.size(0), device=logits.device)

        # Cross-entropy: row i should have max at column i
        loss = F.cross_entropy(logits, labels)
        return loss
