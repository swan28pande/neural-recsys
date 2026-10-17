"""Bayesian Personalized Ranking Matrix Factorization (BPR-MF).

Implements the BPR-MF baseline from:
    Rendle et al., "BPR: Bayesian Personalized Ranking from Implicit Feedback",
    UAI 2009. https://arxiv.org/abs/1205.2618

BPR optimizes a pairwise ranking loss: for each user, a positive (interacted)
item should be scored higher than a negative (unobserved) item. This is more
appropriate for implicit feedback than pointwise losses.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BPRMF(nn.Module):
    """Matrix Factorization with BPR loss.

    Scores user-item pairs as:
        score(u, i) = <p_u, q_i> + b_u + b_i + mu

    where p_u, q_i are latent vectors and b_u, b_i are bias terms.

    The BPR loss encourages:
        score(u, pos) > score(u, neg)  for observed pairs.

    Args:
        num_users: Total number of users.
        num_items: Total number of items.
        embed_dim: Dimensionality of latent factors.
        l2_reg: L2 regularization coefficient for embeddings.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embed_dim: int = 64,
        l2_reg: float = 0.001,
    ) -> None:
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.l2_reg = l2_reg

        # Latent factor embeddings
        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.item_embedding = nn.Embedding(num_items, embed_dim)

        # Bias terms
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize with small normal distribution, zero biases."""
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Compute predicted scores for user-item pairs.

        Args:
            user_ids: User indices, shape (batch_size,).
            item_ids: Item indices, shape (batch_size,).

        Returns:
            Predicted scores, shape (batch_size,).
        """
        user_emb = self.user_embedding(user_ids)   # (B, D)
        item_emb = self.item_embedding(item_ids)   # (B, D)

        # Dot product + biases
        dot = (user_emb * item_emb).sum(dim=-1)    # (B,)
        b_u = self.user_bias(user_ids).squeeze(-1)  # (B,)
        b_i = self.item_bias(item_ids).squeeze(-1)  # (B,)

        return dot + b_u + b_i + self.global_bias

    def bpr_loss(
        self,
        user_ids: torch.Tensor,
        pos_item_ids: torch.Tensor,
        neg_item_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute BPR pairwise ranking loss with L2 regularization.

        Loss = -mean(log(sigma(score_pos - score_neg))) + l2_reg * ||params||^2

        Args:
            user_ids: User indices, shape (batch_size,).
            pos_item_ids: Positive item indices, shape (batch_size,).
            neg_item_ids: Negative item indices, shape (batch_size,).

        Returns:
            Scalar loss value.
        """
        pos_scores = self.forward(user_ids, pos_item_ids)
        neg_scores = self.forward(user_ids, neg_item_ids)

        # BPR loss: -log(sigmoid(pos - neg))
        bpr_loss = -F.logsigmoid(pos_scores - neg_scores).mean()

        # L2 regularization on embeddings involved in this batch
        user_emb = self.user_embedding(user_ids)
        pos_emb = self.item_embedding(pos_item_ids)
        neg_emb = self.item_embedding(neg_item_ids)

        l2_loss = self.l2_reg * (
            user_emb.norm(2).pow(2)
            + pos_emb.norm(2).pow(2)
            + neg_emb.norm(2).pow(2)
            + self.user_bias(user_ids).norm(2).pow(2)
            + self.item_bias(pos_item_ids).norm(2).pow(2)
            + self.item_bias(neg_item_ids).norm(2).pow(2)
        ) / user_ids.size(0)

        return bpr_loss + l2_loss

    def get_user_embeddings(self, user_ids: torch.Tensor) -> torch.Tensor:
        """Get user latent vectors for retrieval."""
        return self.user_embedding(user_ids)

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        """Get item latent vectors for retrieval."""
        return self.item_embedding(item_ids)
