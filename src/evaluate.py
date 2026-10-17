"""Evaluation metrics for top-K recommendation.

Implements standard ranking metrics for the leave-one-out evaluation protocol:
    - Hit Rate @ K (HR@K): Whether the ground-truth item appears in the top-K
    - Normalized Discounted Cumulative Gain @ K (NDCG@K): Position-aware metric
    - Mean Reciprocal Rank (MRR): Average of 1/rank of the first relevant item
    - Area Under the ROC Curve (AUC): Probability that a positive is ranked higher

All metrics follow the evaluation protocol from He et al. (2017):
    For each user, rank the ground-truth item among N sampled negatives
    (or all items for full ranking) and compute metrics on the ranked list.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def hit_rate_at_k(rankings: np.ndarray, k: int) -> float:
    """Compute Hit Rate @ K.

    HR@K = 1 if the ground-truth item is in the top-K, else 0.
    Averaged over all users.

    Args:
        rankings: Rank of the ground-truth item for each user (1-indexed).
        k: Cutoff position.

    Returns:
        Average hit rate.
    """
    return float(np.mean(rankings <= k))


def ndcg_at_k(rankings: np.ndarray, k: int) -> float:
    """Compute Normalized Discounted Cumulative Gain @ K.

    NDCG@K = 1/log2(rank+1) if rank <= K, else 0.
    Since there is exactly one relevant item per user (leave-one-out),
    the ideal DCG is always 1/log2(2) = 1.

    Args:
        rankings: Rank of the ground-truth item for each user (1-indexed).
        k: Cutoff position.

    Returns:
        Average NDCG.
    """
    dcg = np.where(rankings <= k, 1.0 / np.log2(rankings + 1), 0.0)
    return float(np.mean(dcg))


def mean_reciprocal_rank(rankings: np.ndarray) -> float:
    """Compute Mean Reciprocal Rank.

    MRR = mean(1 / rank) over all users.

    Args:
        rankings: Rank of the ground-truth item for each user (1-indexed).

    Returns:
        MRR value.
    """
    return float(np.mean(1.0 / rankings))


def auc_score(rankings: np.ndarray, num_candidates: int) -> float:
    """Compute AUC (Area Under ROC Curve) from rankings.

    For a single positive item among (num_candidates - 1) negatives,
    AUC = (num_candidates - rank) / (num_candidates - 1).

    This counts the fraction of negatives ranked below the positive.

    Args:
        rankings: Rank of the ground-truth item for each user (1-indexed).
        num_candidates: Total number of candidates (1 positive + N negatives).

    Returns:
        Average AUC.
    """
    auc = (num_candidates - rankings) / (num_candidates - 1)
    return float(np.mean(auc))


class Evaluator:
    """Full evaluation pipeline for recommendation models.

    Scores all candidate items for each user, ranks them, and computes
    multiple ranking metrics.

    Args:
        model: The recommendation model (must accept user_ids, item_ids).
        device: Device to run evaluation on.
        top_k_values: List of K values for HR@K and NDCG@K.
        use_features: Whether the model uses auxiliary features.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        top_k_values: list[int] | None = None,
        use_features: bool = False,
    ) -> None:
        self.model = model
        self.device = device
        self.top_k_values = top_k_values or [5, 10, 20]
        self.use_features = use_features

    @torch.no_grad()
    def evaluate(
        self,
        eval_loader: DataLoader,
        full_ranking: bool = False,
        num_items: Optional[int] = None,
        item_genres: Optional[np.ndarray] = None,
    ) -> dict[str, float]:
        """Run evaluation and compute all metrics.

        Args:
            eval_loader: DataLoader yielding evaluation batches.
            full_ranking: If True, rank against ALL items (slow but accurate).
            num_items: Required if full_ranking=True.
            item_genres: Item genre matrix, required if model uses features.

        Returns:
            Dictionary of metric_name -> value.
        """
        self.model.eval()
        all_rankings = []
        all_num_candidates = []

        for batch in tqdm(eval_loader, desc="Evaluating", leave=False):
            if full_ranking:
                rankings, n_cand = self._rank_full(batch, num_items, item_genres)
            else:
                rankings, n_cand = self._rank_sampled(batch, item_genres)

            all_rankings.append(rankings)
            all_num_candidates.append(n_cand)

        rankings = np.concatenate(all_rankings)
        # For AUC, all candidates count should be consistent in sampled mode
        num_candidates = all_num_candidates[0]

        metrics: dict[str, float] = {}
        for k in self.top_k_values:
            metrics[f"hr@{k}"] = hit_rate_at_k(rankings, k)
            metrics[f"ndcg@{k}"] = ndcg_at_k(rankings, k)

        metrics["mrr"] = mean_reciprocal_rank(rankings)
        metrics["auc"] = auc_score(rankings, num_candidates)

        return metrics

    def _rank_sampled(
        self,
        batch: dict[str, torch.Tensor],
        item_genres: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, int]:
        """Rank ground-truth item among sampled negatives.

        The first item in batch["items"] is the positive; the rest are negatives.

        Returns:
            Tuple of (rankings array, number of candidates).
        """
        user_ids = batch["user_id"].to(self.device)          # (B,)
        items = batch["items"].to(self.device)                # (B, 1+N)
        num_candidates = items.size(1)

        batch_size = user_ids.size(0)

        # Expand user_ids to match items: (B,) -> (B * num_candidates,)
        users_expanded = user_ids.unsqueeze(1).expand(-1, num_candidates).reshape(-1)
        items_flat = items.reshape(-1)

        # Score all user-item pairs
        if self.use_features and "user_age" in batch and item_genres is not None:
            user_ages = batch["user_age"].to(self.device)
            ages_expanded = user_ages.unsqueeze(1).expand(-1, num_candidates).reshape(-1)

            if "item_genres" in batch:
                genres_flat = batch["item_genres"].to(self.device).reshape(
                    -1, batch["item_genres"].size(-1)
                )
            else:
                genres_np = item_genres[items_flat.cpu().numpy()]
                genres_flat = torch.tensor(genres_np, dtype=torch.float, device=self.device)

            scores = self.model(
                users_expanded, items_flat,
                user_ages=ages_expanded, item_genres=genres_flat,
            )
        else:
            scores = self.model(users_expanded, items_flat)

        scores = scores.view(batch_size, num_candidates)  # (B, 1+N)

        # Rank: position of positive item (index 0) in descending score order
        # Higher score = better, so count items scored higher than the positive
        pos_scores = scores[:, 0].unsqueeze(1)  # (B, 1)
        rankings = (scores >= pos_scores).sum(dim=1).cpu().numpy()  # (B,)

        return rankings, num_candidates

    def _rank_full(
        self,
        batch: dict[str, torch.Tensor],
        num_items: int,
        item_genres: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, int]:
        """Rank ground-truth item against ALL items.

        More accurate but slower. Scores each user against every item.

        Returns:
            Tuple of (rankings array, total number of items).
        """
        user_ids = batch["user_id"].to(self.device)  # (B,)
        pos_items = batch["items"][:, 0].to(self.device)  # (B,) ground truth

        batch_size = user_ids.size(0)
        all_items = torch.arange(num_items, device=self.device)

        rankings = np.zeros(batch_size, dtype=np.int64)

        # Score each user against all items (one user at a time for memory)
        for i in range(batch_size):
            user = user_ids[i].expand(num_items)  # (num_items,)

            if self.use_features and "user_age" in batch and item_genres is not None:
                user_age = batch["user_age"][i].to(self.device).expand(num_items)
                genres = torch.tensor(
                    item_genres, dtype=torch.float, device=self.device
                )
                scores = self.model(user, all_items, user_ages=user_age, item_genres=genres)
            else:
                scores = self.model(user, all_items)

            # Rank of the positive item
            pos_score = scores[pos_items[i]]
            rank = (scores >= pos_score).sum().item()
            rankings[i] = rank

        return rankings, num_items
