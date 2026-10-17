"""PyTorch Dataset classes for recommendation training and evaluation.

Provides:
    - MovieLensDataset: Pointwise training with negative sampling (for NCF)
    - PairwiseTrainDataset: Pairwise training with BPR-style triplets (for BPR-MF)
    - EvaluationDataset: Leave-one-out evaluation with sampled negatives
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class MovieLensDataset(Dataset):
    """Pointwise implicit feedback dataset with online negative sampling.

    For each positive (user, item) interaction, samples `neg_ratio` negative
    items (items the user has not interacted with). Returns (user, item, label)
    tuples where label=1 for positive and label=0 for negative samples.

    Used for training NCF and Two-Tower models with binary cross-entropy loss.

    Args:
        user_ids: Array of user indices for positive interactions.
        item_ids: Array of item indices for positive interactions.
        num_items: Total number of items in the dataset.
        user_item_set: Set of (user_id, item_id) tuples for all positive interactions.
        neg_ratio: Number of negative samples per positive sample.
        user_ages: Optional array of user age group indices.
        item_genres: Optional array of item genre multi-hot vectors, shape (num_items, num_genres).
        seed: Random seed for negative sampling.
    """

    def __init__(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        num_items: int,
        user_item_set: set[tuple[int, int]],
        neg_ratio: int = 4,
        user_ages: Optional[np.ndarray] = None,
        item_genres: Optional[np.ndarray] = None,
        seed: int = 42,
    ) -> None:
        self.user_ids = user_ids.astype(np.int64)
        self.item_ids = item_ids.astype(np.int64)
        self.num_items = num_items
        self.user_item_set = user_item_set
        self.neg_ratio = neg_ratio
        self.user_ages = user_ages
        self.item_genres = item_genres
        self.rng = np.random.default_rng(seed)

        # Pre-generate negative samples (refreshed each epoch)
        self._generate_negatives()

    def _generate_negatives(self) -> None:
        """Sample negative items for each positive interaction.

        Uses rejection sampling: randomly pick items until finding one
        the user has not interacted with.
        """
        num_pos = len(self.user_ids)
        total = num_pos * (1 + self.neg_ratio)

        all_users = np.zeros(total, dtype=np.int64)
        all_items = np.zeros(total, dtype=np.int64)
        all_labels = np.zeros(total, dtype=np.float32)

        idx = 0
        for i in range(num_pos):
            u = self.user_ids[i]
            pos_item = self.item_ids[i]

            # Positive sample
            all_users[idx] = u
            all_items[idx] = pos_item
            all_labels[idx] = 1.0
            idx += 1

            # Negative samples
            for _ in range(self.neg_ratio):
                neg_item = self.rng.integers(0, self.num_items)
                while (u, neg_item) in self.user_item_set:
                    neg_item = self.rng.integers(0, self.num_items)
                all_users[idx] = u
                all_items[idx] = neg_item
                all_labels[idx] = 0.0
                idx += 1

        self._users = all_users
        self._items = all_items
        self._labels = all_labels

    def refresh_negatives(self) -> None:
        """Re-sample negatives. Call at the start of each epoch."""
        self._generate_negatives()

    def __len__(self) -> int:
        return len(self._users)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return a single training sample.

        Returns:
            Dictionary with keys: user_id, item_id, label, and optionally
            user_age, item_genre.
        """
        sample = {
            "user_id": torch.tensor(self._users[idx], dtype=torch.long),
            "item_id": torch.tensor(self._items[idx], dtype=torch.long),
            "label": torch.tensor(self._labels[idx], dtype=torch.float),
        }

        if self.user_ages is not None:
            uid = self._users[idx]
            sample["user_age"] = torch.tensor(self.user_ages[uid], dtype=torch.long)

        if self.item_genres is not None:
            iid = self._items[idx]
            sample["item_genre"] = torch.tensor(
                self.item_genres[iid], dtype=torch.float
            )

        return sample


class PairwiseTrainDataset(Dataset):
    """Pairwise training dataset for BPR-style models.

    Returns (user, positive_item, negative_item) triplets. The negative
    item is sampled uniformly from items the user has not interacted with.

    Args:
        user_ids: Array of user indices for positive interactions.
        item_ids: Array of positive item indices.
        num_items: Total number of items.
        user_item_set: Set of (user_id, item_id) tuples for all positives.
        seed: Random seed.
    """

    def __init__(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        num_items: int,
        user_item_set: set[tuple[int, int]],
        seed: int = 42,
    ) -> None:
        self.user_ids = user_ids.astype(np.int64)
        self.item_ids = item_ids.astype(np.int64)
        self.num_items = num_items
        self.user_item_set = user_item_set
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.user_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return a BPR triplet: (user, positive_item, negative_item)."""
        user = self.user_ids[idx]
        pos_item = self.item_ids[idx]

        # Sample a negative item
        neg_item = self.rng.integers(0, self.num_items)
        while (user, neg_item) in self.user_item_set:
            neg_item = self.rng.integers(0, self.num_items)

        return {
            "user_id": torch.tensor(user, dtype=torch.long),
            "pos_item_id": torch.tensor(pos_item, dtype=torch.long),
            "neg_item_id": torch.tensor(neg_item, dtype=torch.long),
        }


class EvaluationDataset(Dataset):
    """Evaluation dataset for leave-one-out protocol.

    For each user, pairs one ground-truth positive item with `num_neg_samples`
    randomly sampled negative items. The model must rank the positive item
    above the negatives.

    Args:
        user_ids: Array of user indices (one per eval instance).
        pos_item_ids: Array of ground-truth positive item indices.
        num_items: Total number of items.
        user_item_set: Set of ALL (user, item) interactions (train + val + test)
            to exclude from negative sampling.
        num_neg_samples: Number of negative items to sample per user.
        user_ages: Optional user age group array.
        item_genres: Optional item genre multi-hot array.
        seed: Random seed.
    """

    def __init__(
        self,
        user_ids: np.ndarray,
        pos_item_ids: np.ndarray,
        num_items: int,
        user_item_set: set[tuple[int, int]],
        num_neg_samples: int = 99,
        user_ages: Optional[np.ndarray] = None,
        item_genres: Optional[np.ndarray] = None,
        seed: int = 42,
    ) -> None:
        self.user_ids = user_ids.astype(np.int64)
        self.pos_item_ids = pos_item_ids.astype(np.int64)
        self.num_items = num_items
        self.user_item_set = user_item_set
        self.num_neg_samples = num_neg_samples
        self.user_ages = user_ages
        self.item_genres = item_genres
        self.rng = np.random.default_rng(seed)

        # Pre-sample negatives for deterministic evaluation
        self.neg_items = self._sample_negatives()

    def _sample_negatives(self) -> np.ndarray:
        """Pre-sample negative items for all evaluation instances.

        Returns:
            Array of shape (num_users, num_neg_samples).
        """
        neg_items = np.zeros(
            (len(self.user_ids), self.num_neg_samples), dtype=np.int64
        )

        for i in range(len(self.user_ids)):
            u = self.user_ids[i]
            sampled = []
            while len(sampled) < self.num_neg_samples:
                j = self.rng.integers(0, self.num_items)
                if (u, j) not in self.user_item_set and j not in sampled:
                    sampled.append(j)
            neg_items[i] = sampled

        return neg_items

    def __len__(self) -> int:
        return len(self.user_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return evaluation data for one user.

        Returns:
            Dictionary with:
                user_id: scalar
                items: tensor of shape (1 + num_neg_samples,) with positive first
                and optionally user_age, item_genres.
        """
        user = self.user_ids[idx]
        pos_item = self.pos_item_ids[idx]
        neg_items = self.neg_items[idx]

        # Positive item first, then negatives
        items = np.concatenate([[pos_item], neg_items])

        sample = {
            "user_id": torch.tensor(user, dtype=torch.long),
            "items": torch.tensor(items, dtype=torch.long),
        }

        if self.user_ages is not None:
            sample["user_age"] = torch.tensor(self.user_ages[user], dtype=torch.long)

        if self.item_genres is not None:
            sample["item_genres"] = torch.tensor(
                self.item_genres[items], dtype=torch.float
            )

        return sample
