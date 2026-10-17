"""Data preprocessing for MovieLens datasets.

Handles:
    - Loading raw MovieLens-1M data files
    - User/item ID remapping to contiguous indices
    - Implicit feedback conversion (ratings -> binary interactions)
    - Feature engineering (genre multi-hot, age group encoding)
    - Train/validation/test splitting (leave-one-out or temporal)
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd


# MovieLens-1M genre list (fixed order for consistent multi-hot encoding)
ML1M_GENRES = [
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]

# MovieLens-1M age group mapping
ML1M_AGE_MAP = {1: 0, 18: 1, 25: 2, 35: 3, 45: 4, 50: 5, 56: 6}


class MovieLensPreprocessor:
    """Preprocessor for MovieLens-1M dataset.

    Loads raw data, remaps IDs, engineers features, and creates
    train/validation/test splits suitable for implicit recommendation.

    Args:
        data_dir: Path to the MovieLens-1M directory (containing .dat files).
        min_interactions: Minimum interactions per user to keep (k-core filter).
        implicit_threshold: Ratings >= this value become positive feedback.
        seed: Random seed for reproducibility.

    Attributes:
        num_users: Number of users after filtering.
        num_items: Number of items after filtering.
        user_map: Mapping from original user IDs to contiguous indices.
        item_map: Mapping from original item IDs to contiguous indices.
    """

    def __init__(
        self,
        data_dir: str,
        min_interactions: int = 5,
        implicit_threshold: int = 4,
        seed: int = 42,
    ) -> None:
        self.data_dir = data_dir
        self.min_interactions = min_interactions
        self.implicit_threshold = implicit_threshold
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.user_map: dict[int, int] = {}
        self.item_map: dict[int, int] = {}
        self.num_users: int = 0
        self.num_items: int = 0

        self._user_features: Optional[pd.DataFrame] = None
        self._item_features: Optional[pd.DataFrame] = None

    def load_and_preprocess(self) -> pd.DataFrame:
        """Load MovieLens-1M data and apply all preprocessing steps.

        Returns:
            DataFrame with columns: user_id, item_id, timestamp
            (IDs are remapped to contiguous 0-indexed integers).
        """
        # Load ratings
        ratings = self._load_ratings()

        # Convert to implicit feedback
        ratings = ratings[ratings["rating"] >= self.implicit_threshold].copy()
        ratings.drop(columns=["rating"], inplace=True)

        # Filter users with too few interactions
        user_counts = ratings["user_id"].value_counts()
        valid_users = user_counts[user_counts >= self.min_interactions].index
        ratings = ratings[ratings["user_id"].isin(valid_users)].copy()

        # Remap IDs to contiguous integers
        ratings = self._remap_ids(ratings)

        # Load auxiliary features
        self._load_user_features()
        self._load_item_features()

        # Sort by timestamp for temporal consistency
        ratings.sort_values(["user_id", "timestamp"], inplace=True)
        ratings.reset_index(drop=True, inplace=True)

        return ratings

    def _load_ratings(self) -> pd.DataFrame:
        """Load ratings from the .dat file."""
        ratings_path = os.path.join(self.data_dir, "ratings.dat")
        ratings = pd.read_csv(
            ratings_path,
            sep="::",
            names=["user_id", "item_id", "rating", "timestamp"],
            engine="python",
            encoding="latin-1",
        )
        return ratings

    def _remap_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remap user and item IDs to contiguous 0-indexed integers."""
        unique_users = sorted(df["user_id"].unique())
        unique_items = sorted(df["item_id"].unique())

        self.user_map = {old: new for new, old in enumerate(unique_users)}
        self.item_map = {old: new for new, old in enumerate(unique_items)}

        df["user_id"] = df["user_id"].map(self.user_map)
        df["item_id"] = df["item_id"].map(self.item_map)

        self.num_users = len(unique_users)
        self.num_items = len(unique_items)

        return df

    def _load_user_features(self) -> None:
        """Load and process user demographic features."""
        users_path = os.path.join(self.data_dir, "users.dat")
        if not os.path.exists(users_path):
            return

        users = pd.read_csv(
            users_path,
            sep="::",
            names=["user_id", "gender", "age", "occupation", "zip"],
            engine="python",
            encoding="latin-1",
        )

        # Map age to group index
        users["age_group"] = users["age"].map(ML1M_AGE_MAP).fillna(0).astype(int)

        # Only keep users that survived filtering
        users = users[users["user_id"].isin(self.user_map)]
        users["user_id"] = users["user_id"].map(self.user_map)

        self._user_features = users[["user_id", "age_group"]].set_index("user_id")

    def _load_item_features(self) -> None:
        """Load and process item features (genres)."""
        movies_path = os.path.join(self.data_dir, "movies.dat")
        if not os.path.exists(movies_path):
            return

        movies = pd.read_csv(
            movies_path,
            sep="::",
            names=["item_id", "title", "genres"],
            engine="python",
            encoding="latin-1",
        )

        # Multi-hot encode genres
        genre_matrix = self._encode_genres(movies["genres"])
        movies = pd.concat([movies[["item_id"]], genre_matrix], axis=1)

        # Only keep items that survived filtering
        movies = movies[movies["item_id"].isin(self.item_map)]
        movies["item_id"] = movies["item_id"].map(self.item_map)
        self._item_features = movies.set_index("item_id")

    def _encode_genres(self, genre_series: pd.Series) -> pd.DataFrame:
        """Convert pipe-separated genre strings to multi-hot encoding.

        Args:
            genre_series: Series of genre strings like "Action|Comedy|Drama".

        Returns:
            DataFrame with one column per genre, values 0 or 1.
        """
        genre_matrix = np.zeros((len(genre_series), len(ML1M_GENRES)), dtype=np.float32)

        genre_to_idx = {g: i for i, g in enumerate(ML1M_GENRES)}

        for i, genres_str in enumerate(genre_series):
            for genre in genres_str.split("|"):
                if genre in genre_to_idx:
                    genre_matrix[i, genre_to_idx[genre]] = 1.0

        return pd.DataFrame(genre_matrix, columns=ML1M_GENRES)

    def get_user_age_groups(self) -> Optional[np.ndarray]:
        """Get age group index for each user (indexed by remapped user ID).

        Returns:
            Array of shape (num_users,) with age group indices, or None.
        """
        if self._user_features is None:
            return None

        age_groups = np.zeros(self.num_users, dtype=np.int64)
        for uid, row in self._user_features.iterrows():
            age_groups[uid] = row["age_group"]
        return age_groups

    def get_item_genres(self) -> Optional[np.ndarray]:
        """Get multi-hot genre vectors for each item (indexed by remapped item ID).

        Returns:
            Array of shape (num_items, num_genres) with binary genre indicators, or None.
        """
        if self._item_features is None:
            return None

        genre_matrix = np.zeros((self.num_items, len(ML1M_GENRES)), dtype=np.float32)
        genre_cols = [c for c in self._item_features.columns if c in ML1M_GENRES]

        for iid, row in self._item_features.iterrows():
            genre_matrix[iid] = row[genre_cols].values

        return genre_matrix

    def leave_one_out_split(
        self, interactions: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Leave-one-out split based on temporal order.

        For each user, the last interaction becomes the test item, the
        second-to-last becomes the validation item, and the rest are training.

        This is the standard evaluation protocol from He et al. (2017).

        Args:
            interactions: DataFrame with user_id, item_id, timestamp.

        Returns:
            Tuple of (train_df, val_df, test_df).
        """
        interactions = interactions.sort_values(
            ["user_id", "timestamp"]
        ).reset_index(drop=True)

        test_rows = []
        val_rows = []
        train_rows = []

        for user_id, group in interactions.groupby("user_id"):
            items = group.values.tolist()
            if len(items) >= 3:
                test_rows.append(items[-1])
                val_rows.append(items[-2])
                train_rows.extend(items[:-2])
            elif len(items) == 2:
                test_rows.append(items[-1])
                train_rows.append(items[0])
            else:
                train_rows.extend(items)

        cols = interactions.columns.tolist()
        train_df = pd.DataFrame(train_rows, columns=cols)
        val_df = pd.DataFrame(val_rows, columns=cols)
        test_df = pd.DataFrame(test_rows, columns=cols)

        return train_df, val_df, test_df

    def temporal_split(
        self, interactions: pd.DataFrame, val_ratio: float = 0.1
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data by timestamp into train/val/test.

        Uses global timestamp cutoffs: the most recent val_ratio fraction
        of interactions become test, the next val_ratio become validation.

        Args:
            interactions: DataFrame with user_id, item_id, timestamp.
            val_ratio: Fraction of data for validation and test each.

        Returns:
            Tuple of (train_df, val_df, test_df).
        """
        interactions = interactions.sort_values("timestamp").reset_index(drop=True)
        n = len(interactions)

        test_cutoff = int(n * (1 - val_ratio))
        val_cutoff = int(n * (1 - 2 * val_ratio))

        train_df = interactions.iloc[:val_cutoff].copy()
        val_df = interactions.iloc[val_cutoff:test_cutoff].copy()
        test_df = interactions.iloc[test_cutoff:].copy()

        return train_df, val_df, test_df
