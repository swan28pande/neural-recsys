"""Training pipeline for recommendation models.

Supports training all model variants (GMF, MLP, NeuMF, BPR-MF, Two-Tower)
with:
    - Configurable optimizers and learning rate schedulers
    - Early stopping based on validation metrics
    - Model checkpointing (best and latest)
    - Structured logging with per-epoch metrics
    - Negative sample refresh per epoch

Usage:
    python -m src.train --config configs/default.yaml
    python -m src.train --config configs/default.yaml --model neumf
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import (
    EvaluationDataset,
    MovieLensDataset,
    PairwiseTrainDataset,
)
from src.data.preprocessing import MovieLensPreprocessor
from src.evaluate import Evaluator
from src.models.matrix_factorization import BPRMF
from src.models.ncf import GMF, MLP, NeuMF
from src.models.two_tower import TwoTowerModel


def setup_logging(log_dir: str, verbose: bool = True) -> logging.Logger:
    """Configure logging to both file and console.

    Args:
        log_dir: Directory for log files.
        verbose: If True, log to console at INFO level.

    Returns:
        Configured logger instance.
    """
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("recsys")
    logger.setLevel(logging.DEBUG)

    # File handler
    fh = logging.FileHandler(os.path.join(log_dir, "training.log"))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    )
    logger.addHandler(fh)

    # Console handler
    if verbose:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(ch)

    return logger


def get_device(config: dict) -> torch.device:
    """Determine the compute device from config."""
    device_str = config.get("system", {}).get("device", "auto")
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_str)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(
    config: dict,
    num_users: int,
    num_items: int,
    device: torch.device,
) -> nn.Module:
    """Instantiate the model specified in config.

    Args:
        config: Full configuration dictionary.
        num_users: Total number of users.
        num_items: Total number of items.
        device: Target device.

    Returns:
        Model instance moved to device.
    """
    model_name = config["model"]["name"]
    model_cfg = config["model"]

    if model_name == "gmf":
        model = GMF(
            num_users=num_users,
            num_items=num_items,
            embed_dim=model_cfg["gmf"]["embed_dim"],
        )

    elif model_name == "mlp":
        mlp_cfg = model_cfg["mlp"]
        model = MLP(
            num_users=num_users,
            num_items=num_items,
            embed_dim=mlp_cfg["embed_dim"],
            hidden_layers=mlp_cfg["hidden_layers"],
            dropout=mlp_cfg["dropout"],
            activation=mlp_cfg["activation"],
        )

    elif model_name == "neumf":
        neumf_cfg = model_cfg["neumf"]
        model = NeuMF(
            num_users=num_users,
            num_items=num_items,
            gmf_embed_dim=neumf_cfg["gmf_embed_dim"],
            mlp_embed_dim=neumf_cfg["mlp_embed_dim"],
            mlp_hidden_layers=neumf_cfg["mlp_hidden_layers"],
            dropout=neumf_cfg["dropout"],
        )

        # Load pretrained weights if specified
        if neumf_cfg.get("pretrain_gmf") and neumf_cfg.get("pretrain_mlp"):
            gmf_model = GMF(num_users, num_items, neumf_cfg["gmf_embed_dim"])
            gmf_model.load_state_dict(torch.load(neumf_cfg["pretrain_gmf"], weights_only=True))

            mlp_model = MLP(
                num_users, num_items, neumf_cfg["mlp_embed_dim"],
                neumf_cfg["mlp_hidden_layers"],
            )
            mlp_model.load_state_dict(torch.load(neumf_cfg["pretrain_mlp"], weights_only=True))

            model.load_pretrained(gmf_model, mlp_model, alpha=neumf_cfg["alpha"])

    elif model_name == "bpr_mf":
        bpr_cfg = model_cfg["bpr_mf"]
        model = BPRMF(
            num_users=num_users,
            num_items=num_items,
            embed_dim=bpr_cfg["embed_dim"],
            l2_reg=bpr_cfg["l2_reg"],
        )

    elif model_name == "two_tower":
        tt_cfg = model_cfg["two_tower"]
        model = TwoTowerModel(
            num_users=num_users,
            num_items=num_items,
            user_embed_dim=tt_cfg["user_embed_dim"],
            item_embed_dim=tt_cfg["item_embed_dim"],
            user_hidden_layers=tt_cfg["user_hidden_layers"],
            item_hidden_layers=tt_cfg["item_hidden_layers"],
            dropout=tt_cfg["dropout"],
            temperature=tt_cfg["temperature"],
            similarity=tt_cfg["similarity"],
            use_features=tt_cfg["use_features"],
            num_genres=tt_cfg["num_genres"],
            num_age_groups=tt_cfg["num_age_groups"],
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model.to(device)


def build_optimizer(
    model: nn.Module, config: dict
) -> tuple[optim.Optimizer, Optional[optim.lr_scheduler.LRScheduler]]:
    """Create optimizer and learning rate scheduler from config.

    Returns:
        Tuple of (optimizer, scheduler or None).
    """
    train_cfg = config["training"]
    lr = train_cfg["learning_rate"]
    wd = train_cfg["weight_decay"]

    if train_cfg["optimizer"] == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif train_cfg["optimizer"] == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif train_cfg["optimizer"] == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {train_cfg['optimizer']}")

    # Learning rate scheduler
    scheduler = None
    sched_type = train_cfg.get("lr_scheduler", "none")

    if sched_type == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            patience=train_cfg["lr_patience"],
            factor=train_cfg["lr_factor"],
            min_lr=train_cfg["lr_min"],
        )
    elif sched_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=train_cfg["epochs"], eta_min=train_cfg["lr_min"]
        )
    elif sched_type == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=train_cfg["lr_patience"], gamma=train_cfg["lr_factor"]
        )

    return optimizer, scheduler


class EarlyStopping:
    """Early stopping handler.

    Monitors a validation metric and stops training if no improvement
    is seen for `patience` consecutive epochs.

    Args:
        patience: Number of epochs to wait before stopping.
        mode: 'max' for metrics where higher is better, 'min' for lower.
        min_delta: Minimum change to qualify as an improvement.
    """

    def __init__(self, patience: int = 10, mode: str = "max", min_delta: float = 0.0):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score: Optional[float] = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        """Check if training should stop.

        Args:
            score: Current validation metric value.

        Returns:
            True if training should stop.
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


def train_epoch_pointwise(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    config: dict,
    logger: logging.Logger,
    epoch: int,
) -> float:
    """Train one epoch with pointwise (BCE) loss.

    Used for GMF, MLP, NeuMF models.

    Returns:
        Average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    log_every = config["logging"]["log_every"]
    grad_clip = config["training"].get("gradient_clip", 0)
    criterion = nn.BCEWithLogitsLoss()

    model_name = config["model"]["name"]
    use_features = (
        model_name == "two_tower"
        and config["model"]["two_tower"].get("use_features", False)
    )

    for batch_idx, batch in enumerate(train_loader):
        user_ids = batch["user_id"].to(device)
        item_ids = batch["item_id"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        if use_features:
            user_ages = batch.get("user_age", None)
            item_genres = batch.get("item_genre", None)
            if user_ages is not None:
                user_ages = user_ages.to(device)
            if item_genres is not None:
                item_genres = item_genres.to(device)
            logits = model(user_ids, item_ids, user_ages=user_ages, item_genres=item_genres)
        else:
            logits = model(user_ids, item_ids)

        loss = criterion(logits, labels)
        loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if (batch_idx + 1) % log_every == 0:
            logger.debug(
                f"Epoch {epoch} | Batch {batch_idx + 1}/{len(train_loader)} | "
                f"Loss: {loss.item():.4f}"
            )

    return total_loss / max(num_batches, 1)


def train_epoch_bpr(
    model: BPRMF,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    config: dict,
    logger: logging.Logger,
    epoch: int,
) -> float:
    """Train one epoch with BPR pairwise loss.

    Used for BPR-MF model.

    Returns:
        Average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    log_every = config["logging"]["log_every"]
    grad_clip = config["training"].get("gradient_clip", 0)

    for batch_idx, batch in enumerate(train_loader):
        user_ids = batch["user_id"].to(device)
        pos_items = batch["pos_item_id"].to(device)
        neg_items = batch["neg_item_id"].to(device)

        optimizer.zero_grad()
        loss = model.bpr_loss(user_ids, pos_items, neg_items)
        loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if (batch_idx + 1) % log_every == 0:
            logger.debug(
                f"Epoch {epoch} | Batch {batch_idx + 1}/{len(train_loader)} | "
                f"Loss: {loss.item():.4f}"
            )

    return total_loss / max(num_batches, 1)


def train_epoch_softmax(
    model: TwoTowerModel,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    config: dict,
    logger: logging.Logger,
    epoch: int,
) -> float:
    """Train one epoch with temperature-scaled softmax (in-batch negatives) loss.

    Used for Two-Tower model.

    Returns:
        Average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    log_every = config["logging"]["log_every"]
    grad_clip = config["training"].get("gradient_clip", 0)

    for batch_idx, batch in enumerate(train_loader):
        user_ids = batch["user_id"].to(device)
        item_ids = batch["item_id"].to(device)

        # Only use positive samples for softmax loss (labels == 1)
        labels = batch["label"]
        pos_mask = labels == 1.0
        if pos_mask.sum() < 2:
            continue  # Need at least 2 positives for in-batch negatives

        user_ids = user_ids[pos_mask]
        item_ids = item_ids[pos_mask]

        user_ages = None
        item_genres = None
        if "user_age" in batch:
            user_ages = batch["user_age"][pos_mask].to(device)
        if "item_genre" in batch:
            item_genres = batch["item_genre"][pos_mask].to(device)

        optimizer.zero_grad()
        loss = model.compute_softmax_loss(user_ids, item_ids, user_ages, item_genres)
        loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if (batch_idx + 1) % log_every == 0:
            logger.debug(
                f"Epoch {epoch} | Batch {batch_idx + 1}/{len(train_loader)} | "
                f"Loss: {loss.item():.4f}"
            )

    return total_loss / max(num_batches, 1)


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    path: str,
) -> None:
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )


def train(config: dict) -> dict[str, float]:
    """Full training pipeline.

    1. Load and preprocess data
    2. Create datasets and dataloaders
    3. Build model, optimizer, scheduler
    4. Train with early stopping
    5. Evaluate best model on test set

    Args:
        config: Full configuration dictionary.

    Returns:
        Test set metrics for the best model.
    """
    # Setup
    seed = config.get("system", {}).get("seed", 42)
    set_seed(seed)
    device = get_device(config)
    logger = setup_logging(
        config["logging"]["log_dir"],
        config["logging"]["verbose"],
    )

    logger.info(f"Device: {device}")
    logger.info(f"Model: {config['model']['name']}")

    # ── Data Loading ──────────────────────────────────────────────────
    data_cfg = config["data"]
    preprocessor = MovieLensPreprocessor(
        data_dir=data_cfg["data_dir"],
        min_interactions=data_cfg["min_interactions"],
        implicit_threshold=data_cfg["implicit_threshold"],
        seed=data_cfg["seed"],
    )

    interactions = preprocessor.load_and_preprocess()
    num_users = preprocessor.num_users
    num_items = preprocessor.num_items
    logger.info(f"Users: {num_users}, Items: {num_items}, Interactions: {len(interactions)}")

    user_ages = preprocessor.get_user_age_groups()
    item_genres = preprocessor.get_item_genres()

    # Split data
    if data_cfg["split_method"] == "leave_one_out":
        train_df, val_df, test_df = preprocessor.leave_one_out_split(interactions)
    else:
        train_df, val_df, test_df = preprocessor.temporal_split(
            interactions, data_cfg["val_ratio"]
        )

    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Build positive interaction set (for negative sampling)
    all_positives = set(
        zip(interactions["user_id"].values, interactions["item_id"].values)
    )
    train_positives = set(
        zip(train_df["user_id"].values, train_df["item_id"].values)
    )

    # ── Datasets & DataLoaders ────────────────────────────────────────
    model_name = config["model"]["name"]
    train_cfg = config["training"]
    num_workers = config.get("system", {}).get("num_workers", 4)
    pin_memory = config.get("system", {}).get("pin_memory", True) and device.type == "cuda"

    use_features = (
        model_name == "two_tower"
        and config["model"]["two_tower"].get("use_features", False)
    )

    if model_name == "bpr_mf":
        train_dataset = PairwiseTrainDataset(
            user_ids=train_df["user_id"].values,
            item_ids=train_df["item_id"].values,
            num_items=num_items,
            user_item_set=train_positives,
            seed=seed,
        )
    else:
        train_dataset = MovieLensDataset(
            user_ids=train_df["user_id"].values,
            item_ids=train_df["item_id"].values,
            num_items=num_items,
            user_item_set=train_positives,
            neg_ratio=data_cfg["neg_sample_ratio"],
            user_ages=user_ages if use_features else None,
            item_genres=item_genres if use_features else None,
            seed=seed,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    # Validation dataset
    val_dataset = EvaluationDataset(
        user_ids=val_df["user_id"].values,
        pos_item_ids=val_df["item_id"].values,
        num_items=num_items,
        user_item_set=all_positives,
        num_neg_samples=data_cfg["test_neg_samples"],
        user_ages=user_ages if use_features else None,
        item_genres=item_genres if use_features else None,
        seed=seed,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["evaluation"]["eval_batch_size"],
        shuffle=False,
        num_workers=num_workers,
    )

    # Test dataset
    test_dataset = EvaluationDataset(
        user_ids=test_df["user_id"].values,
        pos_item_ids=test_df["item_id"].values,
        num_items=num_items,
        user_item_set=all_positives,
        num_neg_samples=data_cfg["test_neg_samples"],
        user_ages=user_ages if use_features else None,
        item_genres=item_genres if use_features else None,
        seed=seed + 1,  # Different seed for test negatives
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["evaluation"]["eval_batch_size"],
        shuffle=False,
        num_workers=num_workers,
    )

    # ── Model, Optimizer, Scheduler ───────────────────────────────────
    model = build_model(config, num_users, num_items, device)
    optimizer, scheduler = build_optimizer(model, config)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {param_count:,}")

    evaluator = Evaluator(
        model, device,
        top_k_values=config["evaluation"]["top_k"],
        use_features=use_features,
    )

    # Early stopping
    es_cfg = config["training"]["early_stopping"]
    early_stopping = EarlyStopping(
        patience=es_cfg["patience"],
        mode=es_cfg["mode"],
    ) if es_cfg["enabled"] else None

    checkpoint_dir = config["logging"]["checkpoint_dir"]
    best_metric = -float("inf") if es_cfg["mode"] == "max" else float("inf")
    best_metrics: dict[str, float] = {}

    # ── Training Loop ─────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Starting training...")
    logger.info("=" * 60)

    for epoch in range(1, train_cfg["epochs"] + 1):
        epoch_start = time.time()

        # Refresh negative samples each epoch (for pointwise datasets)
        if hasattr(train_dataset, "refresh_negatives"):
            train_dataset.refresh_negatives()

        # Train
        if model_name == "bpr_mf":
            train_loss = train_epoch_bpr(
                model, train_loader, optimizer, device, config, logger, epoch
            )
        elif model_name == "two_tower":
            train_loss = train_epoch_softmax(
                model, train_loader, optimizer, device, config, logger, epoch
            )
        else:
            train_loss = train_epoch_pointwise(
                model, train_loader, optimizer, device, config, logger, epoch
            )

        epoch_time = time.time() - epoch_start

        # Evaluate
        eval_every = config["logging"]["eval_every"]
        if epoch % eval_every == 0:
            val_metrics = evaluator.evaluate(
                val_loader,
                full_ranking=config["evaluation"]["full_ranking"],
                num_items=num_items,
                item_genres=item_genres if use_features else None,
            )

            # Log metrics
            metric_str = " | ".join(f"{k}: {v:.4f}" for k, v in val_metrics.items())
            logger.info(
                f"Epoch {epoch:3d}/{train_cfg['epochs']} | "
                f"Loss: {train_loss:.4f} | {metric_str} | "
                f"Time: {epoch_time:.1f}s"
            )

            # Learning rate scheduling
            monitor_metric = val_metrics.get(es_cfg["metric"], 0.0)

            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(monitor_metric)
                else:
                    scheduler.step()

            # Checkpointing
            is_better = (
                (monitor_metric > best_metric) if es_cfg["mode"] == "max"
                else (monitor_metric < best_metric)
            )
            if is_better:
                best_metric = monitor_metric
                best_metrics = val_metrics.copy()
                if config["logging"]["save_best"]:
                    save_checkpoint(
                        model, optimizer, epoch, val_metrics,
                        os.path.join(checkpoint_dir, f"{model_name}_best.pt"),
                    )

            # Early stopping
            if early_stopping is not None:
                if early_stopping(monitor_metric):
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
        else:
            logger.info(
                f"Epoch {epoch:3d}/{train_cfg['epochs']} | "
                f"Loss: {train_loss:.4f} | Time: {epoch_time:.1f}s"
            )

    # ── Final Evaluation ──────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Training complete. Evaluating best model on test set...")

    # Load best checkpoint if it exists
    best_path = os.path.join(checkpoint_dir, f"{model_name}_best.pt")
    if os.path.exists(best_path):
        checkpoint = torch.load(best_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded best model from epoch {checkpoint['epoch']}")

    test_metrics = evaluator.evaluate(
        test_loader,
        full_ranking=config["evaluation"]["full_ranking"],
        num_items=num_items,
        item_genres=item_genres if use_features else None,
    )

    logger.info("Test Results:")
    for metric, value in test_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    logger.info("=" * 60)

    return test_metrics


def main() -> None:
    """Entry point: parse args, load config, and run training."""
    parser = argparse.ArgumentParser(description="Train recommendation model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=["gmf", "mlp", "neumf", "bpr_mf", "two_tower"],
        help="Override model selection from config",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Apply overrides
    if args.model:
        config["model"]["name"] = args.model
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.lr:
        config["training"]["learning_rate"] = args.lr
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size

    # Run training
    train(config)


if __name__ == "__main__":
    main()
