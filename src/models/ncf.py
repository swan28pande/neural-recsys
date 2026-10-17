"""Neural Collaborative Filtering (NCF) models.

Implements the NeuMF architecture from:
    He et al., "Neural Collaborative Filtering", WWW 2017.
    https://arxiv.org/abs/1708.05031

Three model variants:
    - GMF: Generalized Matrix Factorization (element-wise product of embeddings)
    - MLP: Multi-Layer Perceptron (concatenation + deep layers)
    - NeuMF: Neural Matrix Factorization (fuses GMF + MLP paths)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional


class GMF(nn.Module):
    """Generalized Matrix Factorization.

    Models user-item interaction as the element-wise product of user and item
    latent vectors, followed by a linear projection to a scalar score.

    This generalizes standard matrix factorization by allowing the output layer
    to learn non-uniform weights over latent dimensions.

    Args:
        num_users: Total number of users.
        num_items: Total number of items.
        embed_dim: Dimensionality of user/item embeddings.
    """

    def __init__(self, num_users: int, num_items: int, embed_dim: int = 64) -> None:
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim

        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.item_embedding = nn.Embedding(num_items, embed_dim)
        self.output_layer = nn.Linear(embed_dim, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize embeddings with normal distribution, output with Xavier."""
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            user_ids: User indices, shape (batch_size,).
            item_ids: Item indices, shape (batch_size,).

        Returns:
            Predicted scores, shape (batch_size,).
        """
        user_emb = self.user_embedding(user_ids)  # (B, D)
        item_emb = self.item_embedding(item_ids)  # (B, D)
        element_product = user_emb * item_emb     # (B, D)
        logits = self.output_layer(element_product).squeeze(-1)  # (B,)
        return logits

    def get_gmf_vector(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Return the GMF interaction vector (before output projection).

        Used by NeuMF for fusion.
        """
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        return user_emb * item_emb


class MLP(nn.Module):
    """Multi-Layer Perceptron path for NCF.

    Concatenates user and item embeddings and passes through a tower of
    fully-connected layers with ReLU activation and optional dropout.

    Args:
        num_users: Total number of users.
        num_items: Total number of items.
        embed_dim: Dimensionality of user/item embeddings.
        hidden_layers: List of hidden layer sizes (e.g. [256, 128, 64]).
        dropout: Dropout probability applied after each hidden layer.
        activation: Activation function name ('relu', 'leaky_relu', 'gelu').
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embed_dim: int = 64,
        hidden_layers: list[int] | None = None,
        dropout: float = 0.2,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim

        if hidden_layers is None:
            hidden_layers = [256, 128, 64]

        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.item_embedding = nn.Embedding(num_items, embed_dim)

        # Build MLP tower
        activation_fn = {
            "relu": nn.ReLU,
            "leaky_relu": nn.LeakyReLU,
            "gelu": nn.GELU,
        }[activation]

        layers: list[nn.Module] = []
        input_dim = embed_dim * 2  # Concatenation of user + item embeddings

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(activation_fn())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        self.mlp_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_layers[-1], 1)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize embeddings with normal distribution, layers with He init."""
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

        for module in self.mlp_layers:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            user_ids: User indices, shape (batch_size,).
            item_ids: Item indices, shape (batch_size,).

        Returns:
            Predicted scores, shape (batch_size,).
        """
        user_emb = self.user_embedding(user_ids)  # (B, D)
        item_emb = self.item_embedding(item_ids)  # (B, D)
        concat = torch.cat([user_emb, item_emb], dim=-1)  # (B, 2D)
        mlp_out = self.mlp_layers(concat)  # (B, hidden[-1])
        logits = self.output_layer(mlp_out).squeeze(-1)  # (B,)
        return logits

    def get_mlp_vector(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Return the MLP hidden vector (before output projection).

        Used by NeuMF for fusion.
        """
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        concat = torch.cat([user_emb, item_emb], dim=-1)
        return self.mlp_layers(concat)


class NeuMF(nn.Module):
    """Neural Matrix Factorization (NeuMF).

    Fuses the GMF and MLP paths by concatenating their penultimate-layer
    outputs and projecting to a final prediction. Supports pretraining:
    load separately trained GMF and MLP weights for better initialization.

    Architecture:
        User ─┬─► GMF_embed ──► element-wise product ──┐
              │                                         ├─► concat ─► output
              └─► MLP_embed ──► MLP tower ──────────────┘
        Item ─┬─► GMF_embed ──► element-wise product ──┘
              └─► MLP_embed ──► MLP tower ──────────────┘

    Args:
        num_users: Total number of users.
        num_items: Total number of items.
        gmf_embed_dim: Embedding dimension for GMF path.
        mlp_embed_dim: Embedding dimension for MLP path.
        mlp_hidden_layers: Hidden layer sizes for MLP path.
        dropout: Dropout probability in MLP layers.
        activation: Activation function name.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        gmf_embed_dim: int = 64,
        mlp_embed_dim: int = 64,
        mlp_hidden_layers: list[int] | None = None,
        dropout: float = 0.2,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items

        if mlp_hidden_layers is None:
            mlp_hidden_layers = [256, 128, 64]

        # GMF path
        self.gmf_user_embedding = nn.Embedding(num_users, gmf_embed_dim)
        self.gmf_item_embedding = nn.Embedding(num_items, gmf_embed_dim)

        # MLP path
        self.mlp_user_embedding = nn.Embedding(num_users, mlp_embed_dim)
        self.mlp_item_embedding = nn.Embedding(num_items, mlp_embed_dim)

        activation_fn = {
            "relu": nn.ReLU,
            "leaky_relu": nn.LeakyReLU,
            "gelu": nn.GELU,
        }[activation]

        layers: list[nn.Module] = []
        input_dim = mlp_embed_dim * 2

        for hidden_dim in mlp_hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(activation_fn())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        self.mlp_layers = nn.Sequential(*layers)

        # Fusion layer: concatenates GMF vector + MLP vector
        fusion_dim = gmf_embed_dim + mlp_hidden_layers[-1]
        self.output_layer = nn.Linear(fusion_dim, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize all parameters."""
        for emb in [
            self.gmf_user_embedding,
            self.gmf_item_embedding,
            self.mlp_user_embedding,
            self.mlp_item_embedding,
        ]:
            nn.init.normal_(emb.weight, std=0.01)

        for module in self.mlp_layers:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def load_pretrained(
        self,
        gmf_model: Optional[GMF] = None,
        mlp_model: Optional[MLP] = None,
        alpha: float = 0.5,
    ) -> None:
        """Load pretrained GMF and/or MLP weights into NeuMF.

        The output layer is initialized as a weighted combination:
            h = alpha * h_GMF  ||  (1-alpha) * h_MLP

        Args:
            gmf_model: Pretrained GMF model.
            mlp_model: Pretrained MLP model.
            alpha: Weight for GMF output layer in the fusion initialization.
        """
        if gmf_model is not None:
            self.gmf_user_embedding.weight.data.copy_(
                gmf_model.user_embedding.weight.data
            )
            self.gmf_item_embedding.weight.data.copy_(
                gmf_model.item_embedding.weight.data
            )

        if mlp_model is not None:
            self.mlp_user_embedding.weight.data.copy_(
                mlp_model.user_embedding.weight.data
            )
            self.mlp_item_embedding.weight.data.copy_(
                mlp_model.item_embedding.weight.data
            )
            # Copy MLP hidden layers
            for (src, dst) in zip(
                mlp_model.mlp_layers.parameters(),
                self.mlp_layers.parameters(),
            ):
                dst.data.copy_(src.data)

        # Initialize the fusion output layer from pretrained output layers
        if gmf_model is not None and mlp_model is not None:
            gmf_weight = gmf_model.output_layer.weight.data
            mlp_weight = mlp_model.output_layer.weight.data

            self.output_layer.weight.data.copy_(
                torch.cat([alpha * gmf_weight, (1 - alpha) * mlp_weight], dim=-1)
            )
            self.output_layer.bias.data.copy_(
                alpha * gmf_model.output_layer.bias.data
                + (1 - alpha) * mlp_model.output_layer.bias.data
            )

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass through both GMF and MLP paths with fusion.

        Args:
            user_ids: User indices, shape (batch_size,).
            item_ids: Item indices, shape (batch_size,).

        Returns:
            Predicted scores, shape (batch_size,).
        """
        # GMF path
        gmf_user = self.gmf_user_embedding(user_ids)  # (B, D_gmf)
        gmf_item = self.gmf_item_embedding(item_ids)  # (B, D_gmf)
        gmf_vector = gmf_user * gmf_item               # (B, D_gmf)

        # MLP path
        mlp_user = self.mlp_user_embedding(user_ids)  # (B, D_mlp)
        mlp_item = self.mlp_item_embedding(item_ids)  # (B, D_mlp)
        mlp_concat = torch.cat([mlp_user, mlp_item], dim=-1)  # (B, 2*D_mlp)
        mlp_vector = self.mlp_layers(mlp_concat)       # (B, hidden[-1])

        # Fusion
        fusion = torch.cat([gmf_vector, mlp_vector], dim=-1)  # (B, D_gmf + hidden[-1])
        logits = self.output_layer(fusion).squeeze(-1)  # (B,)
        return logits
