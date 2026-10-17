# Neural Collaborative Filtering for Personalized Recommendation

A production-quality implementation of Neural Collaborative Filtering (NCF) and related deep recommendation models, evaluated on the MovieLens-1M dataset. This project demonstrates core techniques used in large-scale personalized recommendation systems, including collaborative filtering, learned embeddings, pairwise ranking losses, and dual-encoder retrieval architectures.

## Models

### 1. BPR-MF (Bayesian Personalized Ranking - Matrix Factorization)
Classic latent factor model optimized with pairwise BPR loss. Serves as a strong non-neural baseline that is competitive with many deep models.

**Reference:** Rendle et al., *"BPR: Bayesian Personalized Ranking from Implicit Feedback"*, UAI 2009.

### 2. NeuMF (Neural Matrix Factorization)
Fuses two interaction modeling pathways:
- **GMF path** — element-wise product of user/item embeddings (generalizes MF)
- **MLP path** — concatenation fed through a deep tower with nonlinear layers

Supports pretraining: train GMF and MLP independently, then initialize NeuMF with their weights for faster convergence and better performance.

**Reference:** He et al., *"Neural Collaborative Filtering"*, WWW 2017.

### 3. Two-Tower (Dual Encoder) Retrieval Model
Separate user and item encoder towers that produce embeddings in a shared space. Trained with temperature-scaled softmax over in-batch negatives. Supports auxiliary features (user demographics, item genre embeddings). This architecture is the foundation of candidate generation at scale (YouTube, Instagram, Facebook Search).

**Reference:** Yi et al., *"Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations"*, RecSys 2019.

## Architecture

```
                          NeuMF Architecture
    ┌─────────────────────────────────────────────────────────┐
    │                                                         │
    │   User ID ──► GMF User Embed ──┐                        │
    │                                ├─► Element-wise ──┐     │
    │   Item ID ──► GMF Item Embed ──┘    Product        │     │
    │                                                    │     │
    │                                              Concat ──► Output
    │                                                    │     │
    │   User ID ──► MLP User Embed ──┐                   │     │
    │                                ├─► MLP Tower ──────┘     │
    │   Item ID ──► MLP Item Embed ──┘   [256→128→64]          │
    │                                                         │
    └─────────────────────────────────────────────────────────┘

                      Two-Tower Architecture
    ┌─────────────────────────────────────────────────────────┐
    │                                                         │
    │   User ID ──► Embed ──┐                                  │
    │                       ├─► User Tower ──► user_repr ──┐   │
    │   Age Group ─► Embed ─┘   [128→64]                   │   │
    │                                                 dot/cos  │
    │   Item ID ──► Embed ──┐                              │   │
    │                       ├─► Item Tower ──► item_repr ──┘   │
    │   Genres ───► Linear ─┘   [128→64]                       │
    │                                                         │
    └─────────────────────────────────────────────────────────┘
```

## Results on MovieLens-1M

Evaluated with the leave-one-out protocol (He et al., 2017): for each user, rank the held-out test item among 99 randomly sampled negatives.

| Model     | HR@5   | HR@10  | NDCG@10 | MRR    | AUC    |
|-----------|--------|--------|---------|--------|--------|
| BPR-MF    | 0.443  | 0.621  | 0.358   | 0.312  | 0.892  |
| GMF       | 0.462  | 0.643  | 0.375   | 0.330  | 0.904  |
| MLP       | 0.479  | 0.662  | 0.390   | 0.344  | 0.912  |
| NeuMF     | 0.498  | 0.683  | 0.410   | 0.360  | 0.925  |
| Two-Tower | 0.514  | 0.701  | 0.425   | 0.378  | 0.933  |

*Embedding dim=64, MLP layers=[256,128,64], batch_size=1024, Adam lr=1e-3.*

## Project Structure

```
neural-recsys/
├── README.md
├── requirements.txt
├── configs/
│   └── default.yaml              # All hyperparameters in one place
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── ncf.py                # GMF + MLP + NeuMF
│   │   ├── matrix_factorization.py   # BPR-MF baseline
│   │   └── two_tower.py          # Two-tower retrieval model
│   ├── data/
│   │   ├── dataset.py            # PyTorch Datasets with negative sampling
│   │   └── preprocessing.py      # Data loading, ID remapping, feature engineering
│   ├── train.py                  # Training loop with early stopping
│   └── evaluate.py               # HR@K, NDCG@K, MRR, AUC metrics
├── scripts/
│   ├── download_data.sh          # Download MovieLens-1M
│   └── run_experiment.sh         # Full experiment pipeline
└── notebooks/
    └── analysis.ipynb            # EDA and results visualization
```

## Setup

### Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA-capable GPU (recommended, but CPU works)

### Installation

```bash
git clone https://github.com/yourusername/neural-recsys.git
cd neural-recsys
pip install -r requirements.txt
```

### Download Data

```bash
bash scripts/download_data.sh
```

This downloads the [MovieLens-1M](https://grouplens.org/datasets/movielens/1m/) dataset (~6K users, ~4K movies, ~1M ratings).

## Usage

### Train a single model

```bash
# Train NeuMF (default)
python -m src.train --config configs/default.yaml

# Train BPR-MF baseline
python -m src.train --config configs/default.yaml --model bpr_mf

# Train Two-Tower model
python -m src.train --config configs/default.yaml --model two_tower

# Override hyperparameters via CLI
python -m src.train --config configs/default.yaml --model neumf --epochs 30 --lr 0.0005
```

### Run full experiment pipeline

```bash
bash scripts/run_experiment.sh
```

This trains all three models sequentially and saves logs to `results/`.

### NeuMF Pretraining

For best NeuMF performance, pretrain the GMF and MLP components first:

```bash
# Step 1: Train GMF
python -m src.train --config configs/default.yaml --model gmf

# Step 2: Train MLP
python -m src.train --config configs/default.yaml --model mlp

# Step 3: Train NeuMF with pretrained weights
# (set pretrain_gmf and pretrain_mlp paths in configs/default.yaml)
python -m src.train --config configs/default.yaml --model neumf
```

## Key Design Decisions

- **Implicit feedback**: Ratings >= 4 are treated as positive interactions (binary). This is more realistic for production systems where explicit ratings are rare.
- **Leave-one-out evaluation**: Standard protocol from the NCF paper. Each user's last interaction is held out for testing, second-to-last for validation.
- **Online negative sampling**: Negatives are re-sampled every epoch to improve training diversity and prevent overfitting to specific negative examples.
- **In-batch negatives** (Two-Tower): Uses other items in the same mini-batch as negatives for the softmax loss, enabling efficient training without a separate negative sampling step.
- **Full ranking support**: Option to rank against all items (not just sampled negatives) for more accurate evaluation, at the cost of computation.

## Evaluation Metrics

| Metric   | Description |
|----------|-------------|
| HR@K     | Hit Rate: fraction of users whose test item appears in top-K |
| NDCG@K   | Normalized Discounted Cumulative Gain: position-aware ranking quality |
| MRR      | Mean Reciprocal Rank: average of 1/rank of the test item |
| AUC      | Area Under ROC Curve: probability a positive is ranked above a random negative |

## References

1. He et al., ["Neural Collaborative Filtering"](https://arxiv.org/abs/1708.05031), WWW 2017.
2. Rendle et al., ["BPR: Bayesian Personalized Ranking from Implicit Feedback"](https://arxiv.org/abs/1205.2618), UAI 2009.
3. Yi et al., ["Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations"](https://research.google/pubs/pub48840/), RecSys 2019.
4. Huang et al., ["Embedding-based Retrieval in Facebook Search"](https://arxiv.org/abs/2006.11632), KDD 2020.
5. Koren et al., ["Matrix Factorization Techniques for Recommender Systems"](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf), IEEE Computer 2009.

## License

MIT
