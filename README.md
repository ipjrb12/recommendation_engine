# Recommendation Engine

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?style=flat-square&logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

A collaborative filtering recommendation engine built on Amazon product interaction data. Uses learned embedding layers for user and product IDs to estimate a sparse rating matrix and surface personalised product recommendations.

## What it does

The engine learns low-dimensional embeddings for each user and product from implicit interaction data. During inference, a user's embedding is compared against all product embeddings to rank recommendations. Two versions are included: a baseline collaborative filter (`first.py`) and an extended version with deeper embedding layers (`big_first.py`).

## Running

```bash
git clone https://github.com/ipjrb12/recommendation_engine.git
cd recommendation_engine
pip install torch numpy
python first.py          # baseline
python big_first.py      # extended version
```

The dataset (`amazon_dataset.json`) is included in the repo.

## Project structure

```
├── first.py              # Baseline collaborative filter
├── big_first.py          # Extended embedding model
├── second.py             # Evaluation script
└── amazon_dataset.json   # Interaction dataset
```

## Tools

- PyTorch (embedding layers, sparse matrix operations)
- Amazon product interaction dataset

## License

MIT
