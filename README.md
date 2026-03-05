# Credit Card Fraud Detection

A machine learning project for detecting fraudulent credit card transactions. (Everything is to be modified)

## Overview

This project implements ML models to identify potentially fraudulent transactions in credit card data.

## Features

- Data preprocessing and feature engineering
- Multiple ML model implementations
- Model evaluation and comparison
- Fraud detection pipeline

## Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Usage

```bash
python main.py
```

## Project Structure

```
.
├── data/           # Dataset files
├── models/         # Trained models
├── notebooks/      # Jupyter notebooks
├── src/            # Source code
└── tests/          # Unit tests
```

## License

[To be added]

## Issues Resolved

Issue: 

The multi-index intermediate structure created during the use of the groupby().rolling() functions caused complete memory exhaustion

```python
df['amt_avg_L24hrs'] = df.groupby('nameOrig', group_keys=False)['amount'].apply(
    lambda x: x.rolling(window=24, min_periods=1).mean().shift(1)
)
```

Solution: 

Instead of performing expensive operations like rolling() in a memory-held dataframe, using SQLite window functions is significantly more memory efficient since it computes results row-by-row with O(1) space per window, eliminating the need for intermediate memory allocation. 

## Contributing

Contributions welcome. Please open an issue first to discuss proposed changes.