# Credit Card Fraud Detection

A machine learning project for detecting fraudulent credit card transactions. (Everything is to be modified)

## Overview

This project implements ML models to identify potentially fraudulent transactions in credit card data.

## Features

- Data preprocessing and feature engineering
- Model evaluation and comparison
- Fraud detection pipeline

## Design/Evaluation Choices

Chosen model: XGBoost with scale_pos_weight=neg/pos

Chosen Metrics: 

- accuracy: this metric is not a viable way to evaluate models in datasets with extreme class imbalance (would yield >=99% accuracy regardless)
- precision-recall area-under-curve: the lack of focus on true negatives here is vital. Moreover, this metric is also sensitive to false positives due to the usage of the precision = tp / (tp + fp), hence preventing over-predicting potentially postive cases. 
- custom savings function: not a model evalutation metric, but one for more business-centric evalutation; measures how much money is saved from fraudulent transactions (also punishes over-predicting positive cases here).

```python
def savings(y_true, y_prob, amts, cost_per_fp=2): 
    best_savings, best_threshold = 0, .5
    for threshold in np.arange(.01, 1, .01): 
        y_pred = (y_prob > threshold).astype(int)
        tp, fp = (y_pred == 1) & (y_true == 1), (y_pred == 1) & (y_true == 0)
        fn = (y_pred == 0) & (y_true == 1)
        curr_saving = amts[tp].sum() - cost_per_fp * fp.sum() - amts[fn].sum()
        if curr_saving > best_savings: 
            best_savings, best_threshold = curr_saving, threshold
    return best_savings, best_threshold
```

### Prerequisites

```bash
pip install -r requirements.txt
```

### Usage

```bash
python main.py; python precompute; fastapi dev app.py
```

## Project Structure

```
.
├── source code files
├── data/           # Dataset files
├── models/         # Trained models
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

Instead of performing expensive operations like rolling() in a memory-held dataframe, using SQLite window functions is significantly more memory efficient since it computes results row-by-row with O(1) space per window, eliminating the need for intermediate memory allocation. Same feature above can be computed with:

```sql
COALESCE (AVG(amount) OVER (
        PARTITION BY nameDest ORDER BY step
        ROWS BETWEEN 24 PRECEDING AND 1 PRECEDING
    ), 0) AS avg_amt_rec_L24hrs
```

Issue: 

Some time sensitive features (total received per user, average amount received per user for the last 24 hours, etc) cannot be computed at inference time since transactions would likely be processed one sample at a time in an industry setting. 

Solution:

Precompute.py resolves this, since there typically would be a separate variable or function responsible for keeping track of these behavioral statistics for live time inference decisions (my own primitive take at a feature store). 

Issue: 

Training vs api-serving data skew and inconsistency. When building the api that serves inferences results, I failed to consider that the raw csv used for testing does not include the behavior time sensitive features mentioned above. 

Solution: 

Also resolved with precompute.py since it computes the same features that the model needs for the training process. 

Issue: 

lorem ipsum

Solution: 

lorem ipsum