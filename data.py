import pandas as pd
import numpy as np

def feat_eng(df, drop_ids = True):
    df = df.drop(columns=['newbalanceOrig', 'newbalanceDest', 'isFlaggedFraud'], errors='ignore')
    if drop_ids:
        df = df.drop(columns=['nameOrig', 'nameDest'], errors='ignore')
    df = df.rename(columns={'oldbalanceOrg': 'oldbalanceOrig'})
    df = df.sort_values(['step'])
    df['hour'] = (df['step'] % 24).astype(int)
    df['day'] = (df['step'] // 24).astype(int)
    if 'type' in df.columns:
        df = pd.get_dummies(df, columns=['type'])
    df['large'] = (df['amount'] > 500000).astype(int)
    df['very_large'] = (df['amount'] > 2000000).astype(int)
    df['log_amount'] = np.log1p(df['amount'])
    df['percentage_sent'] = np.where(df['oldbalanceOrig'] <= 0, 100, df['amount'] / df['oldbalanceOrig'] * 100)
    df['balance_depleted'] = (df['percentage_sent'] == 100).astype(int)
    return df