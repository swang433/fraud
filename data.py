import pandas as pd
import numpy as np

def feat_eng(df, drop_ids = True): #main function for computing features
    df = df.drop(columns=['newbalanceOrig', 'newbalanceDest', 'isFlaggedFraud'], errors='ignore')
    if drop_ids:
        df = df.drop(columns=['nameOrig', 'nameDest'], errors='ignore')
    df = df.rename(columns={'oldbalanceOrg': 'oldbalanceOrig'})
    df = df.sort_values(['step'])
    df['hour'] = (df['step'] % 24).astype(int)
    df['day'] = (df['step'] // 24).astype(int)
    #one hot encoding VERY IMPORTANT!!!
    if 'type' in df.columns:
        df = pd.get_dummies(df, columns=['type'])
    df['large'] = (df['amount'] > 500000).astype(int)
    df['very_large'] = (df['amount'] > 2000000).astype(int)
    df['log_amount'] = np.log1p(df['amount'])
    df['percentage_sent'] = np.where(df['amount'] >= df['oldbalanceOrig'], 100, df['amount'] / df['oldbalanceOrig'] * 100)
    df['balance_depleted'] = (df['percentage_sent'] == 100).astype(int)
    return df

def savings(y_true, y_prob, amts, cost_per_fp=2): #purely for monetary evaluation 
    '''
    works well for datasets with extreme class imbalance; 
    business-centric metric for how much money is saved after running the model 
    cost_fn is the transaction value by default (lose if false negative)
    
    intuition: 
    for every threshold there is, use a set savings function (
        amount caught, or sum of true positives - cost - amount lost, or sum of false negatives    
    )
    to estimate the amount
    of money saved after running the model
    '''
    best_savings, best_threshold = 0, .5
    for threshold in np.arange(.01, 1, .01): 
        y_pred = (y_prob > threshold).astype(int)
        tp, fp = (y_pred == 1) & (y_true == 1), (y_pred == 1) & (y_true == 0)
        fn = (y_pred == 0) & (y_true == 1)
        curr_saving = amts[tp].sum() - cost_per_fp * fp.sum() - amts[fn].sum()
        if curr_saving > best_savings: 
            best_savings, best_threshold = curr_saving, threshold
    return best_savings, best_threshold