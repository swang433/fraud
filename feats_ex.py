import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
# import hiplot as hip

# read and clean dataframe (DROP FEATURES THAT COULD BE CONSIDERED CHEATING)
df = pd.read_csv('data/transactions.csv')
df = df.drop(columns=['newbalanceOrig', 'newbalanceDest', 'isFlaggedFraud'])
HOURS_IN_DAY = 24
df = df.rename(columns={'oldbalanceOrg': 'oldbalanceOrig'})
df = df.sort_values(['nameOrig', 'step'])
print('data frame read and clean successful')

'''
1. cost-based features that flag large transactions
2. aggregated features that group by cardholder + one other attribute like country or cardholder type
3. periodic time features via von Mises distribution
'''

# temporal features (not significant on their own)
df['hour'] = df['step'] % HOURS_IN_DAY
df['day'] = df['step'] // HOURS_IN_DAY
df['peak_hours'] = ((df['hour'] >= 8) & (df['hour'] <= 21)).astype(int)

# user distinction features
df['merchant_transaction_orig'] = df['nameOrig'].str.startswith('M').astype(int)
df['merchant_transaction_dest'] = df['nameDest'].str.startswith('M').astype(int)
df['is_merchant'] = (df['merchant_transaction_dest'] | df['merchant_transaction_orig']).astype(int)

# cost-based features
df['large'] = (df['amount'] > 500000).astype(int)
df['very_large'] = (df['amount'] > 2000000).astype(int)
df['log_amount'] = np.log(df['amount'].clip(lower=1e-10))
df['percentage_sent'] = np.where(df['oldbalanceOrig'] <= 0, 100, df['amount'] / df['oldbalanceOrig'] * 100)
df['balance_depleted'] = (df['percentage_sent'] == 100).astype(int)

#aggregated features
df['user_avg_amount'] = df.groupby('nameOrig')['amount'].transform('mean')

# More efficient rolling average calculation (NEED SQL INTEGRATION !!!)
# df['amt_avg_L24hrs'] = df.groupby('nameOrig', group_keys=False)['amount'].apply(
#     lambda x: x.rolling(window=24, min_periods=1).mean().shift(1)
# )

# print(df[['amt_avg_L24hrs', 'user_avg_amount']])

# print(df.describe())
print(df.columns)
# print(df['amount'].median()) median is ~75k
# print(df['merchant_transaction'])