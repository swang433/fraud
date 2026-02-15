import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# read and clean dataframe (drop columns that could be considered cheating)
df = pd.read_csv('data/transactions.csv')
df = df.drop(columns=['newbalanceOrig', 'newbalanceDest', 'isFlaggedFraud'])
HOURS_IN_DAY = 24
print('data frame read and clean successful')

# temporal features (not significant)
df['hour'] = df['step'] % HOURS_IN_DAY
df['day'] = df['step'] // HOURS_IN_DAY
df['peak_hours'] = ((df['hour'] >= 8) & (df['hour'] <= 21)).astype(int)

# user distinction
df['merchant_transaction_orig'] = df['nameOrig'].str.startswith('M').astype(int)
df['merchant_transaction_dest'] = df['nameDest'].str.startswith('M').astype(int)
df['merchant_transaction'] = (df['merchant_transaction_dest'] | df['merchant_transaction_orig']).astype(int)

# see which features correlate the most with 'isFraud'
numeric_cols = df.select_dtypes(include=[np.number]).columns
fig, ax = plt.subplots(1, 2, figsize=(15,5))

ax[0].set_title("Fraudulent Records correlation after clean")
sns.heatmap(
    df.query('isFraud == 1')[numeric_cols].drop(columns=['isFraud']).corr(),  
    cmap="OrRd", 
    ax=ax[0]
)

ax[1].set_title("Non-fraudulent Records correlation after clean")
sns.heatmap(
    df.query('isFraud == 0')[numeric_cols].drop(columns=['isFraud']).corr(),  
    cmap="Blues", 
    ax=ax[1]
)
plt.show()

# print(df.columns)
# print(df.describe())
# print(df['merchant_transaction'])