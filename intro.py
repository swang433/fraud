import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import sklearn
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import hiplot as hip

# read and clean dataframe (DROP FEATURES THAT COULD BE CONSIDERED CHEATING)
df = pd.read_csv('data/transactions.csv')
df = df.drop(columns=['newbalanceOrig', 'newbalanceDest', 'isFlaggedFraud'])
HOURS_IN_DAY = 24
print('data frame read and clean successful')

# temporal features (not significant)
df['hour'] = df['step'] % HOURS_IN_DAY
df['day'] = df['step'] // HOURS_IN_DAY
df['peak_hours'] = ((df['hour'] >= 8) & (df['hour'] <= 21)).astype(int)

# user distinction features
df['merchant_transaction_orig'] = df['nameOrig'].str.startswith('M').astype(int)
df['merchant_transaction_dest'] = df['nameDest'].str.startswith('M').astype(int)
df['merchant_transaction'] = (df['merchant_transaction_dest'] | df['merchant_transaction_orig']).astype(int)

# flagging large transactions
df['large'] = (df['amount'] > 500000).astype(int)
df['very_large'] = (df['amount'] > 2000000).astype(int)

# check amount percentiles since its the most correlated with isFraud
fraud = df[df['isFraud'] == 1]['amount'].quantile([.2, .4, .6, .8, .95])
no_fraud = df[df['isFraud'] == 0]['amount'].quantile([.2, .4, .6, .8, .95])
plt.figure(figsize=(12, 5))

#visualize quantile features
plt.subplot(1, 2, 1)
plt.barh(range(len(no_fraud)), no_fraud.values)
plt.yticks(range(len(no_fraud)), no_fraud.index)
plt.xlabel('Amount')
plt.ylabel('Percentile')
plt.title('Non-Fraudulent Transaction Amounts')

plt.subplot(1, 2, 2)
plt.barh(range(len(fraud)), fraud.values)
plt.yticks(range(len(fraud)), fraud.index)
plt.xlabel('Amount')
plt.ylabel('Percentile')
plt.title('Fraudulent Transaction Amounts')

plt.tight_layout()

# see which features correlate the most with 'isFraud'
numeric_cols = df.select_dtypes(include=[np.number]).columns
correlation_with_fraud = df[numeric_cols].corr()['isFraud'].drop('isFraud').sort_values(ascending=False)
# print("Features most correlated with isFraud:")
# print(correlation_with_fraud)

# Visualize correlations
plt.figure(figsize=(10, 6))
correlation_with_fraud.plot(kind='barh')
plt.title('Feature Correlation with isFraud')
plt.xlabel('Correlation Coefficient')
plt.tight_layout()

df.isFraud.value_counts().plot.pie(autopct='%.2f',figsize=(5, 5), colors=["green","cyan"], explode=[0,.1])
plt.title('Class Distribution')
plt.tight_layout()
plt.show()

# Simple visualization
hip.Experiment.from_dataframe(df).display()

# print(df.describe())
print(df.columns)
# print(df['amount'].median()) median is ~75k
# print(df['merchant_transaction'])