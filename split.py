import shutil
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#construct dataframe and perform stratified train test split
df = pd.read_csv('data/transactions.csv')

train_val, test = train_test_split(
    df, 
    test_size=.2, 
    stratify=df['isFraud'], 
    random_state=42
)

train, val = train_test_split(
    train_val, 
    test_size=.1875, 
    stratify=train_val['isFraud'], 
    random_state=42
)

print('Successfully performed stratified split')