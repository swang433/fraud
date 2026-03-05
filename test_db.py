import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import sqlite3 as db
import os

DB_PATH = 'data/transactions.db'
HRS_PER_DAY = 24
if not os.path.exists(DB_PATH): 
    df = pd.read_csv('data/transactions.csv')
    print('successfully read dataframe')
    df = df.drop(columns=['newbalanceOrig', 'newbalanceDest', 'isFlaggedFraud'])
    print('cheating columns dropped')
    df = df.rename(columns={'oldbalanceOrg': 'oldbalanceOrig'})
    df = df.sort_values(['nameOrig', 'step'])
    print('successfully cleaned dataframe')
    conn = db.connect(DB_PATH)
    df.to_sql(name='transactions_db', con=conn, if_exists='replace', index=False)
    conn.close()
    print('successfully created database')
    
conn = db.connect(DB_PATH)
query = """
SELECT *,
    COALESCE(AVG(amount) OVER (
        PARTITION BY nameDest
        ORDER BY step
        ROWS BETWEEN 24 PRECEDING AND 1 PRECEDING
    ), 0) AS amt_avg_L24hrs_dest
FROM transactions_db
"""
df = pd.read_sql_query(query, conn) #read_sql_query returns a standard pandas dataframe
print(df)
conn.close()