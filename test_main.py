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

def test_main():
    if not os.path.exists(DB_PATH): 
        
        #read and clean df
        df = pd.read_csv('data/transactions.csv')
        print('successfully read dataframe')
        df = df.drop(columns=['newbalanceOrig', 'newbalanceDest', 'isFlaggedFraud'])
        print('cheating columns dropped')
        df = df.rename(columns={'oldbalanceOrg': 'oldbalanceOrig'})
        df = df.sort_values(['nameOrig', 'step'])
        print('successfully cleaned dataframe')
        
        #light feature engineering
        HRS_PER_DAY = 24
        df['hour'] = (df['step'] % HRS_PER_DAY).astype(int)
        df['day'] = (df['step'] // HRS_PER_DAY).astype(int)
        
        #user related features
        df['to_merchant'] = df['nameDest'].str.startswith('M')
        # df['from_merchant'] = df['nameOrig'].str.startswith('M') always false
        # df['is_merchant'] = (df['to_merchant'] | df['from_merchant']).astype(int)
        
        # cost-based features
        df['large'] = (df['amount'] > 500000).astype(int)
        df['very_large'] = (df['amount'] > 2000000).astype(int)
        df['log_amount'] = np.log1p(df['amount'])
        df['percentage_sent'] = np.where(df['oldbalanceOrig'] <= 0, 100, df['amount'] / df['oldbalanceOrig'] * 100)
        df['balance_depleted'] = (df['percentage_sent'] == 100).astype(int)
        
        conn = db.connect(DB_PATH)
        df.to_sql(name='transactions_db', con=conn, if_exists='replace', index=False)
        conn.close()
        print('successfully created database')
    
if __name__ == "__main__":
    test_main()
    conn = db.connect(DB_PATH)
    
    query = """
        SELECT *,
        
            COALESCE(AVG(amount) OVER (
                PARTITION BY nameDest
                ORDER BY step
                ROWS BETWEEN 24 PRECEDING AND 1 PRECEDING
            ), 0) AS avg_amt_rec_L24hrs,
    
            COALESCE(COUNT(*) OVER (
                PARTITION BY nameDest
                ORDER BY step
                ROWS BETWEEN 24 PRECEDING AND 1 PRECEDING
            ), 0) AS tx_count_L24hrs_dest,
            
            COALESCE(SUM(amount) OVER (
                PARTITION BY nameDest
                ORDER BY step
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ), 0) AS total_received_dest,
            
            COALESCE(COUNT(*) OVER (
                PARTITION BY nameDest
                ORDER BY step
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ), 0) AS usr_num_tx_dest

        FROM (SELECT * FROM transactions_db ORDER BY nameOrig, step)
        """
    df = pd.read_sql_query(query, conn) #read_sql_query returns a standard pandas dataframe
    print(df)
    conn.close()