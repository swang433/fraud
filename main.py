import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import sklearn
from xgboost import XGBClassifier
# from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import sqlite3 as db
import os
from sklearn.metrics import classification_report, average_precision_score

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
        
        #trivial feature engineering/encoding
        HRS_PER_DAY = 24
        df['hour'] = (df['step'] % HRS_PER_DAY).astype(int)
        df['day'] = (df['step'] // HRS_PER_DAY).astype(int)
        df = pd.get_dummies(df, columns=['type'])
        
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
            ), 0) AS usr_total_received,
            
            COALESCE(COUNT(*) OVER (
                PARTITION BY nameDest
                ORDER BY step
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ), 0) AS usr_num_tx_dest

        FROM (SELECT * FROM transactions_db ORDER BY nameOrig, step)
        """
        
    df = pd.read_sql_query(query, conn) #read_sql_query returns a standard pandas dataframe
    df = df.drop(columns=['nameOrig', 'nameDest']) #XGBoost can only process numerical cols
    x, y = df.drop('isFraud', axis=1), df['isFraud']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=42)
    
    #model selection
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    model = XGBClassifier(scale_pos_weight=neg/pos, random_state=42)
    
    #train and view results
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    
    '''
    not a viable metric when it comes to model performance evaluation since 
    the extreme class imbalance results in high accuracy with little effort
    (the extremely small subset of fraudulent samples are what we want to detect)
    '''
    accuracy = accuracy_score(y_pred, y_test)
    print('Model accuracy: ' + str(100 * accuracy) + '%')
    
    '''
    precision: of every flagged sample how many were actually really fraud 
    (true positives) / (true + false positives)
                
    recall: of every real fraudulent samples, how many were detected
    (true positives) / (true positives + false negatives) 
    
    in a balanced-class dataset, usage of both PR AUC and ROC AUC are permitted since
    the data isn't dominated by negatives
    '''
    print(classification_report(y_test, y_pred))
    print('PR AUC:', average_precision_score(y_test, model.predict_proba(x_test)[:, 1]))
    
    conn.close()