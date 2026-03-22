import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import sklearn
import yaml
from xgboost import XGBClassifier
# from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import sqlite3 as db
import os
from sklearn.metrics import classification_report, average_precision_score
import joblib
from data import feat_eng, savings

with open('config.yaml') as f: 
    config = yaml.safe_load(f)

DB_PATH = config['raw_data']['db']

def build_db():
    needs_rebuild = not os.path.exists(DB_PATH)
    if not needs_rebuild:
        conn = db.connect(DB_PATH)
        cols = pd.read_sql_query('SELECT * FROM transactions_db LIMIT 1', conn).columns.tolist()
        conn.close()
        if 'isFraud' not in cols or 'nameDest' not in cols:
            needs_rebuild = True

    if needs_rebuild:
        df = pd.read_csv(config['raw_data']['csv'])
        print('successfully read dataframe')
        df = feat_eng(df, drop_ids=False)
        conn = db.connect(DB_PATH)
        df.to_sql(name='transactions_db', con=conn, if_exists='replace', index=False)
        conn.close()
        print('successfully created database')
        

if __name__ == "__main__":
    build_db()
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

        FROM (SELECT * FROM transactions_db ORDER BY step)
        """
        
    df = pd.read_sql_query(query, conn) #read_sql_query returns a standard pandas dataframe
    df = df.drop(columns=['nameOrig', 'nameDest']) #XGBoost can only process numerical cols
    x, y = df.drop('isFraud', axis=1), df['isFraud']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=42, stratify=df['isFraud'])
    # print(x_train.columns)
    
    #model selection
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    model = XGBClassifier(scale_pos_weight=neg/pos, random_state=42)
    
    #train, and save model, then view results in terminal
    model.fit(x_train, y_train)
    #save after training
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/fraud_model.pkl')
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
    savings_value, threshold = savings(y_test.values, model.predict_proba(x_test)[:, 1], x_test['amount'].values)
    print(f'Best savings: ${savings_value:,.2f} at threshold {threshold:.2f}')

    conn.close()