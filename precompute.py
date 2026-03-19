import pandas as pd
import sqlite3 as db

DB_PATH, SAMPLE_PATH = 'data/transactions.db', 'data/samples.csv'

'''
parsing a whole dataframe into an api-based model is not viable, so this file
computes some of the missing features the model needs and extracts 100 samples 
for the api to send to the model

use a sqlite query to compute rolling and expanding features: 
compute avg_amt_rec_L24hrs, tx_count_L24hrs_dest, usr_total_received, usr_num_tx_dest

df['avg_amt_rec_L24HRS'] = df.groupby('nameDest', group_keys=False)['amount'].transform(
    lambda x: x.rolling(24, min_periods=1).mean().shift(1)
)
df['tx_count_L24hrs_dest'] = df.groupby('nameDest', group_keys=False)['amount'].transform(
    lambda x: x.rolling(24, min_periods=1).count().shift(1)
)
df['usr_total_received'] = df.groupby('nameDest', group_keys=False)['amount'].transform(
    lambda x: x.expanding().sum().shift(1)
)
df['usr_num_tx_dest'] = df.groupby('nameDest', group_keys=False)['amount'].transform(
    lambda x: x.expanding().count().shift(1)
)
'''

query = """
    SELECT *, 

    COALESCE (AVG(amount) OVER (
        PARTITION BY nameDest ORDER BY step
        ROWS BETWEEN 24 PRECEDING AND 1 PRECEDING
    ), 0) AS avg_amt_rec_L24hrs, 

    COALESCE (COUNT(*) OVER (
        PARTITION BY nameDest ORDER BY step
        ROWS BETWEEN 24 PRECEDING AND 1 PRECEDING
    ), 0) AS tx_count_L24hrs_dest, 

    COALESCE (SUM(amount) OVER (
        PARTITION BY nameDest ORDER BY step
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ), 0) AS usr_total_received, 

    COALESCE (COUNT(*) OVER (
        PARTITION BY nameDest ORDER BY step
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ), 0) AS usr_num_tx_dest
    
    FROM (SELECT * FROM transactions_db ORDER BY step)
    """
conn = db.connect(DB_PATH)
df = pd.read_sql_query(query, conn)
conn.close()
df.sample(100).to_csv(SAMPLE_PATH, index=False)
print('feature precomputation complete')