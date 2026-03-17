import pandas as pd
import sqlite3 as db

conn = db.connect('data/transactions.db')

query = """
    SELECT *,
        COALESCE(AVG(amount) OVER (
            PARTITION BY nameDest ORDER BY step
            ROWS BETWEEN 24 PRECEDING AND 1 PRECEDING
        ), 0) AS avg_amt_rec_L24hrs,
        COALESCE(COUNT(*) OVER (
            PARTITION BY nameDest ORDER BY step
            ROWS BETWEEN 24 PRECEDING AND 1 PRECEDING
        ), 0) AS tx_count_L24hrs_dest,
        COALESCE(SUM(amount) OVER (
            PARTITION BY nameDest ORDER BY step
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ), 0) AS usr_total_received,
        COALESCE(COUNT(*) OVER (
            PARTITION BY nameDest ORDER BY step
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ), 0) AS usr_num_tx_dest
    FROM (SELECT * FROM transactions_db ORDER BY step)
"""

df = pd.read_sql_query(query, conn)
conn.close()

# save a small sample to test via API
df.sample(100).to_csv('data/test_sample.csv', index=False)
print('precomputed required columns')