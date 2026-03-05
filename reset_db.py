import os 
import sqlite3 as db
import pandas

def reset_db():
    DB_PATH = 'data/transactions.db'
    if not os.path.exists(DB_PATH): 
        print('no file found at {DB_PATH}')
    else: 
        print('database found at {DB_PATH}, removing and cleaning slate')
        os.remove(DB_PATH)
        print('database removed')

if __name__ == '__main__': 
    reset_db()