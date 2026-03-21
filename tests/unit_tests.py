import pandas as pd
import numpy as np
import pytest
import app
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from data import feat_eng

''''
unit tests that check if functions like feat_eng, savings, and api endpoints are working properly
'''

#mock datamframe
def make_df(**overides): 
    '''
    one transaction causes balance insufficiency
    '''
    df = {
        'step': [1, 2, 3], 
        'amount': [67, 500001, 2000001], #should result in normal, large, and very large transactions 
        'oldbalanceOrig': [10, 1000000, 4000000], 
        'oldbalanceDest': [2507.68, 0, 70], 
        'newbalanceOrig': [0, 499999, 1999999], 
        'newbalanceDest': [2574.68, 500001, 2000071], 
        'isFlaggedFraud': [0, 1, 0], 
        'type': ['PAYMENT', 'TRANSFER', 'CASH_OUT'], 
        'nameOrig': ['C1', 'C2', 'C3'], 
        'nameDest': ['M1', 'M2', 'M3']
    }
    df.update(overides)
    return pd.DataFrame(df)

@pytest.fixture
def client(): 
    '''
    generates mock inference results and intercepts load model functions and serves a mock model instead
    (simulates a simple api-served model)
    '''
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[.8, .2]])
    with patch('app.joblib.load', return_value=mock_model): 
        return TestClient(app.app)

test_df = feat_eng(make_df())
print(test_df)

def test_cols(): #checks if all engineered features are present
    for col in ['hour', 'day', 'large', 'very_large', 'log_amount', 'percentage_sent', 'balance_depleted']: 
        assert col in test_df.columns

def test_transaction_size():     
    assert test_df.iloc[0]['large'] == 0 and test_df.iloc[0]['very_large'] == 0
    assert test_df.iloc[1]['large'] == 1 and test_df.iloc[1]['very_large'] == 0
    assert test_df.iloc[2]['large'] == 1 and test_df.iloc[2]['very_large'] == 1
    
def percentage_sent(): #tests if balance insuffiency throws an exception or causes an error
    assert test_df.iloc[0]['percentage_sent'] == 100

def test_depletion(): 
    assert test_df.iloc[0]['balance_depleted'] == 1
    assert test_df.iloc[1]['balance_depleted'] == 0
    assert test_df.iloc[2]['balance_depleted'] == 0