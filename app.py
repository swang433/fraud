from fastapi import FastAPI, File, UploadFile, HTTPException
import joblib
import io
import pandas as pd
from data import feat_eng
import yaml

with open('config.yaml') as f: 
    config = yaml.safe_load(f)
app = FastAPI(title='Transaction Fraud Detector API')
model = joblib.load(config['model']['serve'])

@app.post('/predict')
async def predict_fraud(file: UploadFile = File(...)):
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))
    df = df.drop(columns=['isFraud'], errors='ignore')  # in case it's in the uploaded CSV
    
    #throws an exception instead of internal server error
    required = ['avg_amt_rec_L24hrs', 'tx_count_L24hrs_dest', 'usr_total_received', 'usr_num_tx_dest']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing required window features: {missing}")
    
    df = feat_eng(df)
    prob = model.predict_proba(df)[:, 1]
    return {"fraud_probabilities": prob.tolist()}

@app.get('/')
def root(): 
    return {'status': 'online', 'model': 'XGBoost Fraud Detector'}

# NOTE: run main, precompute, then start api: fastapi dev app.py