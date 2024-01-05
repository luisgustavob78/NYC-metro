import pickle
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, File, UploadFile, HTTPException
import json
from xgboost import XGBRegressor


app = FastAPI(title="NYC metro usage prediction! Upload your csv batch")

@app.on_event("startup")
def load_model():
    # Load classifier from pickle file
    global model
    model = joblib.load("NYC_metro_model.pkl")

@app.get("/")
def home():
    return "Congratulations! Your API is working as expected. This new version allows for batching. Now head over to http://localhost:8000/docs"


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    #loading csv data
    df = pd.read_csv(file.file)

    #converting dataframe to array
    X = list(df.values)

    #generating predictions and converting it to the requested format
    y_pred = list(model.predict(X))
    y_pred = [str(value) for value in y_pred]
    
    return {"Predictions": y_pred}