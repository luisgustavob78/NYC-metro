import pickle
import numpy as np
import joblib
from fastapi import FastAPI, File, UploadFile
import json
from xgboost import XGBRegressor


app = FastAPI(title="NYC metro usage prediction! Upload your json batch")

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

    contents = await file.read()
    contents = contents.decode("utf-8")

    # Parse JSON content
    json_data = json.loads(contents)

    np_batches = np.array(json_data["batches"], dtype=float)
    
    y_pred = model.predict(np_batches)
    
    return {"Prediction": y_pred}