from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# ---- Step 1: Load the saved model and scaler ----
model = joblib.load("src/kmeans_model.pkl")
scaler = joblib.load("src/scaler.pkl")

# ---- Step 2: Create the FastAPI app ----
app = FastAPI(title="Customer Segmentation API")

# ---- Step 3: Define what data the API expects ----
class Customer(BaseModel):
    recency: float
    frequency: float
    monetary: float

# ---- Step 4: Define the prediction endpoint ----
@app.post("/predict-segment")
def predict_segment(customer: Customer):
    # Arrange the input the same way as during training
    X = np.array([[customer.recency, customer.frequency, customer.monetary]])
    X_scaled = scaler.transform(X)

    # Predict which segment this customer belongs to
    segment = model.predict(X_scaled)[0]

    return {"segment": int(segment)}

# ---- Step 5: A simple homepage route (just to check the API is alive) ----
@app.get("/")
def home():
    return {"message": "Customer Segmentation API is running"}