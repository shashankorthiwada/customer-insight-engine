from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import pandas as pd
import joblib
import os

app = FastAPI(title="Customer Value Prediction API")

# Constants
MODEL_PATH = "customer_value_model.pkl"
EXCEL_PATH = "Food_delivery_customer_history_latest.xlsx"

# Load model
model = joblib.load(MODEL_PATH)

# Feature columns (must match training)
feature_columns = [
    "Average Order Value (AOV)",
    "Order Frequency",
    "Purchase Recency",
    "Customer Tenure (Months)",
    "Churn Risk Score",
    "Discount Dependence Score",
    "Margin Contribution per Order",
    "Referral Activity Score",
    "Platform Engagement Score",
    "Variety of Orders",
    "Loyalty Program Participation",
    "High-Value Zone Presence",
    "Delivery Cost Efficiency",
    "Order Smoothness Score",
    "Feedback & Ratings Contribution",
    "Multi-Channel Usage"
]

# Request schema for /predict
class PredictionRequest(BaseModel):
    customer_id: int


# Utilities
def get_cust_details(cust_id: int):
    if not os.path.exists(EXCEL_PATH):
        raise FileNotFoundError("Excel file not found.")

    df = pd.read_excel(EXCEL_PATH)
    type(print(cust_id))
    print(cust_id)
    cust_info = df[df['Customer ID'] == cust_id].copy()

    if cust_info.empty:
        raise ValueError(f"Customer ID {cust_id} not found.")

    cust_info.drop('Customer ID', axis=1, inplace=True)
    return cust_info.values.tolist()  # list of list


def predict_new_customer(cust_data: list, model, feature_columns: list) -> int:
    input_df = pd.DataFrame(cust_data, columns=feature_columns)
    prediction = model.predict(input_df)[0]
    return int(prediction)


# Routes
@app.get("/")
def read_root():
    return {"message": "Customer Value Prediction API is running."}


# v1 JSON-based prediction
@app.post("/v1/predict")
async def predict_by_json(request: Request):
    try:
        data = await request.json()

        missing = [col for col in feature_columns if col not in data]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing fields: {missing}")

        input_df = pd.DataFrame([data], columns=feature_columns)
        prediction = model.predict(input_df)[0]
        return {"prediction": int(prediction)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Default prediction using customer_id from Excel
@app.post("/v2/predict")
async def predict_by_customer_id(request: PredictionRequest):
    try:
        cust_data = get_cust_details(request.customer_id)
        prediction = predict_new_customer(cust_data, model, feature_columns)
        return {
            "customer_id": request.customer_id,
            "prediction": prediction
        }

    except FileNotFoundError as fe:
        raise HTTPException(status_code=500, detail=str(fe))
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
