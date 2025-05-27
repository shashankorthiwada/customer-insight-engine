from fastapi import FastAPI, HTTPException, Request
import pandas as pd
import joblib

model = joblib.load("customer_value_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Customer Value Prediction API"}

@app.post("/predict")
async def predict(request: Request):
    try:
        data = await request.json()

        # Check for missing features
        missing = [col for col in feature_columns if col not in data]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing fields: {missing}")

        input_df = pd.DataFrame([data], columns=feature_columns)
        prediction = model.predict(input_df)[0]
        return {"prediction": int(prediction)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
