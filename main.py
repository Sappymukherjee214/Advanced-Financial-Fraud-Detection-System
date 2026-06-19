import os
import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Extra
from typing import Dict, Any, List

from src.explainability import FraudRiskScorer, setup_lime_explainer, get_lime_explanation_for_instance

app = FastAPI(
    title="Advanced Financial Fraud Detection API",
    description="FastAPI service for transaction fraud scoring, risk assessment, and explainability.",
    version="1.0.0"
)

# Global variables to store our fitted models and pipelines
preprocessor = None
best_model = None
lime_explainer = None
risk_scorer = FraudRiskScorer()
train_cols = []

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.pkl")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")

# Pydantic input schema that accepts dynamic fields
class TransactionPayload(BaseModel):
    TransactionAmt: float
    ProductCD: str
    card1: int
    card2: float = np.nan
    card3: float = np.nan
    card4: str = "missing"
    card5: float = np.nan
    card6: str = "missing"
    addr1: float = np.nan
    addr2: float = np.nan
    P_emaildomain: str = "missing"
    R_emaildomain: str = "missing"
    TransactionDT: int = 86400
    
    # Allow extra fields since the IEEE-CIS dataset has over 400 columns
    class Config:
        extra = Extra.allow

@app.on_event("startup")
def load_assets():
    global preprocessor, best_model, lime_explainer, train_cols
    
    if os.path.exists(PREPROCESSOR_PATH) and os.path.exists(MODEL_PATH):
        try:
            with open(PREPROCESSOR_PATH, 'rb') as f:
                preprocessor = pickle.load(f)
            with open(MODEL_PATH, 'rb') as f:
                best_model = pickle.load(f)
            
            # Setup LIME Explainer using a small background dataset from training data
            train_sampled_path = os.path.join(BASE_DIR, "data", "train_sampled.csv")
            if os.path.exists(train_sampled_path):
                df_train = pd.read_csv(train_sampled_path)
                train_cols = [c for c in df_train.columns if c != 'isFraud']
                X_train_proc = preprocessor.transform(df_train)
                lime_explainer = setup_lime_explainer(
                    X_train_proc,
                    feature_names=preprocessor.selected_features
                )
            print("Successfully loaded model and preprocessor.")
        except Exception as e:
            print(f"Error loading model files: {e}. Running in placeholder mode.")
    else:
        print("Model assets not found. API is running in demo/placeholder mode.")

@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "models_loaded": (best_model is not None and preprocessor is not None),
        "message": "Fraud Detection API is running."
    }

@app.post("/predict")
def predict_fraud(payload: TransactionPayload):
    global preprocessor, best_model, lime_explainer
    
    # 1. Convert input to DataFrame
    input_dict = payload.dict()
    df_input = pd.DataFrame([input_dict])
    
    # 2. Check if we are running in placeholder/demo mode
    if best_model is None or preprocessor is None:
        # Placeholder mock logic
        prob = 0.05 if payload.TransactionAmt < 200 else 0.85
        score = risk_scorer.calculate_score(prob)
        risk_details = risk_scorer.get_category_details(score)
        
        mock_explanations = [
            ("TransactionAmt > 200.00", 0.45) if payload.TransactionAmt > 200 else ("TransactionAmt <= 200.00", -0.15),
            ("P_emaildomain == missing", 0.1),
            ("card1 == unknown", 0.05)
        ]
        
        return {
            "is_fraud_prediction": int(prob >= 0.5),
            "fraud_probability": prob,
            "risk_score": score,
            "risk_level": risk_details["level"],
            "action": risk_details["action"],
            "recommendation": risk_details["recommendation"],
            "explanation": mock_explanations,
            "mode": "demo_placeholder"
        }
        
    try:
        # Align columns to have the same schema as train set (fill missing with NaN)
        if len(train_cols) > 0:
            for col in train_cols:
                if col not in df_input.columns:
                    df_input[col] = np.nan
            df_input = df_input[train_cols]
            
        # 3. Transform input using fitted feature engineering pipeline
        X_proc = preprocessor.transform(df_input)
        
        # 4. Run model inference
        prob = float(best_model.predict_proba(X_proc)[0, 1])
        pred = int(best_model.predict(X_proc)[0])
        
        # 5. Score risk
        score = risk_scorer.calculate_score(prob)
        risk_details = risk_scorer.get_category_details(score)
        
        # 6. Generate local explanation
        explanations = []
        if lime_explainer is not None:
            # Predict probability function for LIME
            predict_fn = lambda x: best_model.predict_proba(x)
            explanations = get_lime_explanation_for_instance(
                lime_explainer,
                X_proc.iloc[0].values,
                predict_fn
            )
            
        return {
            "is_fraud_prediction": pred,
            "fraud_probability": prob,
            "risk_score": score,
            "risk_level": risk_details["level"],
            "action": risk_details["action"],
            "recommendation": risk_details["recommendation"],
            "explanation": explanations,
            "mode": "production"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
