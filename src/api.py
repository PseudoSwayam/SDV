# src/api.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import numpy as np
import subprocess
import os

from src.risk_scorer import calculate_risk_score
from src.nlp_feedback import get_nlp_feedback

app = FastAPI(
    title="AI-Powered SDV Tester API",
    description="An API to evaluate SDV driving decisions using a machine learning model. \
                 Use the /test endpoint to evaluate a scenario or /train to retrain the models.",
    version="1.0.0"
)

# --- State variable to check training status ---
is_training = False

# --- Pydantic Models for API Data Validation ---
class ScenarioPayload(BaseModel):
    road_type: str = Field(..., example='Highway', description="Type of road: 'Highway', 'Rural', 'Semi Urban'")
    weather: str = Field(..., example='Snow', description="Current weather: 'Sunny', 'Rainy', 'Snow', 'Foggy'")
    light: str = Field(..., example='night', description="Light conditions: 'day', 'night'")
    traffic: str = Field(..., example='moderate', description="Traffic density: 'high', 'moderate', 'no'")
    temperature: int = Field(..., example=-5, description="Ambient temperature in Celsius")
    battery: int = Field(..., example=18, description="Vehicle battery percentage (0-100)")
    vehicle_weight: float = Field(..., example=2100.7, description="Total vehicle weight in kg")

class TestResponse(BaseModel):
    scenario: ScenarioPayload
    predicted_action: str
    verdict: str
    confidence_percent: float
    risk_score: float
    nlp_explanation: str

class TrainResponse(BaseModel):
    status: str
    message: str


# --- Helper Function to Load Models ---
def load_models():
    """Loads all necessary models and artifacts. Returns a dict."""
    try:
        decision_payload = joblib.load("models/decision_model.pkl")
        tester_payload = joblib.load("models/tester_model.pkl")
        model_cols = joblib.load("models/model_columns.pkl")
        
        return {
            "decision_model": decision_payload['model'],
            "action_encoder": decision_payload['label_encoder'],
            "tester_model": tester_payload['model'],
            "verdict_encoder": tester_payload['label_encoder'],
            "model_cols": model_cols,
            "tester_cols": tester_payload['model_columns']
        }
    except FileNotFoundError:
        return None

models = load_models()

# --- API Endpoints ---
@app.get("/", include_in_schema=False)
def root():
    return {"message": "Welcome to the SDV Tester API. Navigate to /docs for interactive documentation."}

@app.post("/test", response_model=TestResponse, tags=["Evaluation"])
def test_single_scenario(scenario: ScenarioPayload):
    """
    Accepts a driving scenario and returns a complete AI-powered evaluation.
    """
    if models is None:
        raise HTTPException(
            status_code=503, 
            detail="Models not loaded. Please trigger model training via the /train endpoint and try again later."
        )

    scenario_dict = scenario.model_dump()
    scenario_df = pd.DataFrame([scenario_dict])

    # Get Decision
    X_decision = pd.get_dummies(scenario_df).reindex(columns=models['model_cols'], fill_value=0)
    action_encoded = models['decision_model'].predict(X_decision)[0]
    predicted_action = models['action_encoder'].inverse_transform([action_encoded])[0]

    # Get Verdict
    scenario_with_action = scenario_df.copy()
    scenario_with_action['predicted_action'] = predicted_action
    X_tester = pd.get_dummies(scenario_with_action).reindex(columns=models['tester_cols'], fill_value=0)
    
    verdict_probs = models['tester_model'].predict_proba(X_tester)[0]
    verdict_encoded = np.argmax(verdict_probs)
    confidence = float(verdict_probs[verdict_encoded] * 100)
    verdict = models['verdict_encoder'].inverse_transform([verdict_encoded])[0]

    # Get Score and NLP Feedback
    risk_score = calculate_risk_score(scenario_dict, predicted_action)
    nlp_explanation = get_nlp_feedback(scenario_dict, predicted_action, verdict)

    return TestResponse(
        scenario=scenario,
        predicted_action=predicted_action,
        verdict=verdict,
        confidence_percent=round(confidence, 2),
        risk_score=round(risk_score, 2),
        nlp_explanation=nlp_explanation,
    )

def run_training_pipeline():
    """Target function for the background task to run training scripts."""
    global is_training, models
    is_training = True
    print("Background training started...")
    try:
        # We call the scripts as separate processes to ensure clean state
        subprocess.run(["python", "-m", "src.module_trainer"], check=True, capture_output=True, text=True)
        subprocess.run(["python", "-m", "src.tester_trainer"], check=True, capture_output=True, text=True)
        print("Training scripts completed successfully.")
        # Reload models after training
        models = load_models()
        if models:
            print("Models reloaded successfully.")
        else:
            print("Failed to reload models after training.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred during training background task: {e.stderr}")
    finally:
        is_training = False

@app.post("/train", response_model=TrainResponse, status_code=202, tags=["Training"])
def trigger_training(background_tasks: BackgroundTasks):
    """
    Triggers the full model training pipeline as a background task.
    Returns immediately with a confirmation.
    """
    global is_training
    if is_training:
        raise HTTPException(status_code=409, detail="A training process is already in progress.")
    
    background_tasks.add_task(run_training_pipeline)
    
    return TrainResponse(
        status="Accepted",
        message="Model training process has been started in the background. Models will be reloaded upon completion."
    )