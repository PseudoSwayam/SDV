# src/test_engine.py
import pandas as pd
import numpy as np
import joblib
import os
import time
from typing import List, Dict

from src.risk_scorer import calculate_risk_score
from src.nlp_feedback import get_nlp_feedback

# --- Configuration ---
DECISION_MODEL_FILE = "models/decision_model.pkl"
TESTER_MODEL_FILE = "models/tester_model.pkl"
MODEL_COLUMNS_FILE = "models/model_columns.pkl"
NUM_TEST_CASES = 50
OUTPUT_DIR = "outputs"

# File paths for all outputs
ALL_CASES_CSV = os.path.join(OUTPUT_DIR, "all_cases.csv")
FAILED_CASES_CSV = os.path.join(OUTPUT_DIR, "failed_cases.csv")
EVALUATION_TXT = os.path.join(OUTPUT_DIR, "evaluation.txt")
META_SCORECARD_TXT = os.path.join(OUTPUT_DIR, "meta_scorecard.txt")

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_test_cases(n_cases: int) -> pd.DataFrame:
    """Generates a new, random set of test cases for evaluation."""
    print(f"\nGenerating {n_cases} new test cases for evaluation...")
    data = {
        'road_type': np.random.choice(['Highway', 'Rural', 'Semi Urban'], n_cases),
        'weather': np.random.choice(['Sunny', 'Rainy', 'Snow', 'Foggy'], n_cases),
        'light': np.random.choice(['day', 'night'], n_cases),
        'traffic': np.random.choice(['moderate', 'high', 'no'], n_cases),
        'temperature': np.random.randint(-10, 40, size=n_cases),
        'battery': np.random.randint(10, 101, size=n_cases),
        'vehicle_weight': np.random.uniform(1500, 2500, size=n_cases).round(2),
    }
    df = pd.DataFrame(data)
    
    # Apply logical constraints to make data more realistic
    df.loc[df['temperature'] > 5, 'weather'] = np.random.choice(['Sunny', 'Rainy', 'Foggy'], size=df[df['temperature'] > 5].shape[0])
    print("Test cases generated successfully.")
    return df

def run_test_engine():
    """
    Main engine that runs the full test and evaluation pipeline. It loads models,
    simulates scenarios, gets verdicts, and logs everything.
    """
    start_time = time.time()
    print("--- Starting Test Engine ---")

    # 1. Load All Models and Required Artifacts
    try:
        decision_payload = joblib.load(DECISION_MODEL_FILE)
        tester_payload = joblib.load(TESTER_MODEL_FILE)
        model_cols = joblib.load(MODEL_COLUMNS_FILE)

        decision_model = decision_payload['model']
        action_encoder = decision_payload['label_encoder']
        tester_model = tester_payload['model']
        verdict_encoder = tester_payload['label_encoder']
        tester_cols = tester_payload['model_columns']
        print("All models and encoders loaded successfully.")
    except FileNotFoundError as e:
        print(f"\n[ERROR] Model file not found: {e}")
        print("Please run the training scripts first:")
        print("  - python -m src.module_trainer")
        print("  - python -m src.tester_trainer")
        return

    # 2. Generate new test scenarios
    test_scenarios = generate_test_cases(NUM_TEST_CASES)
    results: List[Dict] = []

    print("\n--- Running Evaluation on Test Cases ---\n")

    # 3. Process each scenario one by one
    for index, scenario in test_scenarios.iterrows():
        scenario_dict = scenario.to_dict()
        scenario_df = pd.DataFrame([scenario_dict])

        # --- Step A: Get Action from Decision Module ---
        X_decision = pd.get_dummies(scenario_df).reindex(columns=model_cols, fill_value=0)
        action_encoded = decision_model.predict(X_decision)[0]
        predicted_action = action_encoder.inverse_transform([action_encoded])[0]

        # --- Step B: Get Verdict & Confidence from Tester Module ---
        scenario_with_action = scenario_df.copy()
        scenario_with_action['predicted_action'] = predicted_action
        
        X_tester = pd.get_dummies(scenario_with_action).reindex(columns=tester_cols, fill_value=0)
        verdict_probs = tester_model.predict_proba(X_tester)[0]
        verdict_encoded = np.argmax(verdict_probs)
        confidence = verdict_probs[verdict_encoded] * 100
        verdict = verdict_encoder.inverse_transform([verdict_encoded])[0]

        # --- Step C: Calculate Rule-Based Risk Score ---
        risk_score = calculate_risk_score(scenario_dict, predicted_action)

        # --- Step D: Get Natural Language Feedback ---
        nlp_explanation = get_nlp_feedback(scenario_dict, predicted_action, verdict)
        
        # --- Terminal Log for real-time feedback ---
        print(f"Case #{index + 1:02d} | Verdict: {verdict:<10} (Conf: {confidence:6.2f}%) | Risk: {risk_score:.2f} | Action: {predicted_action:<22}")
        print(f"  └─ NLP: {nlp_explanation}")
        print("-" * 80)

        # --- Store result for final report generation ---
        results.append({
            **scenario_dict,
            'predicted_action': predicted_action,
            'verdict': verdict,
            'confidence_percent': round(confidence, 2),
            'risk_score': round(risk_score, 2),
            'nlp_explanation': nlp_explanation
        })

    # 4. Compile and Save All Output Files
    results_df = pd.DataFrame(results)
    
    # Save all cases to CSV
    results_df.to_csv(ALL_CASES_CSV, index=False)
    print(f"\n✓ Saved all {len(results_df)} test case results to '{ALL_CASES_CSV}'")

    # Save only 'Risky' or 'Unsafe' cases
    failed_cases_df = results_df[results_df['verdict'].isin(['Risky', 'Unsafe'])]
    failed_cases_df.to_csv(FAILED_CASES_CSV, index=False)
    print(f"✓ Saved {len(failed_cases_df)} failed cases to '{FAILED_CASES_CSV}'")

    # Save detailed human-readable text log
    with open(EVALUATION_TXT, 'w') as f:
        f.write("--- DETAILED TEST EVALUATION ---\n\n")
        for i, res in enumerate(results):
            f.write(f"--- Case #{i+1} ---\n")
            scenario_str = ", ".join([f"{k}: {v}" for k, v in res.items() if k not in ['predicted_action', 'verdict', 'confidence_percent', 'risk_score', 'nlp_explanation']])
            f.write(f"Scenario: {scenario_str}\n")
            f.write(f"  - Predicted Action: {res['predicted_action']}\n")
            f.write(f"  - Verdict: {res['verdict']} (Confidence: {res['confidence_percent']}%)\n")
            f.write(f"  - Risk Score: {res['risk_score']}\n")
            f.write(f"  - Explanation: {res['nlp_explanation']}\n\n")
    print(f"✓ Saved detailed text log to '{EVALUATION_TXT}'")
    
    # Calculate and save meta scorecard
    total_cases = len(results_df)
    verdict_counts = results_df['verdict'].value_counts()
    avg_risk = results_df['risk_score'].mean()
    end_time = time.time()

    with open(META_SCORECARD_TXT, 'w') as f:
        f.write("--- META SCORECARD ---\n")
        f.write(f"Test Execution Time: {end_time - start_time:.2f} seconds\n")
        f.write(f"Total Test Cases Run: {total_cases}\n")
        f.write("-" * 25 + "\n")
        f.write("Verdict Distribution:\n")
        for verdict, count in verdict_counts.items():
            f.write(f"  - {verdict:<12}: {count} ({count/total_cases:.2%})\n")
        f.write("-" * 25 + "\n")
        f.write(f"Average Risk Score: {avg_risk:.3f}\n")
    print(f"✓ Saved meta scorecard to '{META_SCORECARD_TXT}'")
    
    print("\n--- Test Engine Finished ---")

if __name__ == '__main__':
    run_test_engine()