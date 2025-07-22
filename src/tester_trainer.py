# src/tester_trainer.py
import pandas as pd
import joblib
import os
import random
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Configuration
DECISION_MODEL_FILE = "models/decision_model.pkl"
MODEL_DIR = "models"
TESTER_MODEL_FILE = os.path.join(MODEL_DIR, "tester_model.pkl")
TRAINING_DATA_FILE = "data/module_training.csv"
TESTER_DATA_FILE = "data/tester_training.csv"
TARGET_ACCURACY = 0.98

# Ensure directories exist
os.makedirs("data", exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# List of all possible actions for generating faulty data
ALL_POSSIBLE_ACTIONS = ['Maintain speed', 'Reduce speed', 'Turn on headlights', 'Pull over', 'Activate hazard lights']

def define_verdict(row):
    """Compares the module's predicted action to the ground truth to assign a verdict."""
    correct = row['correct_action']
    predicted = row['predicted_action']

    if predicted == correct:
        return "Optimal"

    # Define acceptable but not optimal scenarios
    if correct == 'Reduce speed' and predicted == 'Activate hazard lights':
        return "Acceptable"
    if correct == 'Pull over' and predicted == 'Activate hazard lights':
        return "Acceptable"

    # Define risky scenarios
    if correct == 'Reduce speed' and predicted == 'Maintain speed':
        return "Risky"
    if correct == 'Turn on headlights' and predicted != 'Turn on headlights' and row['light'] == 'night':
        return "Risky"

    # Unsafe is the most critical failure
    if correct == 'Pull over' and predicted == 'Maintain speed':
        return "Unsafe"
    if correct == 'Reduce speed' and predicted == 'Maintain speed' and row['weather'] == 'Snow':
        return "Unsafe"

    return "Risky" # Default for other mismatches

def train_tester_model():
    """
    Trains a second ML model to act as an intelligent tester. It learns to
    predict a verdict by comparing the decision module's output with the ground truth.
    """
    print("\n--- Training AI-Powered Tester Model ---")

    # 1. Load trained decision model and original data
    if not os.path.exists(DECISION_MODEL_FILE) or not os.path.exists(TRAINING_DATA_FILE):
        raise FileNotFoundError("Decision model or training data not found. Run 'module_trainer.py' first.")

    model_payload = joblib.load(DECISION_MODEL_FILE)
    decision_model = model_payload['model']
    action_encoder = model_payload['label_encoder']
    model_columns = joblib.load("models/model_columns.pkl")

    df = pd.read_csv(TRAINING_DATA_FILE)

    # 2. Generate predictions from the decision module
    X_for_prediction = df.drop('correct_action', axis=1)
    X_encoded = pd.get_dummies(X_for_prediction, columns=['road_type', 'weather', 'light', 'traffic'])
    X_encoded = X_encoded.reindex(columns=model_columns, fill_value=0) # Align columns

    predictions_encoded = decision_model.predict(X_encoded)
    df['predicted_action'] = action_encoder.inverse_transform(predictions_encoded)
    
    # --- FIX STARTS HERE ---
    # Artificially create incorrect predictions for the tester to learn from.
    # Since the base model is 100% accurate, we must manufacture "Risky" and "Unsafe" cases.
    print("Artificially generating failed test cases for tester training...")
    
    # Take a sample of the data (e.g., 50%) to corrupt
    incorrect_indices = df.sample(frac=0.5, random_state=42).index
    
    def generate_wrong_action(correct_action):
        # Pick a random action that is NOT the correct one
        wrong_actions = [a for a in ALL_POSSIBLE_ACTIONS if a != correct_action]
        return random.choice(wrong_actions)

    # For the selected rows, replace the "predicted_action" with a deliberately wrong one
    df.loc[incorrect_indices, 'predicted_action'] = df.loc[incorrect_indices, 'correct_action'].apply(generate_wrong_action)
    # --- FIX ENDS HERE ---

    # 3. Create the tester's training labels (verdicts)
    df['verdict'] = df.apply(define_verdict, axis=1)
    print("Verdict distribution for tester training:")
    print(df['verdict'].value_counts()) # This should now show all verdict types

    df.to_csv(TESTER_DATA_FILE, index=False)
    print(f"Tester training data with verdicts saved to '{TESTER_DATA_FILE}'")

    # 4. Prepare data for tester model
    features = df.drop(['correct_action', 'verdict'], axis=1)
    target = df['verdict']

    X_tester = pd.get_dummies(features, columns=['road_type', 'weather', 'light', 'traffic', 'predicted_action'])
    
    # Ensure verdict encoder learns all possible verdicts, even if some are rare
    verdict_encoder = LabelEncoder().fit(ALL_POSSIBLE_VERDICTS)
    y_tester = verdict_encoder.transform(target)

    # 5. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_tester, y_tester, test_size=0.2, random_state=42, stratify=y_tester
    )

    # 6. Model Training
    print("Training XGBoost classifier for the tester...")
    # Add num_class explicitly as a safeguard
    tester_model = XGBClassifier(
        objective='multi:softprob',
        num_class=len(verdict_encoder.classes_), # Explicitly tell XGBoost how many classes to expect
        eval_metric='mlogloss',
        random_state=42
    )
    tester_model.fit(X_train, y_train)

    # 7. Evaluation
    y_pred_tester = tester_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_tester)
    print(f"Tester Model Training Accuracy: {accuracy:.4f}")

    if accuracy < TARGET_ACCURACY:
        print(f"Warning: Tester accuracy ({accuracy:.4f}) is below the target of {TARGET_ACCURACY}.")
    else:
        print("Target accuracy for tester achieved.")

    # 8. Save Tester Model
    tester_payload = {
        'model': tester_model,
        'label_encoder': verdict_encoder,
        'model_columns': X_tester.columns
    }
    joblib.dump(tester_payload, TESTER_MODEL_FILE)
    print(f"Tester Model saved to '{TESTER_MODEL_FILE}'")

# Define all possible verdicts for robust LabelEncoding
ALL_POSSIBLE_VERDICTS = ["Optimal", "Acceptable", "Risky", "Unsafe"]

if __name__ == '__main__':
    train_tester_model()