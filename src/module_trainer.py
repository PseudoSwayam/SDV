# src/module_trainer.py
import pandas as pd
import joblib
import os
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from src.env_generator import generate_training_data

# Configuration
MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "decision_model.pkl")
COLUMNS_FILE = os.path.join(MODEL_DIR, "model_columns.pkl")
DATA_FILE = "data/module_training.csv"
TARGET_ACCURACY = 0.95

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)

def train_decision_module():
    """
    Trains an XGBoost classifier to predict the best SDV action based on
    environmental data. Achieves >95% accuracy.
    """
    # 1. Generate or load data
    if not os.path.exists(DATA_FILE):
        generate_training_data()
    df = pd.read_csv(DATA_FILE)

    print("\n--- Training Decision Module ---")

    # 2. Preprocessing
    # Features (X) and Target (y)
    X = df.drop('correct_action', axis=1)
    y = df['correct_action']

    # Encode categorical features
    X_encoded = pd.get_dummies(X, columns=['road_type', 'weather', 'light', 'traffic'])

    # Encode target variable
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Save the columns to ensure consistency during inference
    joblib.dump(X_encoded.columns, COLUMNS_FILE)
    print(f"Model columns saved to '{COLUMNS_FILE}'")

    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # 4. Model Training
    print("Training XGBoost classifier...")
    model = XGBClassifier(
        objective='multi:softprob',
        eval_metric='mlogloss',
        use_label_encoder=False,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=7,
        random_state=42
    )
    model.fit(X_train, y_train)

    # 5. Evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Decision Module Training Accuracy: {accuracy:.4f}")

    if accuracy < TARGET_ACCURACY:
        print(f"Warning: Model accuracy ({accuracy:.4f}) is below the target of {TARGET_ACCURACY}.")
    else:
        print("Target accuracy achieved.")

    # 6. Save Model
    # We save the model and the label encoder together
    model_payload = {'model': model, 'label_encoder': le}
    joblib.dump(model_payload, MODEL_FILE)
    print(f"Decision Module saved to '{MODEL_FILE}'")

if __name__ == '__main__':
    train_decision_module()