# src/env_generator.py
import pandas as pd
import numpy as np
import os

# Configuration
NUM_ROWS = 25000
OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "module_training.csv")

# Ensure directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def define_ground_truth_action(row):
    """Defines the optimal action based on a set of rules."""
    # Critical safety overrides
    if row['weather'] == 'Snow' and row['temperature'] < 0:
        return 'Reduce speed'
    if row['battery'] < 15:
        return 'Pull over'
    if row['light'] == 'night' and row['traffic'] == 'high':
        return 'Reduce speed'

    # Secondary actions
    if row['light'] == 'night':
        return 'Turn on headlights'
    if row['weather'] == 'Rainy' and row['traffic'] in ['high', 'moderate']:
        return 'Reduce speed'
    if row['battery'] < 30:
        return 'Activate hazard lights'

    # Default action
    return 'Maintain speed'

def generate_training_data():
    """
    Generates a realistic dataset for SDV decision module training.
    Each row includes environmental factors and a rule-based 'correct_action'.
    """
    print(f"Generating {NUM_ROWS} training scenarios...")

    data = {
        'road_type': np.random.choice(['Highway', 'Rural', 'Semi Urban'], NUM_ROWS),
        'weather': np.random.choice(['Sunny', 'Rainy', 'Snow', 'Foggy'], NUM_ROWS),
        'light': np.random.choice(['day', 'night'], NUM_ROWS),
        'traffic': np.random.choice(['moderate', 'high', 'no'], NUM_ROWS),
        'temperature': np.random.randint(-10, 40, size=NUM_ROWS),
        'battery': np.random.randint(10, 101, size=NUM_ROWS),
        'vehicle_weight': np.random.uniform(1500, 2500, size=NUM_ROWS).round(2),
    }
    df = pd.DataFrame(data)

    # Apply logical constraints
    # No snow in high temperatures
    df.loc[df['temperature'] > 5, 'weather'] = np.random.choice(['Sunny', 'Rainy', 'Foggy'], size=df[df['temperature'] > 5].shape[0])

    # Assign the ground truth action based on rules
    df['correct_action'] = df.apply(define_ground_truth_action, axis=1)

    # Save to CSV
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Successfully generated and saved training data to '{OUTPUT_FILE}'")
    return df

if __name__ == '__main__':
    generate_training_data()