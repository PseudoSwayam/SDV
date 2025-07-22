# src/risk_scorer.py

def calculate_risk_score(scenario: dict, predicted_action: str) -> float:
    """
    Calculates a risk score from 0.0 to 1.0 based on a set of rules.

    Args:
        scenario (dict): A dictionary representing the driving environment.
        predicted_action (str): The action predicted by the decision module.

    Returns:
        float: A risk score.
    """
    score = 0.0
    
    # Rule 1: Environmental Danger
    if scenario.get('weather') == 'Snow':
        score += 0.3
    if scenario.get('weather') == 'Rainy':
        score += 0.15
    if scenario.get('light') == 'night':
        score += 0.15
    if scenario.get('traffic') == 'high':
        score += 0.2

    # Rule 2: Low Battery is a critical risk factor
    if scenario.get('battery', 100) < 20:
        score += 0.5
        if predicted_action not in ['Pull over', 'Activate hazard lights']:
            score += 0.4 # Penalize heavily for ignoring low battery

    # Rule 3: Visibility and Action Mismatch
    if scenario.get('light') == 'night' and predicted_action != 'Turn on headlights':
        # If it's not the primary action, it's still a risk
        score += 0.25
        
    # Rule 4: Inappropriate speed for conditions
    if predicted_action == 'Maintain speed':
        if scenario.get('weather') in ['Snow', 'Rainy', 'Foggy']:
            score += 0.4
        if scenario.get('traffic') == 'high' and scenario.get('light') == 'night':
            score += 0.35

    # Ensure score is capped at 1.0
    return min(score, 1.0)