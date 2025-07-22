# src/nlp_feedback.py
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from a .env file if it exists
load_dotenv()

def get_nlp_feedback(scenario: dict, action: str, verdict: str) -> str:
    """
    Generates a one-line natural language explanation for a test verdict
    using the Google Gemini API.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "NLP feedback unavailable: GEMINI_API_KEY not set."

    try:
        genai.configure(api_key=api_key)
        # --- THIS IS THE FIX ---
        # Changed 'gemini-pro' to the stable 'gemini-1.0-pro' model name
        model = genai.GenerativeModel('gemini-2.5-pro')
        # ---------------------
    except Exception as e:
        return f"NLP feedback unavailable: Could not configure API. Error: {e}"

    # The rest of the prompt is the same
    scenario_str = ", ".join([f"{k.replace('_', ' ')} is {v}" for k, v in scenario.items()])

    prompt = f"""
    Analyze the following self-driving vehicle (SDV) test case and provide a concise, one-sentence explanation for the final report.

    **Instructions:**
    1.  The explanation must be a single sentence.
    2.  Focus on the *reason* for the verdict.
    3.  If the verdict is "Optimal", provide a simple confirmation.
    4.  Do not use technical jargon.

    **Test Case Details:**
    - **Scenario:** {scenario_str}
    - **SDV Action Taken:** "{action}"
    - **Verdict:** "{verdict}"

    **Your one-sentence explanation:**
    """

    try:
        response = model.generate_content(prompt)
        feedback = response.text.strip().replace("*", "")
        return feedback
    except Exception as e:
        return f"NLP feedback generation failed. Error: {e}"