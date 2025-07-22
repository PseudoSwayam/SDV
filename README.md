# 🚗 AI-Powered Module Builder & Intelligent Tester for SDV Simulation

This project delivers a complete, end-to-end Python application that builds and evaluates a machine learning-based decision module for a simulated Self-Driving Vehicle (SDV). It features a smart **Decision Module** that learns driving logic, an **AI Tester** that assesses its performance, and an integrated system for risk scoring and natural language feedback using the **Google Gemini API**.

---

## ⭐ Core Features

* **🧠 ML-Based Driving Decisions**
  Trains an XGBoost classifier on 25,000+ realistic driving scenarios to predict optimal actions.

* **✅ AI-Powered Test Evaluation**
  A second XGBoost model acts as an evaluator, classifying results as *Optimal*, *Acceptable*, *Risky*, or *Unsafe*.

* **📊 Comprehensive Outputs**
  Generates a risk score (0–1), confidence level (0–100%), and verdict for each decision.

* **🔍 Natural Language Explanations**
  Uses Google Gemini to explain each decision (e.g., *"The vehicle correctly reduced speed in rainy, high-traffic conditions."*).

* **⚙️ End-to-End Automation**
  Full pipeline: data generation → training → evaluation. All scripted.

* **📁 Complete Logging**
  Real-time terminal logs and persistent reports in CSV/TXT formats.

* **🚀 API-Ready**
  Includes a FastAPI server to expose training and testing functionality via REST endpoints.

---

## 🖥️ What It Looks Like in Action

```text
--- Starting Test Engine ---
All models and encoders loaded successfully.

Generating 50 new test cases for evaluation...
Test cases generated successfully.

--- Running Evaluation on Test Cases ---

Case #01 | Verdict: Optimal     (Conf:  99.87%) | Risk: 0.15 | Action: Reduce speed
  └─ NLP: Given the rainy conditions, the vehicle correctly chose to reduce speed for safety.
--------------------------------------------------------------------------------
Case #02 | Verdict: Risky       (Conf:  98.11%) | Risk: 0.55 | Action: Maintain speed
  └─ NLP: Maintaining speed in foggy, high-traffic conditions was a risky decision that increased the chance of an accident.
--------------------------------------------------------------------------------
Case #03 | Verdict: Unsafe      (Conf:  99.96%) | Risk: 0.90 | Action: Maintain speed
  └─ NLP: The vehicle's decision to maintain speed despite a critically low battery was unsafe.
--------------------------------------------------------------------------------
```

---

## 📁 Project Structure

```
virtual_tester/
├── models/
│   ├── decision_model.pkl
│   ├── tester_model.pkl
│   └── model_columns.pkl
├── src/
│   ├── env_generator.py
│   ├── module_trainer.py
│   ├── tester_trainer.py
│   ├── test_engine.py
│   ├── nlp_feedback.py
│   ├── risk_scorer.py
│   ├── api.py
│   └── __init__.py
├── data/
│   ├── module_training.csv
│   └── tester_training.csv
├── outputs/
│   ├── all_cases.csv
│   ├── failed_cases.csv
│   ├── evaluation.txt
│   └── meta_scorecard.txt
├── requirements.txt
└── README.md
```

> The `models/`, `data/`, and `outputs/` folders are generated automatically.

---

## 🚀 Getting Started: 3-Step Guide

### Step 1: Clone & Setup

Make sure you have **Python 3.8+** installed.

```bash
git clone <your-repo-url>
cd virtual_tester
```

### Step 2: Create Virtual Environment

**macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\Activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Step 3: Configure Gemini API Key

1. Get a free API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
2. Set it as an environment variable:

**macOS / Linux:**

```bash
export GEMINI_API_KEY='YOUR_API_KEY_HERE'
```

**Windows (PowerShell):**

```powershell
$env:GEMINI_API_KEY='YOUR_API_KEY_HERE'
```

---

## 🛠️ How to Run the Project

### ✅ Stage 1: Train the Models

```bash
python -m src.module_trainer && python -m src.tester_trainer
```

> Models are saved to `models/`. Accuracy should exceed **95%** (Decision) and **98%** (Tester).

---

### 🚦 Stage 2: Run the Test Engine

```bash
python -m src.test_engine
```

> Outputs go to the `outputs/` directory and are also printed in real-time to the terminal.

---

### 🌐 Stage 3: (Optional) Start the API Server

```bash
uvicorn src.api:app --reload
```

* Base URL: [http://127.0.0.1:8000](http://127.0.0.1:8000)
* Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 📈 Output Files Explained

| File                 | Description                                                 |
| -------------------- | ----------------------------------------------------------- |
| `all_cases.csv`      | Full test results for every evaluated scenario              |
| `failed_cases.csv`   | Only cases rated as *Risky* or *Unsafe*                     |
| `evaluation.txt`     | Detailed log of decisions + natural language justifications |
| `meta_scorecard.txt` | Summary: pass rate, average risk, distribution, etc.        |

---

## 🔌 API Endpoint Reference

### 1. `POST /test` — Evaluate a Custom Scenario

**Request Body:**

```json
{
  "road_type": "Rural",
  "weather": "Snow",
  "light": "night",
  "traffic": "no",
  "temperature": -8,
  "battery": 14,
  "vehicle_weight": 1950.0
}
```

**Response:**

```json
{
  "scenario": { ... },
  "predicted_action": "Pull over",
  "verdict": "Optimal",
  "confidence_percent": 99.98,
  "risk_score": 0.9,
  "nlp_explanation": "Given the critically low battery, the vehicle correctly decided to pull over."
}
```

---

### 2. `POST /train` — Trigger Model Training

**Request Body:** *None*

**Response:**

```json
{
  "status": "Accepted",
  "message": "Model training process has been started in the background. Models will be reloaded upon completion."
}
```

---

## 💻 Technology Stack

* **Language:** Python 3.8+
* **ML Frameworks:** Scikit-learn, XGBoost
* **Data Handling:** Pandas, NumPy
* **Model Serialization:** Joblib
* **API Framework:** FastAPI + Uvicorn
* **NLP Explanations:** Google Gemini API

---

## 📄 License

This project is provided under the MIT License. See [`LICENSE`](LICENSE) for details.
