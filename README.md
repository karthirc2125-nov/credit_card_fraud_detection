# Credit Card Fraud Detection — Complete Beginner Project

## Features
- `app.py`: simple Streamlit UI with only transaction amount and bank-style message.
- Shows a transparent **rule-based demo risk indicator** and the warning phrases/amount thresholds that triggered it.
- `train_model.py`: separately cleans `creditcard.csv`, saves basic EDA outputs, trains Logistic Regression and Random Forest, compares average precision, and saves the best numerical model.

## Important limitation
The classic `creditcard.csv` dataset contains `Time`, `Amount`, `V1` through `V28`, and `Class`. It has **no transaction message/reason text**. Therefore, a model trained on it cannot legitimately predict fraud from only an amount and message.

The app's message screening is a transparent demonstration, **not a trained fraud classifier, calibrated probability, or real bank security system**. For genuine message-based ML, you need labeled examples that include transaction messages/descriptions and fraud labels.

## Project structure
```text
credit_card_fraud_detection_complete/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
└── artifacts/  # created after model training
```

## 1. Open in VS Code
Extract the ZIP and open the extracted folder in VS Code.

## 2. Install dependencies (PowerShell)
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
If activation is blocked:
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 3. Run the frontend
```powershell
streamlit run app.py
```
Open the Local URL printed in the terminal, commonly `http://localhost:8501` or `http://127.0.0.1:8501`. The frontend does not require CSV upload.

## 4. Train the numerical ML backend (optional separate step)
Download:
https://zenodo.org/records/7395559/files/creditcard.csv?download=1

Save it as `creditcard.csv` in the same folder as `train_model.py`, then run:
```powershell
python train_model.py
```
Training can take several minutes. It creates `artifacts/` containing `eda_summary.json`, `class_counts.csv`, `model_metrics.json`, `feature_names.json`, and `numerical_fraud_model.joblib`.

The numerical model expects the full set of dataset features and is not connected to the amount/message UI because the dataset has no message field.

## Safety
Never enter real card numbers, CVV, PIN, passwords, or OTPs. A message and amount alone cannot reliably determine whether a payment is fraudulent. Verify real concerns using your bank's official app or the number printed on your card.
