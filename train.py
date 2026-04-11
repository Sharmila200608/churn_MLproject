"""
train.py
--------
Downloads the Telco Customer Churn dataset (or generates synthetic data),
trains both ML models, evaluates them, and saves the best one to disk.

Run this FIRST before launching the Streamlit app:
    python train.py
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, roc_auc_score
)

from preprocessing import preprocess_data
from model import get_random_forest, get_logistic_regression


# ─── Config ───────────────────────────────────────────────────────────────────

MODEL_PATH       = "churn_model.pkl"
FEATURE_PATH     = "feature_names.pkl"
DATA_PATH        = "telco_churn.csv"       # put your CSV here, or we generate data
TEST_SIZE        = 0.2
RANDOM_STATE     = 42


# ─── Synthetic data generator (fallback if no CSV) ───────────────────────────

def generate_synthetic_data(n=2000):
    """
    Generate a realistic synthetic Telco-style churn dataset so the project
    runs out-of-the-box even without a real CSV file.
    """
    print("📦  No CSV found — generating synthetic dataset...")
    np.random.seed(42)
    n = 2000

    tenure            = np.random.randint(0, 72, n)
    monthly_charges   = np.round(np.random.uniform(18, 120, n), 2)
    total_charges     = np.round(tenure * monthly_charges + np.random.normal(0, 50, n), 2)
    total_charges     = np.clip(total_charges, 0, None)

    contract          = np.random.choice(["Month-to-month", "One year", "Two year"], n,
                                          p=[0.55, 0.25, 0.20])
    internet_service  = np.random.choice(["DSL", "Fiber optic", "No"], n,
                                          p=[0.35, 0.45, 0.20])
    payment_method    = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)",
         "Credit card (automatic)"], n, p=[0.34, 0.23, 0.22, 0.21])

    # Churn probability is influenced by real-world factors
    churn_prob = (
        0.4 * (contract == "Month-to-month").astype(float)
        + 0.2 * (internet_service == "Fiber optic").astype(float)
        + 0.15 * (monthly_charges > 70).astype(float)
        - 0.3 * (tenure > 36).astype(float)
        + np.random.normal(0, 0.1, n)
    )
    churn = (churn_prob > 0.2).astype(int)

    yes_no = lambda arr: np.where(arr, "Yes", "No")

    df = pd.DataFrame({
        "customerID":       [f"CUST-{i:04d}" for i in range(n)],
        "gender":           np.random.choice(["Male", "Female"], n),
        "SeniorCitizen":    np.random.choice([0, 1], n, p=[0.84, 0.16]),
        "Partner":          yes_no(np.random.rand(n) > 0.5),
        "Dependents":       yes_no(np.random.rand(n) > 0.7),
        "tenure":           tenure,
        "PhoneService":     yes_no(np.random.rand(n) > 0.1),
        "MultipleLines":    np.random.choice(["Yes", "No", "No phone service"], n),
        "InternetService":  internet_service,
        "OnlineSecurity":   np.random.choice(["Yes", "No", "No internet service"], n),
        "OnlineBackup":     np.random.choice(["Yes", "No", "No internet service"], n),
        "DeviceProtection": np.random.choice(["Yes", "No", "No internet service"], n),
        "TechSupport":      np.random.choice(["Yes", "No", "No internet service"], n),
        "StreamingTV":      np.random.choice(["Yes", "No", "No internet service"], n),
        "StreamingMovies":  np.random.choice(["Yes", "No", "No internet service"], n),
        "Contract":         contract,
        "PaperlessBilling": yes_no(np.random.rand(n) > 0.4),
        "PaymentMethod":    payment_method,
        "MonthlyCharges":   monthly_charges,
        "TotalCharges":     total_charges.astype(str),
        "Churn":            np.where(churn == 1, "Yes", "No"),
    })
    return df


# ─── Main training pipeline ───────────────────────────────────────────────────

def train():
    # 1. Load or generate data
    if os.path.exists(DATA_PATH):
        print(f"✅  Loading data from {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
    else:
        df = generate_synthetic_data()

    print(f"   Rows: {len(df)} | Churn rate: {df['Churn'].value_counts(normalize=True)['Yes']:.1%}")

    # 2. Preprocess
    X, y, feature_names = preprocess_data(df, training=True)
    print(f"   Features: {len(feature_names)}")

    # 3. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # 4. Train both models and pick the better one
    results = {}
    for name, clf in [("Random Forest", get_random_forest()),
                      ("Logistic Regression", get_logistic_regression())]:
        print(f"\n🔧  Training {name}...")
        clf.fit(X_train, y_train)
        preds     = clf.predict(X_test)
        probs     = clf.predict_proba(X_test)[:, 1]
        acc       = accuracy_score(y_test, preds)
        auc       = roc_auc_score(y_test, probs)
        results[name] = {"model": clf, "acc": acc, "auc": auc}
        print(f"   Accuracy: {acc:.4f}  |  AUC-ROC: {auc:.4f}")
        print(classification_report(y_test, preds, target_names=["No Churn", "Churn"]))

    # 5. Save the better model (by AUC)
    best_name  = max(results, key=lambda k: results[k]["auc"])
    best_model = results[best_name]["model"]
    print(f"\n🏆  Best model: {best_name}  (AUC={results[best_name]['auc']:.4f})")

    joblib.dump(best_model,    MODEL_PATH)
    joblib.dump(feature_names, FEATURE_PATH)
    print(f"💾  Saved to:  {MODEL_PATH}  &  {FEATURE_PATH}")
    print("\n✅  Training complete! Run:  streamlit run app.py")


if __name__ == "__main__":
    train()
