"""
preprocessing.py
----------------
Handles all data cleaning, encoding, and feature engineering
for the Customer Churn Intelligence System.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


# ─── Column definitions ───────────────────────────────────────────────────────

CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]

NUMERICAL_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

TARGET_COL = "Churn"


# ─── Main preprocessing function ─────────────────────────────────────────────

def preprocess_data(df: pd.DataFrame, training: bool = True):
    """
    Clean and encode the raw Telco churn dataframe.

    Parameters
    ----------
    df       : Raw dataframe (from CSV).
    training : If True, also return the target column y.

    Returns
    -------
    X        : Feature matrix (pd.DataFrame).
    y        : Target series (only when training=True).
    feature_names : List of column names in X.
    """
    df = df.copy()

    # 1. Drop customer ID – not a predictive feature
    if "customerID" in df.columns:
        df.drop(columns=["customerID"], inplace=True)

    # 2. Fix TotalCharges: it's stored as a string in the raw CSV
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # 3. Fill NaN TotalCharges with median (new customers with 0 tenure)
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # 4. Encode target column (Yes → 1, No → 0)
    if training and TARGET_COL in df.columns:
        df[TARGET_COL] = df[TARGET_COL].map({"Yes": 1, "No": 0})
        y = df.pop(TARGET_COL)
    else:
        y = None

    # 5. Label-encode all categorical columns
    le = LabelEncoder()
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))

    # 6. Scale numerical columns
    scaler = StandardScaler()
    existing_num = [c for c in NUMERICAL_COLS if c in df.columns]
    df[existing_num] = scaler.fit_transform(df[existing_num])

    feature_names = list(df.columns)

    if training:
        return df, y, feature_names
    return df, feature_names


def preprocess_single_input(user_input: dict):
    """
    Convert a single-row dict (from the Streamlit form) into
    a model-ready dataframe row.
    """
    df = pd.DataFrame([user_input])
    X, _ = preprocess_data(df, training=False)
    return X
