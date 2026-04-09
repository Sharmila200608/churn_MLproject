"""
model.py
--------
Defines ML models, risk classification logic,
and the business recommendation engine.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import numpy as np


# ─── Model factory ────────────────────────────────────────────────────────────

def get_random_forest():
    """Return a configured Random Forest classifier."""
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        class_weight="balanced",   # handles class imbalance
        n_jobs=-1,
    )


def get_logistic_regression():
    """Return a configured Logistic Regression classifier."""
    return LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight="balanced",
        solver="lbfgs",
    )


# ─── Risk classification ──────────────────────────────────────────────────────

def classify_risk(probability: float) -> dict:
    """
    Map a churn probability to a risk level with label, emoji, and color.

    Parameters
    ----------
    probability : float between 0 and 1.

    Returns
    -------
    dict with keys: level, emoji, color, hex
    """
    pct = probability * 100

    if pct < 40:
        return {
            "level":  "Low Risk",
            "emoji":  "🟢",
            "color":  "green",
            "hex":    "#22c55e",
            "badge":  "success",
        }
    elif pct < 70:
        return {
            "level":  "Medium Risk",
            "emoji":  "🟡",
            "color":  "orange",
            "hex":    "#f59e0b",
            "badge":  "warning",
        }
    else:
        return {
            "level":  "High Risk",
            "emoji":  "🔴",
            "color":  "red",
            "hex":    "#ef4444",
            "badge":  "danger",
        }


# ─── Recommendation engine ────────────────────────────────────────────────────

RECOMMENDATIONS = {
    "Low Risk": [
        ("✅", "No immediate action needed"),
        ("🎁", "Consider loyalty reward program"),
        ("📊", "Monitor quarterly — stay engaged"),
    ],
    "Medium Risk": [
        ("📧", "Launch a personalised email campaign"),
        ("💬", "Offer live chat or callback support"),
        ("🎯", "Highlight unused features the customer hasn't activated"),
        ("📅", "Invite to a product demo or webinar"),
    ],
    "High Risk": [
        ("💰", "Offer an exclusive discount or price lock"),
        ("📞", "Schedule a proactive retention call"),
        ("🔄", "Propose a contract upgrade with added benefits"),
        ("🛡️", "Assign a dedicated customer success manager"),
        ("⚡", "Escalate to retention team immediately"),
    ],
}


def get_recommendations(risk_level: str) -> list:
    """Return a list of (icon, text) recommendation tuples for the given risk level."""
    return RECOMMENDATIONS.get(risk_level, [])
