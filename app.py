"""
app.py
------
Streamlit dashboard for the AI Customer Churn Intelligence System.

Launch with:  streamlit run app.py
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.graph_objects as go

from preprocessing import preprocess_single_input
from model import classify_risk, get_recommendations


# ─── Page config (MUST be first Streamlit call) ───────────────────────────────

st.set_page_config(
    page_title="Churn Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* Sidebar style */
[data-testid="stSidebar"] { background: #0f172a; }
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    color: white;
}
.metric-card h2 { font-size: 2rem; margin: 0; }
.metric-card p  { color: #94a3b8; margin: 4px 0 0; font-size: 0.85rem; }

/* Section headers */
.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #6366f1;
    border-left: 4px solid #6366f1;
    padding-left: 10px;
    margin: 24px 0 12px;
}

/* Recommendation card */
.rec-card {
    background: #1e293b;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 6px 0;
    border-left: 3px solid #6366f1;
    color: #e2e8f0;
    font-size: 0.95rem;
}

/* Progress bar override */
.stProgress > div > div > div { border-radius: 6px; }
</style>
""", unsafe_allow_html=True)


# ─── Load model ───────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    try:
        model         = joblib.load("churn_model.pkl")
        feature_names = joblib.load("feature_names.pkl")
        return model, feature_names
    except FileNotFoundError:
        return None, None


model, feature_names = load_model()


# ─── Sidebar navigation ───────────────────────────────────────────────────────

st.sidebar.title("🧠 Churn Intelligence")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["🏠  Home", "🔍  Predict", "📊  Insights"],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")
st.sidebar.caption("Customer Churn Prediction System")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ═══════════════════════════════════════════════════════════════════════════════

if page == "🏠  Home":
    st.title("🧠 Customer Churn Prediction System")
    st.markdown("Predict churn, classify risk levels, and get smart retention recommendations.")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""<div class="metric-card">
            <h2>🎯</h2><h2 style="font-size:1.4rem">Churn Prediction</h2>
            <p>Yes / No + probability</p></div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("""<div class="metric-card">
            <h2>⚠️</h2><h2 style="font-size:1.4rem">Risk Detection</h2>
            <p>Low · Medium · High</p></div>""", unsafe_allow_html=True)

    with col3:
        st.markdown("""<div class="metric-card">
            <h2>💡</h2><h2 style="font-size:1.4rem">Recommendations</h2>
            <p>Actionable retention strategies</p></div>""", unsafe_allow_html=True)

    with col4:
        st.markdown("""<div class="metric-card">
            <h2>📈</h2><h2 style="font-size:1.4rem">Feature Insights</h2>
            <p>Why customers churn</p></div>""", unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("🚀 How to use")
    st.markdown("""
    1. **Predict tab** → fill in customer details → get instant churn prediction
    2. **Insights tab** → explore which features drive churn the most
    3. Use the risk level and recommendations to take action
    """)

    if model is None:
        st.error("⚠️ Model not found! Run `python train.py` first, then refresh this page.")
    else:
        st.success("✅ Model loaded and ready!")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PREDICT
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🔍  Predict":
    st.title("🔍 Customer Churn Predictor")
    st.markdown("Fill in the customer details below and click **Predict**.")

    if model is None:
        st.error("⚠️ Model not found! Run `python train.py` first.")
        st.stop()

    # ── Input form ────────────────────────────────────────────────────────────
    with st.form("predict_form"):
        st.markdown('<div class="section-header">👤 Customer Profile</div>',
                    unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        with col1:
            gender         = st.selectbox("Gender",        ["Male", "Female"])
            senior_citizen = st.selectbox("Senior Citizen",["No", "Yes"])
            partner        = st.selectbox("Partner",       ["Yes", "No"])
            dependents     = st.selectbox("Dependents",    ["Yes", "No"])

        with col2:
            tenure          = st.slider("Tenure (months)", 0, 72, 12)
            monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 50.0, 0.5)
            total_charges   = st.number_input(
                "Total Charges ($)", min_value=0.0,
                value=float(tenure * monthly_charges)
            )
            phone_service   = st.selectbox("Phone Service", ["Yes", "No"])

        with col3:
            internet_service  = st.selectbox("Internet Service",
                                             ["DSL", "Fiber optic", "No"])
            contract          = st.selectbox("Contract",
                                             ["Month-to-month", "One year", "Two year"])
            payment_method    = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

        st.markdown('<div class="section-header">📦 Add-on Services</div>',
                    unsafe_allow_html=True)
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            multiple_lines     = st.selectbox("Multiple Lines",
                                              ["Yes", "No", "No phone service"])
            online_security    = st.selectbox("Online Security",
                                              ["Yes", "No", "No internet service"])
        with sc2:
            online_backup      = st.selectbox("Online Backup",
                                              ["Yes", "No", "No internet service"])
            device_protection  = st.selectbox("Device Protection",
                                              ["Yes", "No", "No internet service"])
        with sc3:
            tech_support       = st.selectbox("Tech Support",
                                              ["Yes", "No", "No internet service"])
            streaming_tv       = st.selectbox("Streaming TV",
                                              ["Yes", "No", "No internet service"])
            streaming_movies   = st.selectbox("Streaming Movies",
                                              ["Yes", "No", "No internet service"])

        submitted = st.form_submit_button("⚡ Predict Churn", use_container_width=True)

    # ── Prediction logic ──────────────────────────────────────────────────────
    if submitted:
        user_input = {
            "gender":           gender,
            "SeniorCitizen":    1 if senior_citizen == "Yes" else 0,
            "Partner":          partner,
            "Dependents":       dependents,
            "tenure":           tenure,
            "PhoneService":     phone_service,
            "MultipleLines":    multiple_lines,
            "InternetService":  internet_service,
            "OnlineSecurity":   online_security,
            "OnlineBackup":     online_backup,
            "DeviceProtection": device_protection,
            "TechSupport":      tech_support,
            "StreamingTV":      streaming_tv,
            "StreamingMovies":  streaming_movies,
            "Contract":         contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod":    payment_method,
            "MonthlyCharges":   monthly_charges,
            "TotalCharges":     str(total_charges),
        }

        X = preprocess_single_input(user_input)

        # Align columns with training features
        for col in feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[feature_names]

        probability = model.predict_proba(X)[0][1]
        prediction  = "Yes" if probability >= 0.5 else "No"
        risk_info   = classify_risk(probability)
        recs        = get_recommendations(risk_info["level"])

        st.markdown("---")
        st.subheader("📊 Prediction Results")

        # ── Result cards ──────────────────────────────────────────────────────
        r1, r2, r3 = st.columns(3)

        churn_color = "#ef4444" if prediction == "Yes" else "#22c55e"
        churn_label = "🚨 Churning" if prediction == "Yes" else "✅ Staying"

        with r1:
            st.markdown(f"""<div class="metric-card">
                <h2 style="color:{churn_color}">{churn_label}</h2>
                <p>Churn prediction</p></div>""", unsafe_allow_html=True)

        with r2:
            st.markdown(f"""<div class="metric-card">
                <h2 style="color:{risk_info['hex']}">{probability*100:.1f}%</h2>
                <p>Churn probability</p></div>""", unsafe_allow_html=True)

        with r3:
            st.markdown(f"""<div class="metric-card">
                <h2>{risk_info['emoji']} {risk_info['level']}</h2>
                <p>Risk classification</p></div>""", unsafe_allow_html=True)

        # ── Probability gauge ─────────────────────────────────────────────────
        st.markdown('<div class="section-header">📈 Churn Probability Gauge</div>',
                    unsafe_allow_html=True)

        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=probability * 100,
            delta={"reference": 50, "suffix": "%"},
            number={"suffix": "%", "font": {"size": 28}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar":  {"color": risk_info["hex"]},
                "steps": [
                    {"range": [0, 40],  "color": "#dcfce7"},
                    {"range": [40, 70], "color": "#fef9c3"},
                    {"range": [70, 100],"color": "#fee2e2"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.75,
                    "value": probability * 100,
                },
            },
            title={"text": "Risk Meter"},
        ))
        fig.update_layout(height=280, margin=dict(t=40, b=10, l=30, r=30),
                          paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0")
        st.plotly_chart(fig, use_container_width=True)

        # Simple progress bar as alternative visual
        st.markdown(f"**Churn probability: {probability*100:.1f}%**")
        st.progress(float(probability))

        # ── Recommendations ───────────────────────────────────────────────────
        st.markdown(
            f'<div class="section-header">💡 Recommendations for {risk_info["level"]}</div>',
            unsafe_allow_html=True,
        )
        for icon, text in recs:
            st.markdown(
                f'<div class="rec-card">{icon} &nbsp; {text}</div>',
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "📊  Insights":
    st.title("📊 Churn Insights & Feature Importance")

    if model is None:
        st.error("⚠️ Model not found! Run `python train.py` first.")
        st.stop()

    # ── Feature importance (Random Forest only) ───────────────────────────────
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices     = np.argsort(importances)[::-1]
        top_n       = min(12, len(feature_names))

        top_features = [feature_names[i] for i in indices[:top_n]]
        top_scores   = [importances[i]   for i in indices[:top_n]]

        st.markdown('<div class="section-header">🔑 Top Features Driving Churn</div>',
                    unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(10, 5))
        colors  = ["#6366f1" if s > np.mean(top_scores) else "#94a3b8"
                   for s in top_scores]
        bars    = ax.barh(range(top_n), top_scores[::-1], color=colors[::-1],
                          edgecolor="none", height=0.65)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_features[::-1], fontsize=11)
        ax.set_xlabel("Importance Score", fontsize=11)
        ax.set_title("Feature Importance — Random Forest", fontsize=13, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_facecolor("#0f172a")
        fig.patch.set_facecolor("#0f172a")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#334155")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Plotly interactive bar chart
        st.markdown('<div class="section-header">📊 Interactive Feature Chart</div>',
                    unsafe_allow_html=True)

        fig2 = go.Figure(go.Bar(
            x=top_scores,
            y=top_features,
            orientation="h",
            marker=dict(
                color=top_scores,
                colorscale="Purples",
                showscale=True,
                colorbar=dict(title="Score"),
            ),
        ))
        fig2.update_layout(
            title="Feature Importance (hover for values)",
            xaxis_title="Importance Score",
            height=420,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
        )
        st.plotly_chart(fig2, use_container_width=True)

    else:
        st.info("Feature importance is available only for tree-based models (Random Forest).")

    # ── Key insights panel ────────────────────────────────────────────────────
    st.markdown('<div class="section-header">💡 Key Churn Drivers</div>',
                unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**📄 Contract type**\n\nMonth-to-month customers churn 3× more than annual contract holders.")
    with col2:
        st.warning("**💸 Monthly charges**\n\nCustomers paying >$70/month show significantly higher churn rates.")
    with col3:
        st.success("**⏳ Tenure**\n\nLonger-tenured customers are far more loyal — focus retention early.")

    # ── Risk distribution visual ──────────────────────────────────────────────
    st.markdown('<div class="section-header">📉 Churn Risk Zones</div>',
                unsafe_allow_html=True)

    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        x=["Low Risk (0–40%)", "Medium Risk (40–70%)", "High Risk (70–100%)"],
        y=[45, 30, 25],
        marker_color=["#22c55e", "#f59e0b", "#ef4444"],
        text=["45%", "30%", "25%"],
        textposition="auto",
    ))
    fig3.update_layout(
        title="Typical Risk Distribution in Customer Base",
        yaxis_title="% of Customers",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        height=320,
        showlegend=False,
    )
    st.plotly_chart(fig3, use_container_width=True)
