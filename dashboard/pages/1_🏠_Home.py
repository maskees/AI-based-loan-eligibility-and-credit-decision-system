"""
🏠 Home Page
=============
Landing page for the CreditLens AI Streamlit dashboard.
Displays project overview, live model metrics, system status,
and a quick-start guide.
"""

import json
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Home | CreditLens AI",
    page_icon="🏠",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS — consistent with app.py
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .hero-banner {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem 3rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.35);
    }
    .hero-banner h1 { font-size: 2.6rem; font-weight: 700; margin-bottom: 0.4rem; }
    .hero-banner p  { font-size: 1.15rem; opacity: 0.92; margin: 0; }

    .stat-card {
        background: white;
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border-top: 4px solid #667eea;
    }
    .stat-card .value { font-size: 2.1rem; font-weight: 700; color: #2d3748; }
    .stat-card .label { font-size: 0.85rem; color: #718096; margin-top: 0.2rem; }

    .model-row {
        background: #f7fafc;
        border-radius: 10px;
        padding: 0.9rem 1.2rem;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        border-left: 4px solid #667eea;
    }

    .status-ok   { color: #38a169; font-weight: 600; }
    .status-warn { color: #dd6b20; font-weight: 600; }
    .status-err  { color: #e53e3e; font-weight: 600; }

    .guide-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        border-left: 4px solid #764ba2;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a202c 0%, #2d3748 100%);
    }
    div[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS_DIR  = PROJECT_ROOT / "reports"
MODELS_DIR   = PROJECT_ROOT / "models"

# ---------------------------------------------------------------------------
# Hero Banner
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-banner">
        <h1>🏦 CreditLens AI</h1>
        <p>AI-Based Loan Eligibility &amp; Credit Decision System with Explainable AI</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# System Status
# ---------------------------------------------------------------------------
model_ready = (MODELS_DIR / "best_model.joblib").exists()
preprocessor_ready = (MODELS_DIR / "preprocessor.joblib").exists()
report_ready = (REPORTS_DIR / "model_comparison.json").exists()

col_s1, col_s2, col_s3 = st.columns(3)

with col_s1:
    icon = "✅" if model_ready else "❌"
    cls  = "status-ok" if model_ready else "status-err"
    label = "Model loaded" if model_ready else "Model not found — run main.py"
    st.markdown(
        f"<div class='guide-card'>{icon} <span class='{cls}'>{label}</span></div>",
        unsafe_allow_html=True,
    )

with col_s2:
    icon = "✅" if preprocessor_ready else "❌"
    cls  = "status-ok" if preprocessor_ready else "status-err"
    label = "Preprocessor ready" if preprocessor_ready else "Preprocessor missing"
    st.markdown(
        f"<div class='guide-card'>{icon} <span class='{cls}'>{label}</span></div>",
        unsafe_allow_html=True,
    )

with col_s3:
    icon = "✅" if report_ready else "⚠️"
    cls  = "status-ok" if report_ready else "status-warn"
    label = "Evaluation report available" if report_ready else "Report not generated yet"
    st.markdown(
        f"<div class='guide-card'>{icon} <span class='{cls}'>{label}</span></div>",
        unsafe_allow_html=True,
    )

st.divider()

# ---------------------------------------------------------------------------
# Live Model Metrics  (from saved model_comparison.json)
# ---------------------------------------------------------------------------
st.markdown("## 🏆 Trained Model Performance")

comparison_path = REPORTS_DIR / "model_comparison.json"
if comparison_path.exists():
    with open(comparison_path) as f:
        model_data = json.load(f)

    best = model_data[0]  # Already sorted best-first

    # KPI strip
    m1, m2, m3, m4, m5 = st.columns(5)
    metrics = [
        ("Best Model",  best["Model"],             ""),
        ("AUC-ROC",     f"{best['AUC-ROC']:.4f}",  "Target ≥ 0.85 ✅"),
        ("F1-Score",    f"{best['F1-Score']:.4f}",  "Target ≥ 0.80 ✅"),
        ("Accuracy",    f"{best['Accuracy']:.4f}",  ""),
        ("Precision",   f"{best['Precision']:.4f}", ""),
    ]
    for col, (label, value, delta) in zip([m1, m2, m3, m4, m5], metrics):
        with col:
            st.metric(label=label, value=value, delta=delta if delta else None)

    st.markdown("#### All Models — Comparison")

    rank_icons = ["🥇", "🥈", "🥉", "4️⃣"]
    for i, model in enumerate(model_data):
        icon = rank_icons[i] if i < len(rank_icons) else f"{i+1}."
        st.markdown(
            f"""
            <div class="model-row">
                <span style="font-size:1.3rem;margin-right:0.8rem">{icon}</span>
                <div style="flex:1">
                    <strong>{model['Model']}</strong>
                </div>
                <div style="display:flex;gap:2rem;font-size:0.9rem;color:#4a5568">
                    <span>AUC <strong>{model['AUC-ROC']:.4f}</strong></span>
                    <span>F1 <strong>{model['F1-Score']:.4f}</strong></span>
                    <span>Acc <strong>{model['Accuracy']:.4f}</strong></span>
                    <span>Prec <strong>{model['Precision']:.4f}</strong></span>
                    <span>Recall <strong>{model['Recall']:.4f}</strong></span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
else:
    st.info(
        "⚠️ Model comparison report not found. "
        "Run `uv run python main.py` to train models and generate reports."
    )

st.divider()

# ---------------------------------------------------------------------------
# SHAP Plots preview (if available)
# ---------------------------------------------------------------------------
shap_summary = REPORTS_DIR / "shap_summary_plot.png"
shap_importance = REPORTS_DIR / "shap_feature_importance.png"

if shap_summary.exists() or shap_importance.exists():
    st.markdown("## 🔍 SHAP Explainability Snapshots")
    img_cols = st.columns(2)
    if shap_importance.exists():
        with img_cols[0]:
            st.image(str(shap_importance), caption="Global Feature Importance (SHAP)", use_container_width=True)
    if shap_summary.exists():
        with img_cols[1]:
            st.image(str(shap_summary), caption="SHAP Summary Plot", use_container_width=True)
    st.divider()

# ---------------------------------------------------------------------------
# Navigation Guide
# ---------------------------------------------------------------------------
st.markdown("## 🧭 Dashboard Navigation")

nav = [
    ("📊", "Analytics",        "Explore the dataset — distributions, correlations, class balance"),
    ("🤖", "Predict",          "Submit a loan application and get an instant AI decision with probability score"),
    ("🔍", "Explain",          "Understand WHY a prediction was made via SHAP waterfall & LIME tables"),
    ("📈", "Model Performance", "Compare all 4 models side-by-side — ROC curves, confusion matrices, rankings"),
]
n1, n2 = st.columns(2)
for i, (icon, title, desc) in enumerate(nav):
    col = n1 if i % 2 == 0 else n2
    with col:
        st.markdown(
            f"""
            <div class="guide-card">
                <strong>{icon} {title}</strong><br/>
                <span style="color:#718096;font-size:0.9rem">{desc}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.divider()

# ---------------------------------------------------------------------------
# Quick-Start Commands
# ---------------------------------------------------------------------------
st.markdown("## ⚡ Quick-Start Commands")

col_q1, col_q2, col_q3 = st.columns(3)

with col_q1:
    st.markdown("**Train / retrain models**")
    st.code("uv run python main.py", language="bash")
    st.caption("Runs the full 11-step ML pipeline")

with col_q2:
    st.markdown("**Start the FastAPI backend**")
    st.code("uv run uvicorn api.main:app --reload", language="bash")
    st.caption("API docs → http://localhost:8000/docs")

with col_q3:
    st.markdown("**Run tests**")
    st.code("uv run pytest tests/ -v", language="bash")
    st.caption("All 24 tests should pass")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.markdown(
    "<p style='text-align:center;color:#718096;font-size:0.9rem'>"
    "Built with ❤️ using Python · Scikit-learn · XGBoost · LightGBM · SHAP · FastAPI · Streamlit"
    "</p>",
    unsafe_allow_html=True,
)
