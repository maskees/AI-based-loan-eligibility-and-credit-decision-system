"""
CreditLens AI — Streamlit Dashboard
=====================================
Main entry point for the interactive dashboard.

Run with:
    uv run streamlit run dashboard/app.py
"""

import streamlit as st

# ---------------------------------------------------------------------------
# Page Configuration (must be the first Streamlit command)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CreditLens AI — Loan Eligibility System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for premium look
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 3rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
    }

    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
    }

    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.07);
    }

    .metric-card h3 {
        font-size: 2rem;
        font-weight: 700;
        color: #2d3748;
        margin: 0;
    }

    .metric-card p {
        font-size: 0.9rem;
        color: #718096;
        margin: 0;
    }

    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.6rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }

    .feature-card {
        background: white;
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a202c 0%, #2d3748 100%);
    }

    div[data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Main Landing Page
# ---------------------------------------------------------------------------

# Header
st.markdown(
    """
    <div class="main-header">
        <h1>🏦 CreditLens AI</h1>
        <p>AI-Based Loan Eligibility & Credit Decision System with Explainable AI</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Feature highlights
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        """
        <div class="metric-card">
            <h3>🤖</h3>
            <p><strong>ML-Powered</strong><br/>XGBoost & LightGBM models trained on real credit data</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div class="metric-card">
            <h3>🔍</h3>
            <p><strong>Explainable</strong><br/>SHAP & LIME explanations for every decision</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        """
        <div class="metric-card">
            <h3>⚡</h3>
            <p><strong>Real-Time</strong><br/>Instant predictions via FastAPI backend</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        """
        <div class="metric-card">
            <h3>📊</h3>
            <p><strong>Analytics</strong><br/>Interactive visualizations & risk scoring</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

# Project overview
st.markdown("## 📌 About This Project")
st.markdown(
    """
    **CreditLens AI** is an end-to-end AI system that automates loan eligibility assessment 
    and provides transparent, explainable credit decisions. Built as a capstone project, it 
    demonstrates the application of modern machine learning and Explainable AI (XAI) in the 
    financial domain.

    ### Key Capabilities
    - **Predict** loan approval/rejection with probability scores
    - **Score** applicant risk level (Low / Medium / High)
    - **Explain** every decision using SHAP & LIME — why was this loan approved or rejected?
    - **Compare** multiple ML models (Logistic Regression, Random Forest, XGBoost, LightGBM)
    - **Visualize** data patterns, model performance, and feature contributions
    """
)

st.divider()

# Navigation guide
st.markdown("## 🧭 Navigation Guide")

nav_col1, nav_col2 = st.columns(2)

with nav_col1:
    st.markdown(
        """
        <div class="feature-card">
            <strong>📊 Analytics</strong><br/>
            Explore the dataset with interactive charts — distributions, 
            correlations, and class balance analysis.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="feature-card">
            <strong>🤖 Predict</strong><br/>
            Submit a loan application and get an instant prediction 
            with approval probability and risk category.
        </div>
        """,
        unsafe_allow_html=True,
    )

with nav_col2:
    st.markdown(
        """
        <div class="feature-card">
            <strong>🔍 Explain</strong><br/>
            Understand why a prediction was made with SHAP waterfall 
            plots and LIME explanation tables.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="feature-card">
            <strong>📈 Model Performance</strong><br/>
            Compare all 4 trained models side-by-side — ROC curves, 
            confusion matrices, and metric rankings.
        </div>
        """,
        unsafe_allow_html=True,
    )

# Footer
st.divider()
st.markdown(
    "<p style='text-align: center; color: #718096;'>"
    "Built with ❤️ using Python, Scikit-learn, XGBoost, SHAP, FastAPI & Streamlit"
    "</p>",
    unsafe_allow_html=True,
)
