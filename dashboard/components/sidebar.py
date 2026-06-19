"""
CreditLens AI — Sidebar Component
====================================
Shared sidebar for all dashboard pages.
Import and call render_sidebar() at the top of each page.
"""

from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"


def render_sidebar() -> None:
    """Render the shared sidebar with system status and navigation links."""
    with st.sidebar:
        # Branding
        st.markdown(
            """
            <div style="text-align:center;padding:0.8rem 0 0.4rem">
                <span style="font-size:2.4rem">🏦</span>
                <h2 style="margin:0.3rem 0 0;font-size:1.15rem;font-weight:700">
                    CreditLens AI
                </h2>
                <p style="font-size:0.76rem;opacity:0.65;margin:0">
                    Loan Eligibility System v1.0
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        # System status
        st.markdown("**⚙️ System Status**")
        model_ok = (MODELS_DIR / "best_model.joblib").exists()
        prep_ok  = (MODELS_DIR / "preprocessor.joblib").exists()

        if model_ok and prep_ok:
            st.success("✅ Model ready")
        elif model_ok:
            st.warning("⚠️ Preprocessor missing")
        else:
            st.error("❌ Model not trained")
            st.caption("Run `uv run python main.py` first")

        st.divider()

        # Key capabilities
        st.markdown(
            """
            **🔑 Key Capabilities**
            - 🏦 Instant eligibility check
            - 📊 Risk assessment (Low/Medium/High)
            - 🔍 SHAP & LIME explanations
            - 📈 4-model comparative analysis
            """
        )

        st.divider()

        # API links
        st.markdown("**📡 API**")
        st.markdown("[Swagger UI](http://localhost:8000/docs) · [Health](http://localhost:8000/api/v1/health)")

        st.divider()
        st.caption("Capstone Project © 2026")
