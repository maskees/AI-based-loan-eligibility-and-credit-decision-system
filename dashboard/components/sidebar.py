"""
CreditLens AI — Sidebar Component
====================================

Shared sidebar for all dashboard pages.
"""

import streamlit as st


def render_sidebar():
    """Render the shared sidebar component."""
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/bank-building.png", width=80)
        st.title("CreditLens AI")
        st.caption("v0.1.0")
        st.markdown("---")
        st.markdown(
            """
            **AI-Powered Loan System**

            - 🏦 Instant Eligibility Check
            - 📊 Risk Assessment
            - 🔍 Explainable Decisions
            - 📈 Model Analytics
            """
        )
        st.markdown("---")
        st.caption("Capstone Project © 2026")
