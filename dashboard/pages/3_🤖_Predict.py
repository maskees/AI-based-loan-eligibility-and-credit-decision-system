"""
🤖 Predict Page
================
Loan application form for real-time prediction.
"""

import streamlit as st
import httpx
import json

st.set_page_config(page_title="Predict | CreditLens AI", page_icon="🤖", layout="wide")

st.markdown("# 🤖 Loan Eligibility Prediction")
st.markdown("Fill in the applicant details below to get an instant AI-powered prediction.")
st.divider()

# API endpoint
API_URL = "http://127.0.0.1:8000/api/v1"

# ---------------------------------------------------------------------------
# Application Form
# ---------------------------------------------------------------------------

with st.form("loan_application_form"):
    st.markdown("### 👤 Applicant Information")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])

    with col2:
        employment_status = st.selectbox(
            "Employment Status",
            ["Employed", "Self-Employed", "Unemployed"],
        )
        marital_status = st.selectbox(
            "Marital Status",
            ["Married", "Single", "Divorced"],
        )
        number_of_dependents = st.number_input(
            "Number of Dependents",
            min_value=0, max_value=15, value=2,
        )

    with col3:
        annual_income = st.number_input(
            "Annual Income (₹)",
            min_value=0.0, value=750000.0, step=50000.0,
            format="%.0f",
        )
        credit_score = st.slider(
            "Credit / CIBIL Score",
            min_value=300, max_value=900, value=720,
        )
        previous_defaults = st.number_input(
            "Previous Defaults",
            min_value=0, max_value=20, value=0,
        )

    st.markdown("### 💰 Loan Details")
    col4, col5, col6 = st.columns(3)

    with col4:
        loan_amount = st.number_input(
            "Loan Amount (₹)",
            min_value=0.0, value=2500000.0, step=100000.0,
            format="%.0f",
        )

    with col5:
        loan_term = st.number_input(
            "Loan Term (years)",
            min_value=1, max_value=30, value=15,
        )

    with col6:
        loan_purpose = st.selectbox(
            "Loan Purpose",
            ["Home", "Car", "Education", "Personal", "Business"],
        )

    st.markdown("### 🏠 Assets")
    assets_value = st.number_input(
        "Total Assets Value (₹)",
        min_value=0.0, value=5000000.0, step=500000.0,
        format="%.0f",
    )

    st.divider()
    submitted = st.form_submit_button(
        "🚀 Get Prediction",
        use_container_width=True,
    )

# ---------------------------------------------------------------------------
# Handle Submission
# ---------------------------------------------------------------------------

if submitted:
    application_data = {
        "age": age,
        "gender": gender,
        "education": education,
        "employment_status": employment_status,
        "marital_status": marital_status,
        "annual_income": annual_income,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "loan_purpose": loan_purpose,
        "credit_score": credit_score,
        "number_of_dependents": number_of_dependents,
        "previous_defaults": previous_defaults,
        "assets_value": assets_value,
    }

    # Store in session state for the Explain page
    st.session_state["last_application"] = application_data

    with st.spinner("🔄 Analyzing application..."):
        try:
            response = httpx.post(
                f"{API_URL}/predict/",
                json=application_data,
                timeout=30.0,
            )

            if response.status_code == 200:
                result = response.json()
                st.session_state["last_prediction"] = result

                st.divider()
                st.markdown("## 📋 Prediction Result")

                # Result cards
                col1, col2, col3, col4 = st.columns(4)

                prediction = result["prediction"]
                probability = result["probability"]
                risk = result["risk_category"]

                with col1:
                    color = "#38a169" if prediction == "Approved" else "#e53e3e"
                    icon = "✅" if prediction == "Approved" else "❌"
                    st.markdown(
                        f"""
                        <div style="background: {color}; padding: 1.5rem; 
                        border-radius: 12px; text-align: center; color: white;">
                            <h2 style="margin:0">{icon} {prediction}</h2>
                            <p style="margin:0">Loan Decision</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with col2:
                    st.metric("Approval Probability", f"{probability:.1%}")

                with col3:
                    risk_colors = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
                    st.metric(
                        "Risk Category",
                        f"{risk_colors.get(risk, '')} {risk}",
                    )

                with col4:
                    st.metric("Loan ID", result["loan_id"])

                # Top factors
                if result.get("top_factors"):
                    st.markdown("### 🔍 Top Contributing Factors")
                    for factor in result["top_factors"]:
                        impact_icon = "📈" if factor["impact"] == "positive" else "📉"
                        st.markdown(
                            f"- {impact_icon} **{factor['feature']}**: "
                            f"SHAP value = {factor.get('shap_value', 'N/A')}"
                        )

                st.info(
                    "💡 Go to the **🔍 Explain** page for detailed SHAP & LIME explanations."
                )

            elif response.status_code == 503:
                st.error(
                    "⚠️ Model not loaded! Please run the training pipeline first:\n\n"
                    "```bash\nuv run python main.py\n```"
                )
            else:
                st.error(f"API Error: {response.status_code} — {response.text}")

        except httpx.ConnectError:
            st.error(
                "🔌 Cannot connect to the API server. Please start it first:\n\n"
                "```bash\nuv run uvicorn api.main:app --reload\n```"
            )
        except Exception as e:
            st.error(f"Error: {str(e)}")
