"""
🔍 Explain Page
================
SHAP and LIME explanations for loan predictions.
"""

import streamlit as st
import httpx
import plotly.graph_objects as go

st.set_page_config(page_title="Explain | CreditLens AI", page_icon="🔍", layout="wide")

st.markdown("# 🔍 Decision Explainability")
st.markdown(
    "Understand **why** a loan was approved or rejected using "
    "SHAP (global + local) and LIME (local) explanations."
)
st.divider()

API_URL = "http://127.0.0.1:8000/api/v1"

# ---------------------------------------------------------------------------
# Check for previous prediction
# ---------------------------------------------------------------------------

if "last_application" not in st.session_state:
    st.info(
        "💡 No prediction to explain yet. "
        "Go to the **🤖 Predict** page first to submit a loan application."
    )
    st.stop()

application = st.session_state["last_application"]
prediction = st.session_state.get("last_prediction", {})

# Show current application summary
with st.expander("📋 Current Application Details", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(f"**Age:** {application['age']}")
        st.write(f"**Gender:** {application['gender']}")
        st.write(f"**Education:** {application['education']}")
        st.write(f"**Employment:** {application['employment_status']}")
    with col2:
        st.write(f"**Income:** ₹{application['annual_income']:,.0f}")
        st.write(f"**Loan Amount:** ₹{application['loan_amount']:,.0f}")
        st.write(f"**Loan Term:** {application['loan_term']} years")
        st.write(f"**Credit Score:** {application['credit_score']}")
    with col3:
        st.write(f"**Dependents:** {application['number_of_dependents']}")
        st.write(f"**Previous Defaults:** {application['previous_defaults']}")
        st.write(f"**Assets Value:** ₹{application['assets_value']:,.0f}")
        if prediction:
            color = "green" if prediction.get("prediction") == "Approved" else "red"
            st.markdown(
                f"**Prediction:** :{color}[{prediction.get('prediction', 'N/A')}] "
                f"({prediction.get('probability', 0):.1%})"
            )

st.divider()

# ---------------------------------------------------------------------------
# Explanation Controls
# ---------------------------------------------------------------------------

col1, col2 = st.columns([1, 3])

with col1:
    method = st.radio(
        "Explanation Method",
        ["SHAP", "LIME", "Both"],
        index=0,
    )
    top_n = st.slider("Top N Features", min_value=3, max_value=15, value=5)

with col2:
    if st.button("🔍 Generate Explanation", use_container_width=True):
        with st.spinner("Generating explanation..."):
            try:
                response = httpx.post(
                    f"{API_URL}/explain/",
                    json={
                        "application": application,
                        "method": method.lower(),
                        "top_n_features": top_n,
                    },
                    timeout=60.0,
                )

                if response.status_code == 200:
                    result = response.json()
                    st.session_state["last_explanation"] = result
                    st.success("✅ Explanation generated!")
                else:
                    st.error(f"API Error: {response.status_code} — {response.text}")

            except httpx.ConnectError:
                st.error(
                    "🔌 Cannot connect to the API. Start it with:\n\n"
                    "```bash\nuv run uvicorn api.main:app --reload\n```"
                )
            except Exception as e:
                st.error(f"Error: {str(e)}")

# ---------------------------------------------------------------------------
# Display Explanation
# ---------------------------------------------------------------------------

if "last_explanation" in st.session_state:
    explanation = st.session_state["last_explanation"]
    st.divider()

    # SHAP Explanation
    if explanation.get("shap_explanation"):
        st.markdown("## 📊 SHAP Explanation")
        st.markdown(
            "SHAP (SHapley Additive exPlanations) shows how each feature "
            "pushed the prediction toward approval or rejection."
        )

        shap_data = explanation["shap_explanation"]

        # Create waterfall-style horizontal bar chart
        features = [d["feature"] for d in shap_data]
        shap_values = [d.get("shap_value", 0) for d in shap_data]
        colors = ["#38a169" if v > 0 else "#e53e3e" for v in shap_values]

        fig = go.Figure(
            go.Bar(
                x=shap_values,
                y=features,
                orientation="h",
                marker_color=colors,
                text=[f"{v:+.4f}" for v in shap_values],
                textposition="outside",
            )
        )
        fig.update_layout(
            title="Feature Contributions (SHAP Values)",
            xaxis_title="SHAP Value (Impact on Prediction)",
            yaxis_title="Feature",
            font=dict(family="Inter"),
            height=max(400, len(features) * 50),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Explanation table
        st.markdown("### 📝 Feature Impact Summary")
        for factor in shap_data:
            icon = "🟢" if factor["impact"] == "positive" else "🔴"
            st.markdown(
                f"{icon} **{factor['feature']}** — "
                f"SHAP = {factor.get('shap_value', 0):+.4f} | "
                f"Impact: *{factor['impact']}* (pushes toward "
                f"{'approval' if factor['impact'] == 'positive' else 'rejection'})"
            )

    # LIME Explanation
    if explanation.get("lime_explanation"):
        st.divider()
        st.markdown("## 🍋 LIME Explanation")
        st.markdown(
            "LIME creates a local linear approximation to explain "
            "individual predictions in human-readable rules."
        )

        lime_data = explanation["lime_explanation"]

        features = [d["feature_rule"] for d in lime_data]
        contributions = [d["contribution"] for d in lime_data]
        colors = ["#38a169" if c > 0 else "#e53e3e" for c in contributions]

        fig = go.Figure(
            go.Bar(
                x=contributions,
                y=features,
                orientation="h",
                marker_color=colors,
                text=[f"{c:+.4f}" for c in contributions],
                textposition="outside",
            )
        )
        fig.update_layout(
            title="LIME Feature Contributions",
            xaxis_title="Contribution to Prediction",
            font=dict(family="Inter"),
            height=max(400, len(features) * 50),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig, use_container_width=True)
