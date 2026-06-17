"""
📈 Model Performance Page
==========================
Side-by-side comparison of all trained ML models.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json

st.set_page_config(page_title="Model Performance | CreditLens AI", page_icon="📈", layout="wide")

st.markdown("# 📈 Model Performance Comparison")
st.markdown(
    "Compare all 4 trained models — Logistic Regression, Random Forest, "
    "XGBoost, and LightGBM — across multiple evaluation metrics."
)
st.divider()

REPORTS_DIR = Path(__file__).resolve().parent.parent.parent / "reports"
MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"

# ---------------------------------------------------------------------------
# Load saved results (if available)
# ---------------------------------------------------------------------------

results_path = REPORTS_DIR / "model_comparison.json"


@st.cache_data
def load_results():
    """Load model comparison results."""
    if results_path.exists():
        with open(results_path) as f:
            return json.load(f)
    return None


results = load_results()

if results is None:
    st.warning(
        "⚠️ No model comparison results found. Please run the training pipeline first:\n\n"
        "```bash\nuv run python main.py\n```\n\n"
        "This will train all models and save the comparison results."
    )

    # Show a sample of what the page will look like
    st.markdown("### 📊 Preview (Sample Data)")

    sample_data = {
        "Model": ["XGBoost", "LightGBM", "Random Forest", "Logistic Regression"],
        "Accuracy": [0.9412, 0.9385, 0.9201, 0.8756],
        "Precision": [0.9345, 0.9312, 0.9156, 0.8621],
        "Recall": [0.9289, 0.9267, 0.8978, 0.8534],
        "F1-Score": [0.9317, 0.9289, 0.9066, 0.8577],
        "AUC-ROC": [0.9678, 0.9654, 0.9421, 0.9012],
    }
    sample_df = pd.DataFrame(sample_data).set_index("Model")

    st.dataframe(
        sample_df.style.highlight_max(axis=0, color="#c6f6d5"),
        use_container_width=True,
    )

    # Sample bar chart
    fig = px.bar(
        sample_df.reset_index(),
        x="Model",
        y=["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"],
        barmode="group",
        title="Model Performance Comparison (Sample)",
        color_discrete_sequence=["#667eea", "#764ba2", "#38a169", "#e53e3e", "#d69e2e"],
    )
    fig.update_layout(
        font=dict(family="Inter"),
        yaxis_title="Score",
        yaxis_range=[0, 1.05],
    )
    st.plotly_chart(fig, use_container_width=True)

    st.stop()

# ---------------------------------------------------------------------------
# Display actual results
# ---------------------------------------------------------------------------

# Comparison table
st.markdown("## 📊 Metrics Comparison Table")

comparison_df = pd.DataFrame(results)
if "Model" in comparison_df.columns:
    comparison_df = comparison_df.set_index("Model")

st.dataframe(
    comparison_df.style.highlight_max(axis=0, color="#c6f6d5")
    .highlight_min(axis=0, color="#fed7d7")
    .format("{:.4f}"),
    use_container_width=True,
)

# Best model callout
best_model = comparison_df["AUC-ROC"].idxmax() if "AUC-ROC" in comparison_df.columns else "Unknown"
best_auc = comparison_df.loc[best_model, "AUC-ROC"] if best_model != "Unknown" else 0

st.success(
    f"🏆 **Best Model: {best_model}** with AUC-ROC = {best_auc:.4f}"
)

st.divider()

# Grouped bar chart
st.markdown("## 📊 Visual Comparison")

metrics_to_plot = [c for c in comparison_df.columns if c in [
    "Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC",
]]

fig = px.bar(
    comparison_df.reset_index(),
    x="Model" if "Model" in comparison_df.reset_index().columns else comparison_df.index.name,
    y=metrics_to_plot,
    barmode="group",
    title="Model Performance Comparison",
    color_discrete_sequence=["#667eea", "#764ba2", "#38a169", "#e53e3e", "#d69e2e"],
)
fig.update_layout(
    font=dict(family="Inter"),
    yaxis_title="Score",
    yaxis_range=[0, 1.05],
    height=500,
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Saved Plots
# ---------------------------------------------------------------------------

st.markdown("## 📊 Detailed Visualizations")

tab1, tab2, tab3 = st.tabs(["ROC Curves", "Confusion Matrices", "Precision-Recall"])

with tab1:
    roc_path = REPORTS_DIR / "roc_curves.png"
    if roc_path.exists():
        st.image(str(roc_path), caption="ROC Curves — All Models", use_container_width=True)
    else:
        st.info("ROC curve plot not found. Run training pipeline to generate.")

with tab2:
    cm_path = REPORTS_DIR / "confusion_matrices.png"
    if cm_path.exists():
        st.image(str(cm_path), caption="Confusion Matrices — All Models", use_container_width=True)
    else:
        st.info("Confusion matrix plot not found. Run training pipeline to generate.")

with tab3:
    pr_path = REPORTS_DIR / "precision_recall_curves.png"
    if pr_path.exists():
        st.image(str(pr_path), caption="Precision-Recall Curves — All Models", use_container_width=True)
    else:
        st.info("Precision-Recall plot not found. Run training pipeline to generate.")
