"""
📊 Analytics Page
=================
Interactive Exploratory Data Analysis (EDA) dashboard.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Analytics | CreditLens AI", page_icon="📊", layout="wide")

st.markdown("# 📊 Dataset Analytics")
st.markdown("Explore the loan dataset with interactive visualizations.")
st.divider()

# ---------------------------------------------------------------------------
# Load Data
# ---------------------------------------------------------------------------
DATA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "raw"


@st.cache_data
def load_data():
    """Load the raw dataset."""
    csv_files = list(DATA_PATH.glob("*.csv"))
    if not csv_files:
        return None
    return pd.read_csv(csv_files[0])


df = load_data()

if df is None:
    st.warning(
        "⚠️ No dataset found in `data/raw/`. "
        "Please place your CSV file there and refresh this page."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Dataset Overview
# ---------------------------------------------------------------------------
st.markdown("## 📋 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Rows", f"{len(df):,}")
col2.metric("Total Columns", f"{len(df.columns):,}")
col3.metric("Missing Values", f"{df.isnull().sum().sum():,}")
col4.metric(
    "Memory Usage",
    f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB",
)

with st.expander("🔍 View Raw Data Sample", expanded=False):
    st.dataframe(df.head(20), use_container_width=True)

with st.expander("📑 Column Details", expanded=False):
    col_info = pd.DataFrame({
        "Data Type": df.dtypes.astype(str),
        "Non-Null Count": df.count(),
        "Null Count": df.isnull().sum(),
        "Null %": (df.isnull().mean() * 100).round(2),
        "Unique Values": df.nunique(),
    })
    st.dataframe(col_info, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Target Distribution
# ---------------------------------------------------------------------------
st.markdown("## 🎯 Target Variable Distribution")

target_col = None
for candidate in ["loan_status", "Loan_Status", "target", "label", "approved"]:
    if candidate in df.columns:
        target_col = candidate
        break

if target_col:
    col1, col2 = st.columns(2)

    with col1:
        target_counts = df[target_col].value_counts()
        fig = px.pie(
            values=target_counts.values,
            names=target_counts.index.astype(str),
            title="Class Distribution",
            color_discrete_sequence=["#667eea", "#e53e3e"],
            hole=0.4,
        )
        fig.update_layout(font=dict(family="Inter"))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            x=target_counts.index.astype(str),
            y=target_counts.values,
            title="Class Counts",
            labels={"x": target_col, "y": "Count"},
            color=target_counts.index.astype(str),
            color_discrete_sequence=["#667eea", "#e53e3e"],
        )
        fig.update_layout(font=dict(family="Inter"), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    balance_ratio = target_counts.min() / target_counts.max()
    if balance_ratio < 0.5:
        st.warning(
            f"⚠️ Class imbalance detected! Ratio = {balance_ratio:.2f}. "
            "SMOTE will be applied during preprocessing."
        )
    else:
        st.success(f"✅ Classes are reasonably balanced (ratio = {balance_ratio:.2f})")
else:
    st.info("Target column not detected. Please verify column names.")

st.divider()

# ---------------------------------------------------------------------------
# Feature Distributions
# ---------------------------------------------------------------------------
st.markdown("## 📊 Feature Distributions")

numerical_cols = df.select_dtypes(include=["number"]).columns.tolist()
categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

tab1, tab2 = st.tabs(["📈 Numerical Features", "📊 Categorical Features"])

with tab1:
    if numerical_cols:
        selected_num = st.selectbox(
            "Select a numerical feature:",
            numerical_cols,
            key="num_feature",
        )
        col1, col2 = st.columns(2)

        with col1:
            fig = px.histogram(
                df, x=selected_num,
                title=f"Distribution of {selected_num}",
                color_discrete_sequence=["#667eea"],
                marginal="box",
            )
            fig.update_layout(font=dict(family="Inter"))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            if target_col:
                fig = px.box(
                    df, x=target_col, y=selected_num,
                    title=f"{selected_num} by {target_col}",
                    color=target_col,
                    color_discrete_sequence=["#667eea", "#e53e3e"],
                )
                fig.update_layout(font=dict(family="Inter"))
                st.plotly_chart(fig, use_container_width=True)

with tab2:
    if categorical_cols:
        selected_cat = st.selectbox(
            "Select a categorical feature:",
            categorical_cols,
            key="cat_feature",
        )
        fig = px.histogram(
            df, x=selected_cat,
            title=f"Distribution of {selected_cat}",
            color=target_col if target_col else None,
            barmode="group",
            color_discrete_sequence=["#667eea", "#e53e3e"],
        )
        fig.update_layout(font=dict(family="Inter"))
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Correlation Heatmap
# ---------------------------------------------------------------------------
st.markdown("## 🔥 Correlation Heatmap")

if len(numerical_cols) >= 2:
    corr_matrix = df[numerical_cols].corr()
    fig = px.imshow(
        corr_matrix,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        title="Feature Correlation Matrix",
        aspect="auto",
    )
    fig.update_layout(
        font=dict(family="Inter"),
        height=600,
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Not enough numerical features for a correlation matrix.")

# ---------------------------------------------------------------------------
# Missing Values
# ---------------------------------------------------------------------------
st.markdown("## ❓ Missing Values Analysis")

missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)

if len(missing) > 0:
    fig = px.bar(
        x=missing.index,
        y=missing.values,
        title="Missing Values by Feature",
        labels={"x": "Feature", "y": "Missing Count"},
        color_discrete_sequence=["#e53e3e"],
    )
    fig.update_layout(font=dict(family="Inter"))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.success("✅ No missing values in the dataset!")
