# 🏦 CreditLens AI

### AI-Powered Loan Eligibility & Credit Decision System with Explainable AI

[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.45-red.svg)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.0-orange.svg)](https://xgboost.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Overview

**CreditLens AI** is an intelligent loan eligibility and credit decision system that leverages machine learning to automate and streamline the loan approval process. Unlike traditional systems, CreditLens AI doesn't just predict — it **explains** every decision using state-of-the-art Explainable AI (XAI) techniques.

### Key Features

| Feature | Description |
|:--------|:------------|
| 🤖 **AI Prediction** | Instant loan eligibility prediction using ensemble ML models |
| 📊 **Risk Scoring** | Credit risk classification (Low / Medium / High) |
| 🔍 **Explainable AI** | SHAP & LIME explanations for every decision |
| ⚡ **Real-time API** | FastAPI REST endpoints for production integration |
| 📈 **Interactive Dashboard** | Streamlit-based analytics and prediction interface |
| 🔬 **Model Comparison** | Side-by-side analysis of 4 ML algorithms |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    APPLICATION LAYER                         │
│   ┌──────────────────┐    ┌──────────────────────────┐      │
│   │  FastAPI Backend  │    │   Streamlit Dashboard    │      │
│   │  REST API Server  │◄──►│  Interactive Analytics   │      │
│   └────────┬─────────┘    └──────────────────────────┘      │
├────────────┼────────────────────────────────────────────────┤
│            │           XAI LAYER                             │
│   ┌────────▼─────────────────────────────────────────┐      │
│   │  SHAP (Global + Local)  │  LIME (Local What-If)  │      │
│   └────────┬─────────────────────────────────────────┘      │
├────────────┼────────────────────────────────────────────────┤
│            │        ML ENGINE LAYER                          │
│   ┌────────▼─────────────────────────────────────────┐      │
│   │  XGBoost │ LightGBM │ Random Forest │ LogReg    │      │
│   │  Cross-Validation │ Optuna Tuning │ Evaluation   │      │
│   └────────┬─────────────────────────────────────────┘      │
├────────────┼────────────────────────────────────────────────┤
│            │          DATA LAYER                             │
│   ┌────────▼─────────────────────────────────────────┐      │
│   │  Loading │ Cleaning │ Feature Engineering │ SMOTE│      │
│   └──────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Category | Technologies |
|:---------|:-------------|
| **Language** | Python 3.13 |
| **ML Models** | XGBoost, LightGBM, Random Forest, Logistic Regression |
| **XAI** | SHAP, LIME |
| **Data** | Pandas, NumPy, Scikit-learn, imbalanced-learn (SMOTE) |
| **Tuning** | Optuna (Bayesian Optimization) |
| **Backend** | FastAPI, Uvicorn, Pydantic |
| **Frontend** | Streamlit |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Package Manager** | uv |

---

## 📁 Project Structure

```
creditlens-ai/
├── pyproject.toml              # Project config & dependencies
├── requirements.txt            # Pinned dependency versions
├── .env.example                # Environment variable template
│
├── src/                        # Core source code
│   ├── config.py               # Project settings & paths
│   ├── data/                   # Data loading & preprocessing
│   │   ├── loader.py           # CSV loading & train/test split
│   │   └── preprocessor.py     # Cleaning, encoding, scaling, SMOTE
│   ├── features/
│   │   └── engineer.py         # Feature engineering (DTI, LTI ratios)
│   ├── models/
│   │   ├── trainer.py          # Model training pipeline
│   │   ├── evaluator.py        # Metrics & comparison charts
│   │   └── tuner.py            # Optuna hyperparameter tuning
│   ├── explainability/
│   │   ├── shap_explainer.py   # SHAP explanations
│   │   └── lime_explainer.py   # LIME explanations
│   └── utils/
│       └── helpers.py          # Logging, I/O, utilities
│
├── api/                        # FastAPI REST API
│   ├── main.py                 # App entry point
│   ├── schemas.py              # Pydantic models
│   ├── routes/                 # API endpoints
│   └── services/               # Business logic
│
├── dashboard/                  # Streamlit dashboard
│   ├── app.py                  # Main dashboard
│   ├── pages/                  # Multi-page navigation
│   └── components/             # Reusable UI components
│
├── data/                       # Dataset storage
│   ├── raw/                    # Original CSV files
│   └── processed/              # Cleaned datasets
│
├── notebooks/                  # Jupyter notebooks for EDA
├── models/                     # Saved trained models (.joblib)
├── reports/                    # Generated plots & reports
└── tests/                      # Unit & integration tests
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager

### 1. Clone & Setup

```bash
git clone <repository-url>
cd capstone-project

# Create virtual environment
python -m uv venv
.venv\Scripts\activate    # Windows

# Install dependencies
python -m uv pip install -r requirements.txt
```

### 2. Configure Environment

```bash
copy .env.example .env
# Edit .env with your settings
```

### 3. Download Dataset

Download the **Loan Approval Prediction Dataset** from Kaggle and place it in `data/raw/`:
- [Kaggle: Loan Approval Prediction](https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data)

### 4. Run the API Server

```bash
python -m uvicorn api.main:app --reload --port 8000
```

Visit the API docs at: http://localhost:8000/docs

### 5. Run the Dashboard

```bash
python -m streamlit run dashboard/app.py
```

Visit the dashboard at: http://localhost:8501

### 6. Run Tests

```bash
python -m pytest tests/ -v
```

---

## 📊 Models & Performance

| Model | AUC-ROC | F1-Score | Status |
|:------|:--------|:---------|:-------|
| XGBoost | — | — | Pending |
| LightGBM | — | — | Pending |
| Random Forest | — | — | Pending |
| Logistic Regression | — | — | Pending |

> 🎯 **Target:** AUC-ROC ≥ 0.85, F1-Score ≥ 0.80

---

## 🔍 Explainable AI

CreditLens AI uses two complementary XAI methods:

- **SHAP (SHapley Additive exPlanations):** Game-theory-based feature attribution for global and local explanations
- **LIME (Local Interpretable Model-agnostic Explanations):** Local linear approximations for What-If analysis

---

## 📝 API Endpoints

| Method | Endpoint | Description |
|:-------|:---------|:------------|
| `GET` | `/api/v1/health` | Health check |
| `GET` | `/api/v1/model/info` | Model metadata |
| `POST` | `/api/v1/predict` | Loan prediction |
| `POST` | `/api/v1/explain` | SHAP/LIME explanation |

---

## 📄 License

This project is licensed under the MIT License.

---

## 👤 Author

**Shaksham** — Capstone Project, 2026

---

<p align="center">
  Built with ❤️ using Python, XGBoost, SHAP & FastAPI
</p>
