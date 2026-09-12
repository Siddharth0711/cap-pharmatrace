# 🏥 PharmaTrace AI (Core Edition) — Dedicated Lightweight Repository

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Forecasting-EB5424?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **ISB AMPBA Capstone | Dedicated 3-Engine Edition**  
> An optimized, lightweight deployment of **PharmaTrace AI** focused specifically on:
> 1. 🤖 **ML Expiry Classifier** (Strategic Engine #1)
> 2. 📈 **Shipments-Driven Demand Prediction & Seasonality** (Strategic Engine #6)
> 3. 🔄 **Reverse Logistics & Certified Disposal** (Strategic Engine #5)

---

## 🎯 Included Strategic Engines

### 1. 🤖 Strategic Engine #1: 4-Color RAG Matrix & ML Expiry Classifier
* **Clinical 4-Color RAG Matrix:** Residual Shelf Life (RSL) tracking (Green >12M, Yellow 7–12M, Amber 4–6M, Red <3M).
* **Multi-Model Tournament:** Evaluates Random Forest, Gradient Boosting, Logistic Regression, and Decision Trees for expiry risk prediction.
* **Explainable AI (XAI):** Feature importance analysis, Velocity-to-Save simulation, and structured recommendations with quantified confidence scores.

### 2. 📈 Strategic Engine #6: Shipments-Driven Demand Prediction & Seasonality
* **XGBoost Regressor:** 1M / 3M / 6M forecasting horizons trained on real shipment transactions.
* **Clinical Pattern Classification:** Rule-based categorization into *Chronic Maintenance*, *Winter Surge*, *Controlled Substance*, and *Specialty Oncology*.
* **80/20 Chronological Split:** Strict out-of-time test set (most recent 20% of months) to evaluate true generalization on future demand.
* **Procurement Workorders & Net Reorder Logic:** Combines Gross Forecast, Safety Stock Buffer (+18%), and Usable On-Hand Warehouse Stock to compute Net Purchase Orders.

### 3. 🔄 Strategic Engine #5: Reverse Logistics & Certified Disposal
* **Customer Returns (RMA) Diagnostics:** Root-cause analysis across damaged goods, recalls, and expired stock.
* **Predictive Recall Severity & Return Classification:** Machine learning triage for regulatory recall alerts.
* **Certified Hazardous Destruction Accounting:** Compliance tracking under US FDA 21 CFR §211 and EPA/DEA incineration mandates.

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Siddharth0711/cap-pharmatrace.git
cd cap-pharmatrace
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Streamlit App
```bash
streamlit run streamlit_app.py
```

The dashboard will launch locally at `http://localhost:8501`.

---

## 📂 Repository Structure

```
cap-pharmatrace/
├── streamlit_app.py                         # Dedicated 3-engine Streamlit application
├── demand_prediction.py                     # XGBoost demand forecasting pipeline
├── requirements.txt                         # Python dependencies
├── PharmaTrace_Data_Template.xlsx           # Uploadable template
├── PharmaTrace_Demand_and_Seasonality_Guide.docx # Detailed operational guide
├── data/
│   ├── warehouse_demand_summary.csv        # Pre-computed warehouse volumes
│   └── warehouse_monthly_trend.csv         # Monthly trends per facility
└── README.md
```

---

## ⚖️ License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
