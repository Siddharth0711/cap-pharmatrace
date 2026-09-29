# 🏥 PharmaTrace AI — Setup & Execution Guide

**ISB AMPBA Capstone | Sponsor: Innodatatics Inc.**
**Live Dashboard:** https://cap-pharmatrace.streamlit.app
**GitHub Repository:** https://github.com/Siddharth0711/cap-pharmatrace

---

## 📋 Table of Contents
1. [What This Application Does](#what-this-application-does)
2. [Prerequisites](#prerequisites)
3. [Environment Setup](#environment-setup)
4. [Installation Steps](#installation-steps)
5. [How to Run Locally](#how-to-run-locally)
6. [How to Upload Your Data](#how-to-upload-your-data)
7. [Running Individual Engines](#running-individual-engines)
8. [Deployment to Streamlit Cloud](#deployment-to-streamlit-cloud)
9. [Troubleshooting](#troubleshooting)

---

## What This Application Does

PharmaTrace AI is a pharmaceutical supply chain intelligence platform with **4 strategic AI engines**:

| Engine | What It Does |
|--------|-------------|
| 🤖 **ML Expiry Classifier** | Predicts which batches will expire before being sold; 4-colour RAG triage |
| 🌐 **Network Rebalancing & Transfers** | Identifies surplus vs. stockout warehouses; recommends inter-DC transfers |
| 📈 **Demand & Seasonality** | XGBoost 1M/3M/6M demand forecasting per SKU from shipment history |
| 🔄 **Reverse Logistics & Certified Disposal** | Tracks returns, stages recall root-causes, predicts new-batch recall risk |

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.10 or 3.11 | 3.12+ may have scikit-learn conflicts |
| pip | Latest | `python -m pip install --upgrade pip` |
| Git | Any | For cloning the repository |
| RAM | ≥ 4 GB | 8 GB recommended for full dataset |
| Browser | Chrome / Firefox | Safari works but slower rendering |

---

## Environment Setup

### Option A — Virtual Environment (Recommended)

```bash
# Clone the repository
git clone https://github.com/Siddharth0711/cap-pharmatrace.git
cd cap-pharmatrace

# Create a virtual environment
python3 -m venv .venv

# Activate it
# macOS/Linux:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate
```

### Option B — Conda

```bash
conda create -n pharmatrace python=3.10
conda activate pharmatrace
git clone https://github.com/Siddharth0711/cap-pharmatrace.git
cd cap-pharmatrace
```

---

## Installation Steps

```bash
# Install all dependencies from requirements.txt
pip install -r requirements.txt
```

### Key Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit>=1.32` | Dashboard framework |
| `pandas>=2.0` | Data manipulation |
| `numpy` | Numerical computing |
| `scikit-learn` | ML models (RF, LR, GBM, Isolation Forest, K-Means) |
| `xgboost` | Demand forecasting |
| `matplotlib` | Charts and visualizations |
| `openpyxl` | Excel file reading |
| `scipy` | Statistical analysis |

---

## How to Run Locally

```bash
# From the cap-pharmatrace folder (with venv activated):
streamlit run streamlit_app.py
```

The app launches at: **http://localhost:8501**

> ⚠️ The app works with the **built-in synthetic dataset** (no upload needed for demo).
> Upload your real data via **Step 2 → Upload Your Data** in the sidebar.

---

## How to Upload Your Data

The app accepts a **single multi-sheet Excel file**. Download the template first:

```
📥 PharmaTrace_Data_Template.xlsx  ← available in the repository root
```

### Required Sheets (fill and upload)

| Sheet Name | Description | Key Columns |
|------------|-------------|-------------|
| `inventory` | Current warehouse stock | `batch_number`, `product_id`, `warehouse_id`, `quantity_on_hand`, `expiry_date`, `unit_price` |
| `products` | Product master | `product_id`, `generic_name`, `dosage_form`, `shelf_life_months`, `unit_price` |
| `warehouses` | DC/warehouse master | `warehouse_id`, `warehouse_name`, `city`, `type` |
| `shipments` | Historical dispatch data | `shipment_id`, `product_id`, `warehouse_id`, `quantity`, `shipment_date` |
| `monthly_demand` | Monthly demand per SKU/WH | `warehouse_id`, `product_id`, `month`, `quantity_demanded_units`, `quantity_dispatched_units` |
| `freight_matrix` | Inter-warehouse freight costs | `origin_warehouse_id`, `dest_warehouse_id`, `freight_cost_per_unit` |
| `returns` | RMA/customer returns | `return_id`, `fp_batch_id`, `warehouse_id`, `return_reason`, `quantity`, `return_date` |
| `recalls` | FDA recall events | `recall_id`, `product_id`, `classification`, `reason_for_recall` |

### Upload Steps

1. Fill in the template Excel sheets with your data
2. Open the dashboard → sidebar → **Step 2: Upload Your Data**
3. Select your filled `.xlsx` file
4. All 4 engines auto-refresh with your data

---

## Running Individual Engines

All engines load automatically when you navigate to them in the sidebar. No separate commands needed.

### Demand Forecasting Pipeline (standalone)

If you want to pre-generate demand forecasts for faster dashboard loading:

```bash
python demand_prediction.py
```

This generates cached files in `data/`:
- `demand_model_cache.pkl` — trained XGBoost models
- `demand_forecast_results.xlsx` — forecast output table

---

## Deployment to Streamlit Cloud

### Step 1 — Push to GitHub (if not already done)
```bash
git add .
git commit -m "final: submission-ready"
git push origin main
```

### Step 2 — Deploy on Streamlit Cloud
1. Go to https://share.streamlit.io
2. Sign in with your GitHub account
3. Click **"New app"**
4. Fill in:
   - **Repository:** `Siddharth0711/cap-pharmatrace`
   - **Branch:** `main`
   - **Main file path:** `streamlit_app.py`
5. Click **Deploy**

> The live app is already deployed at: **https://cap-pharmatrace.streamlit.app**

### Environment Variables (if needed)
No environment variables or secrets are required. The app runs entirely on uploaded Excel data.

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `ModuleNotFoundError` | Dependencies not installed | `pip install -r requirements.txt` |
| `StreamlitAPIException` | Old Streamlit version | `pip install --upgrade streamlit` |
| Charts not rendering | Matplotlib backend issue | Add `import matplotlib; matplotlib.use('Agg')` at top |
| Upload file too large | Excel > 200 MB | Split into separate uploads per engine |
| XGBoost forecast fails | Insufficient history | Need ≥ 12 months of monthly_demand data |
| Isolation Forest error | Too few batches | Need ≥ 50 rows in `finished_product_batches` sheet |
| Slow loading | Large dataset | Use `@st.cache_data` — already implemented |
| Port 8501 in use | Another Streamlit app | Run `streamlit run streamlit_app.py --server.port 8502` |

---

## Repository File Structure

```
cap-pharmatrace/
│
├── streamlit_app.py                    # Main app — all 4 engines
├── demand_prediction.py                # XGBoost demand forecasting pipeline
├── requirements.txt                    # All Python dependencies
│
├── PharmaTrace_Data_Template.xlsx      # Upload template for your data
├── sample_data.xlsx                    # Pre-loaded demo dataset
│
├── data/                               # Pre-computed cache files
│   ├── demand_model_cache.pkl
│   ├── demand_forecast_results.xlsx
│   ├── warehouse_demand_summary.csv
│   └── warehouse_monthly_trend.csv
│
├── final_submission_requirements/      # ← Submission deliverables (this folder)
│   ├── 04_final_deployable_code/
│   ├── 05_presentation/
│   ├── 07_complete_documentation/
│   └── 08_readme_setup/
│
├── .binder/                            # Binder cloud config
├── .devcontainer/                      # VS Code Dev Container config
├── Dockerfile                          # Docker container definition
├── docker-compose.yml                  # Docker Compose config
├── LICENSE                             # MIT License
└── README.md                           # GitHub landing page
```

---

*PharmaTrace AI | ISB AMPBA Capstone 2026 | Sponsor: Innodatatics Inc.*
