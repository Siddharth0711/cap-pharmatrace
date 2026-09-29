# PharmaTrace AI — Code Manifest & Deployable Code Guide

**Deliverable 4: Final Deployable Code**
**ISB AMPBA Capstone | Sponsor: Innodatatics Inc.**

---

## What to Submit

The following files constitute the **complete, production-ready codebase**. All are in the root of the `cap-pharmatrace` GitHub repository.

---

## Core Application Files

| File | Size | Role |
|------|------|------|
| `streamlit_app.py` | ~384 KB | **Main application** — contains all 4 AI engines, all UI pages, all charts, all AI insight cards |
| `demand_prediction.py` | ~32 KB | **Demand forecasting pipeline** — standalone XGBoost training script for Engine 3 |
| `requirements.txt` | ~1.5 KB | **Python dependencies** — all packages needed to run the app |

---

## Engine-to-Code Mapping

### Engine 1: ML Expiry Classifier
**Location in `streamlit_app.py`:** Search for `"🤖 ML Expiry Classifier"`

| Component | Code Location | Description |
|-----------|--------------|-------------|
| RAG Matrix | `elif selected_page == "🤖 ML Expiry Classifier"` | 4-colour RAG zone assignment logic |
| Tab 1 — RAG Dashboard | `with tab_rag:` | KPI metrics, zone breakdown charts, action cards |
| Tab 2 — ML Classifier | `with tab_ml_strat:` | Model training, demand scenario toggle, SKU Pareto, warehouse risk |
| Tab 3 — Recovery Playbook | `with tab_xai_strat:` | Zone-by-zone recovery action register |
| RAG Assignment Logic | `def assign_rag(row)` | Green/Yellow/Amber/Red classification function |
| Feature Engineering | `ml_df[...]` computations | cover_days, velocity_pressure, risk_score, pct_life_remaining |
| Model Tournament | `LogisticRegression`, `RandomForestClassifier`, `GradientBoostingClassifier` | Champion model selection by Weighted F1 |

### Engine 2: Network Rebalancing & Transfers
**Location in `streamlit_app.py`:** Search for `"🌐 Network Rebalancing & Transfers"`

| Component | Code Location | Description |
|-----------|--------------|-------------|
| HOT/COLD detection | `build_network_data()` | Days-of-stock < 30 = HOT, > 120 = COLD |
| Transfer Recommender | `df_b_trans` computation block | Batch-level ML transfer scoring |
| Freight vs. Manufacturing cost | `_cost_compare` block | Side-by-side cost analysis |
| AI Insight | `_net_bullets` + `ai_insight()` | Dynamic network intelligence card |
| Recovery Playbook | `_playbook_bullets` block | Capital reclamation recommendations |

### Engine 3: Demand & Seasonality
**Location in `streamlit_app.py`:** Search for `"📈 Demand & Seasonality"`
**Also:** `demand_prediction.py` (standalone pipeline)

| Component | Code Location | Description |
|-----------|--------------|-------------|
| XGBoost training | `demand_prediction.py` — `train_model()` | Per-horizon model training |
| Pattern classification | `demand_prediction.py` — `classify_pattern()` | 7-class rule-based clinical classifier |
| 1M/3M/6M forecasts | `_df_forecasts` in `streamlit_app.py` | Loaded from cache or live-trained |
| Procurement workorders | Tab 2 — `Procurement Action Plan` | Net order = Gross forecast − usable on-hand |
| AI Insight — Demand | `ai_insight("Demand Patterns...")` | Dynamic demand intelligence |
| AI Insight — Warehouse | `ai_insight("Warehouse & Supply...")` | Dynamic facility intelligence |

### Engine 4: Reverse Logistics & Certified Disposal
**Location in `streamlit_app.py`:** Search for `"🔄 Reverse Logistics & Certified Disposal"`

| Component | Code Location | Description |
|-----------|--------------|-------------|
| Tab 1 — Operations | `with tab_ops:` | RMA charts, disposal methods, financial leakage |
| Tab 2 — Recall Intelligence | `with tab_recall_intel:` | 4-stage recall staging, 1-10-100 table, K-Means NLP |
| Tab 3 — Batch Predictor | `with tab_predictor:` | Random Forest batch recall probability simulator |
| Supply chain staging | `map_recall_stage()` | NLP rule-based stage classification |
| K-Means clustering | `KMeans(n_clusters=4)` + `TfidfVectorizer` | Defect archetype discovery |
| Isolation Forest | `IsolationForest(contamination=0.035)` | Manufacturing anomaly detection |
| Disposal ML | `clf_dsp = RandomForestClassifier(...)` | EPA/DEA disposal method recommender |
| AI Insight | `ai_bullets_m5` + `ai_insight()` | Dynamic reverse logistics intelligence |

---

## Supporting Data Files

| File | Role |
|------|------|
| `PharmaTrace_Data_Template.xlsx` | Upload template — users fill this with real data |
| `sample_data.xlsx` | Pre-loaded synthetic demo dataset (242 KB) |
| `data/demand_model_cache.pkl` | Pre-trained XGBoost model cache (avoids re-training on load) |
| `data/demand_forecast_results.xlsx` | Pre-computed forecast results |
| `data/warehouse_demand_summary.csv` | Pre-aggregated warehouse demand |
| `data/warehouse_monthly_trend.csv` | Monthly trend data per warehouse |

---

## Configuration Files

| File | Purpose |
|------|---------|
| `requirements.txt` | Python package versions |
| `.python-version` | Python runtime version (`3.11.x`) |
| `Dockerfile` | Docker container definition |
| `docker-compose.yml` | Docker Compose multi-service config |
| `.binder/requirements.txt` | Binder cloud deployment dependencies |
| `.devcontainer/` | VS Code Dev Container configuration |
| `.github/` | GitHub Actions CI/CD workflows |
| `.gitignore` | Files excluded from version control |

---

## Utility Functions (inside streamlit_app.py)

| Function | Purpose |
|----------|---------|
| `fmt_curr(value, compact, decimals)` | Format currency with K/M/B suffix |
| `show_fig(fig)` | Render matplotlib figure to Streamlit |
| `ai_insight(title, bullets, icon, color)` | Render AI insight card with dark-themed styling |
| `info_box(key, text)` | Collapsible info/glossary expander |
| `get_current_glossary()` | Returns all engine glossary definitions |
| `get_data(uploaded_file)` | Load & parse uploaded Excel → all DataFrames |
| `build_network_data(demand_hash, freight_hash)` | Cached network geo computation |
| `assign_rag(row)` | 4-colour RAG zone assignment |

---

## Models Used — Summary

| Engine | Model | Type | Target | Performance |
|--------|-------|------|--------|-------------|
| Engine 1 | Random Forest | Supervised — Classification | Will batch expire before sale? | ~95%+ F1 |
| Engine 1 | Gradient Boosting | Supervised — Classification | Same | Tournament competitor |
| Engine 1 | Logistic Regression | Supervised — Classification | Same | Tournament competitor |
| Engine 2 | Rule-based scoring | Heuristic | HOT/COLD warehouse detection | — |
| Engine 2 | ML batch scoring | Score-based ranking | Transfer priority | — |
| Engine 3 | XGBoost Regressor | Supervised — Regression | Units demanded per SKU/WH | MAPE < 20% |
| Engine 3 | Rule-based | Classification | Demand pattern (7 classes) | — |
| Engine 4 | K-Means + TF-IDF | Unsupervised — Clustering | Recall defect archetypes | k=4 clusters |
| Engine 4 | Isolation Forest | Unsupervised — Anomaly | Manufacturing yield anomaly | Contamination=3.5% |
| Engine 4 | Random Forest | Supervised — Classification | New batch recall probability | ~85%+ accuracy |
| Engine 4 | Random Forest | Supervised — Classification | EPA disposal method | Multi-class |

---

## GitHub Repository

**URL:** https://github.com/Siddharth0711/cap-pharmatrace
**Branch:** `main`
**Live Deployment:** https://cap-pharmatrace.streamlit.app

*All files are version-controlled and production-ready as of the submission date.*
