# PharmaTrace AI — Complete Technical & Functional Documentation

**Deliverable 7: Complete Project Documentation**
**ISB AMPBA Capstone | Sponsor: Innodatatics Inc.**
**Version:** Final Submission | September 2026

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture & Data Flow](#2-architecture--data-flow)
3. [Engine 1 — ML Expiry Classifier](#3-engine-1--ml-expiry-classifier)
4. [Engine 2 — Network Rebalancing & Transfers](#4-engine-2--network-rebalancing--transfers)
5. [Engine 3 — Demand & Seasonality](#5-engine-3--demand--seasonality)
6. [Engine 4 — Reverse Logistics & Certified Disposal](#6-engine-4--reverse-logistics--certified-disposal)
7. [Shared UI Components](#7-shared-ui-components)
8. [Data Schema Reference](#8-data-schema-reference)
9. [Models Reference](#9-models-reference)
10. [Regulatory Compliance Mapping](#10-regulatory-compliance-mapping)
11. [Known Limitations](#11-known-limitations)
12. [Future Enhancements](#12-future-enhancements)

---

## 1. System Overview

**PharmaTrace AI** is a Streamlit-based pharmaceutical supply chain intelligence dashboard. It combines machine learning, rule-based clinical logic, and interactive visualizations to give top management actionable insights across the full pharmaceutical supply chain lifecycle.

### Application Entry Point

```
streamlit run streamlit_app.py
```

The app loads via `get_data(uploaded_file)` which parses the uploaded Excel file and populates all DataFrames. If no file is uploaded, synthetic fallback data is used automatically.

### Navigation Structure

```
Sidebar Navigation
│
├── 🏠 Overview & Data Upload
├── 🤖 ML Expiry Classifier          ← Engine 1
├── 🌐 Network Rebalancing & Transfers ← Engine 2
├── 📈 Demand & Seasonality          ← Engine 3
├── 🔄 Reverse Logistics & Certified Disposal ← Engine 4
├── 🏭 Warehouse & Supply Intelligence
├── 📦 Batch Traceability
└── 📋 Compliance & Audit
```

---

## 2. Architecture & Data Flow

### Startup Sequence

```
1. App start → load CSS + global styles
2. Sidebar renders → user selects page
3. get_data(uploaded_file) → parse Excel → return (inventory, products, warehouses, ...)
4. Selected page block executes → computes engine-specific features
5. Charts render → st.cache_data returns cached results where available
6. AI insight cards rendered with live computed variables
```

### Key Global Variables (available across all pages)

| Variable | Type | Description |
|----------|------|-------------|
| `inventory` | DataFrame | Core batch-level stock data |
| `products` | DataFrame | Product master (SKU, price, shelf life) |
| `warehouses` | DataFrame | Warehouse/DC master |
| `df_demand` | DataFrame | Monthly demand data |
| `df_freight` | DataFrame | Inter-warehouse freight matrix |
| `df_txns` | DataFrame | Shipment transaction history |
| `extended_tables` | Dict[str, DataFrame] | All other sheets from uploaded Excel |
| `curr_code` | str | Currency code (default: "USD") |
| `curr_sym` | str | Currency symbol (default: "$") |

---

## 3. Engine 1 — ML Expiry Classifier

### Purpose
Identify which inventory batches will expire before being fully sold, triage them by urgency using a clinical 4-color RAG matrix, and generate prescriptive recovery actions with quantified financial impact.

### Tabs

| Tab | Name | Key Content |
|-----|------|-------------|
| Tab 1 | RAG Matrix Dashboard | 4-color zone breakdown, KPI metrics, action cards per batch, AI insight |
| Tab 2 | Expiry Risk Prediction | ML model tournament, demand scenario, SKU Pareto, warehouse risk |
| Tab 3 | Recovery Playbook | Zone-by-zone recovery actions with $ recovery estimates |

### Data Inputs
- `inventory` — `days_to_expiry`, `quantity_on_hand`, `unit_price`, `shelf_life_days`, `warehouse_id`, `product_id`
- `df_txns` — OUTBOUND_DISPATCH_PICK transactions for velocity calculation

### Feature Engineering

```python
ml_df["avg_monthly_dispatch"]   = total_dispatched / 24
ml_df["cover_days"]             = (quantity_on_hand / avg_monthly_dispatch * 30).clip(0, 9999)
ml_df["risk_score"]             = (days_to_expiry / shelf_life_days).clip(0, 1)
ml_df["value_per_day"]          = inventory_value_usd / days_to_expiry.clip(1)
ml_df["velocity_pressure"]      = (cover_days / days_to_expiry.clip(1)).clip(0, 10)
ml_df["capital_velocity_ratio"] = (value_per_day / avg_monthly_dispatch).clip(0, 9999)
ml_df["pct_life_remaining"]     = (days_to_expiry / shelf_life_days).clip(0, 1)
ml_df["shelf_life_consumed_pct"]= 1 - pct_life_remaining
```

### RAG Zone Assignment (`assign_rag` function)

```python
def assign_rag(row):
    dte = row["days_to_expiry"]
    if dte <= 0:          return "🔴 Red (<3M / Expired)"
    elif dte <= 90:       return "🔴 Red (<3M / Expired)"
    elif dte <= 180:      return "🟠 Amber (4-6M)"
    elif dte <= 365:      return "🟡 Yellow (7-12M)"
    else:                 return "🟢 Green (>12M)"
```

### Action Planning Matrix

| RAG Zone | Days to Expiry | Action |
|----------|---------------|--------|
| 🔴 Red | ≤ 0 (expired) | Certified disposal — EPA/DEA manifest mandatory |
| 🔴 Red | 1–90 days | Emergency liquidation or certified destruction |
| 🟠 Amber | 91–180 days | Expedite dispatch; inter-warehouse transfer to high-velocity node |
| 🟡 Yellow | 181–365 days | Velocity sprint; promotional pricing if needed |
| 🟢 Green | > 365 days | Standard FEFO dispatch — no immediate action |

### Demand Scenario Analysis

Three toggleable scenarios stress-test the classifier:
- **Pessimistic (−1.5σ):** Demand drops ~37.5% (disease decline, tender loss)
- **Base Case:** 24-month historical average dispatch velocity
- **Optimistic (+1σ):** Demand rises ~25% (outbreak, new tender win)

The scenario adjusts `_scenario_vel`, recomputes `_scen_cover` and `_scen_vp`, then shows how many additional batches become at-risk under that assumption.

### ML Model Tournament

```python
models = {
    "Logistic Regression": LogisticRegression(max_iter=400, class_weight="balanced"),
    "Random Forest":       RandomForestClassifier(n_estimators=120, max_depth=9, class_weight="balanced"),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, learning_rate=0.08)
}
# Champion selected by: champion = max(results, key=lambda k: results[k]["f1"])
```

### Recovery Playbook Zones

| Zone | Strategy | Recovery Rate |
|------|---------|--------------|
| 🔴 Expired | Certified destruction | 0% recovery |
| 🟠 Critical (<90d) | Secondary market liquidation | ~35% of book value |
| 🟡 At-Risk (90–180d) | Inter-warehouse transfer + velocity programs | ~65% |
| 🔵 Monitor (180–365d) | Standard acceleration | ~85% |
| 🟢 Safe (>365d) | Normal FEFO dispatch | ~100% |

### AI Insight Card

**Title:** "Expiry Risk Intelligence — What the Data Is Telling Management"

Dynamic bullets computed from:
- `_top3_sku` — top 3 SKUs by at-risk inventory value (from live data)
- `_avg_vp_risk` — average velocity pressure of at-risk batches
- `_vel_gap` — velocity gap ratio between at-risk and safe batches
- `_expired_cnt`, `_expired_val` — batches already past expiry date

---

## 4. Engine 2 — Network Rebalancing & Transfers

### Purpose
Identify geographical imbalances in the distribution network (surplus "cold" warehouses vs. demand-deficit "hot" warehouses) and generate batch-level, freight-cost-aware inter-warehouse transfer recommendations.

### Tabs

| Tab | Name | Key Content |
|-----|------|-------------|
| Tab 1 | Network Overview | HOT/COLD detection, heatmaps, fill rate analysis |
| Tab 2 | Transfer Recommender | ML batch-level transfer manifest, cost comparison |

### Data Inputs
- `df_demand` — `warehouse_id`, `product_id`, `quantity_demanded_units`, `quantity_dispatched_units`
- `inventory` — `warehouse_id`, `product_id`, `quantity_on_hand`, `inventory_value_usd`, RAG zones
- `df_freight` — `origin_warehouse_id`, `dest_warehouse_id`, `freight_cost_per_unit`

### HOT/COLD Detection Logic

```python
# Built inside build_network_data() — cached with @st.cache_data
geo["days_of_stock"] = stock_on_hand / (avg_monthly_demand / 30)

hot  = geo[geo["days_of_stock"] < 30]   # Stockout risk
cold = geo[geo["days_of_stock"] > 120]  # Surplus / write-off risk
```

### Transfer Scoring Logic

```python
for each (surplus_wh, product) in cold_locations:
    for each (deficit_wh, product) in hot_locations:
        freight_cost = transfer_qty × freight_rate
        net_savings  = batch_value_rescued - freight_cost
        clearance_days = transfer_qty / dest_monthly_demand × 30
        runway_margin  = batch_dte - clearance_days
        → Approved if: net_savings > 0 AND runway_margin > 0
```

### Transfer Recommendation Card Fields

| Field | Description |
|-------|-------------|
| Batch ID | `fp_batch_id` of the at-risk batch |
| Product | `generic_name` |
| Origin DC | Source warehouse with surplus |
| Destination DC | Deficit warehouse needing stock |
| Transfer Qty | Units recommended to move |
| DTE | Days to Expiry at transfer time |
| RAG Zone | Current RAG classification |
| Freight Cost | Total freight expense |
| Net Savings | Rescued value minus freight |
| Clearance Days | Time to sell at destination velocity |
| Runway Margin | DTE minus clearance days (safety buffer) |

### Cost Comparison: Transfer vs. Manufacturing

```
Comparison shows:
  Transfer Cost  = Freight rate × Transfer units
  Manufacturing  = Estimated CMO cost per unit × Same quantity
→ Transfer is always faster (2–4 days vs. 3–6 weeks CMO lead time)
→ Transfer is cheaper when Net Savings > 0
```

### AI Insight Card

**Title:** "Network Rebalancing & Supply Chain Economics"

Dynamic bullets:
- `_hot_cnt` — count of demand hotspot warehouses
- `_cold_cnt` — count of surplus cold warehouses
- `tot_sav` — total network savings from rebalancing
- `tot_viable_trans` — transferable batch count
- `tot_net_rescued` — total rescued inventory value

---

## 5. Engine 3 — Demand & Seasonality

### Purpose
Provide forward-looking demand forecasts (1M / 3M / 6M) per SKU, classify demand into clinical patterns, and generate procurement action plans ensuring no over-ordering against already-at-risk inventory.

### Tabs

| Tab | Name | Key Content |
|-----|------|-------------|
| Tab 1 | Demand Patterns & Forecasts | XGBoost forecasts, pattern chart, top movers |
| Tab 2 | Procurement Action Plan | Net order quantities, at-risk SKU block list |
| Tab 3 | Warehouse & Supply Intelligence | Facility-level demand intelligence |

### Data Flow

```
uploaded_file (monthly_demand sheet)
    ↓
demand_prediction.py → train_model()
    ↓
XGBoost models (3 × horizon)
    ↓
demand_model_cache.pkl + demand_forecast_results.xlsx
    ↓
streamlit_app.py loads cache → _df_forecasts
```

### XGBoost Features (per SKU/WH pair)

```python
features = [
    "lag_1", "lag_2", "lag_3",        # 1, 2, 3-month lagged demand
    "rolling_mean_3", "rolling_std_3", # 3-month rolling statistics
    "month_sin", "month_cos",          # Cyclical month encoding
    "product_encoded",                 # Label-encoded product ID
    "warehouse_encoded"                # Label-encoded warehouse ID
]
```

### 7 Clinical Demand Patterns

| Pattern | Classification Rule | Typical Products |
|---------|---------------------|-----------------|
| Winter Surge Respiratory | Peak demand Nov–Feb, trough Jun–Aug | Antivirals, cough syrups |
| Seasonal Allergy | Peak demand Mar–Jun | Antihistamines, nasal sprays |
| Chronic Maintenance | CV < 15%, flat year-round | Antihypertensives, statins |
| Controlled Substance | Stable + DEA schedule flag | Opioids, benzodiazepines |
| Specialty Oncology | High variance, low volume | Chemotherapy agents |
| Biologic / Specialty | High price, cold-chain required | Monoclonal antibodies |
| Generic / OTC | High volume, price-elastic | Paracetamol, ibuprofen |

### Procurement Formula

```
Net Order = max(0, Gross_Forecast_6M × (1 + Safety_Buffer_18%)
                   - Usable_On_Hand
                   - In_Transit_Orders)

Where:
  Usable_On_Hand = Inventory with DTE > 90 days (Amber/Green/Yellow only)
  Safety_Buffer  = 18% standard pharmaceutical buffer
```

### Monotonic Enforcement

```python
# Prevents forecasts from "shrinking" at longer horizons (clinically unrealistic)
df_forecasts["forecast_3m"] = df_forecasts[["forecast_1m", "forecast_3m"]].max(axis=1)
df_forecasts["forecast_6m"] = df_forecasts[["forecast_3m", "forecast_6m"]].max(axis=1)
```

### AI Insight Cards

**Demand Patterns Card:** "Demand Patterns & Strategic Procurement Intelligence"
- Dominant pattern name and share %
- Winter surge ratio (Nov–Feb vs. off-season)
- Top growing and top declining SKU names + values

**Warehouse Intelligence Card:** "Warehouse & Supply Chain Intelligence"
- Top warehouse type by volume
- Delay rate (unfulfilled demand %)
- 3-month demand trend direction

Both fully dynamic — computed from `_df_forecasts` and `_src3` DataFrames at render time.

---

## 6. Engine 4 — Reverse Logistics & Certified Disposal

### Purpose
Track the full lifecycle of returned, recalled, and disposed pharmaceutical inventory — from RMA intake through FDA recall root-cause analysis, defect clustering, and predictive batch recall risk modeling.

### Tabs

| Tab | Name | Key Content |
|-----|------|-------------|
| Tab 1 | Operations & Disposal Audit | RMA charts, disposal methods, financial leakage metrics, EPA disposal AI |
| Tab 2 | Recall Intelligence, Root-Causes & Defect AI | 4-stage staging, 1-10-100 table, K-Means clustering, Isolation Forest |
| Tab 3 | New Batch Recall Risk Predictor | Random Forest recall probability simulator |

### Data Inputs

| Table | Required Columns |
|-------|-----------------|
| `returns` | `return_id`, `fp_batch_id`, `warehouse_id`, `return_reason`, `quantity`, `return_date` |
| `disposal` | `disposal_id`, `fp_batch_id`, `warehouse_id`, `disposal_reason`, `quantity`, `disposal_method`, `certificate_document_id` |
| `recalls` | `recall_id`, `product_id`, `classification`, `reason_for_recall` |
| `finished_product_batches` | `fp_batch_id`, `product_id`, `batch_qty`, `mo_id` |
| `manufacturing_orders` | `mo_id`, `planned_qty`, `produced_qty` |

> If any table is empty, realistic synthetic fallback data is auto-generated with `np.random.default_rng()`.

### Tab 1 — Operations & Disposal Audit

**Charts:**
1. Customer & Hospital Return Reasons (RMA Influx) — horizontal bar chart
2. Certified Destruction Methods (EPA/DEA) — vertical bar chart

**Financial Leakage Radar:**
- Transit Breakage (carrier penalties claimable)
- Customer Overstock (hospital restock fee)
- Regulatory Returns (credit note authorized)
- Total Return Value (capital tied in pipeline)
- EPA Destruction Cost (~8% of return value)

**Expiry Risk → Returns Loop Closure:**
If RAG Amber/Red batches correlate with returns, a dynamic callout shows:
> "X.X% of all customer returns originated from Amber/Red RAG zone batches"

**EPA/DEA Disposal AI Recommender:**
- Trained Random Forest on `disposal_method` (multi-class: incineration / witnessed_incineration / chemical_neutralization)
- Input: disposal reason, warehouse, quantity, dosage form, unit price
- Output: Mandated EPA method + confidence % + compliance requirement

### Tab 2 — Recall Intelligence

#### Supply Chain Stage Mapping

```python
def map_recall_stage(txt):
    t = txt.lower()
    if any(k in t for k in ['impurity','nitrosamine','ndma','api','chemical','analytical']):
        return 'Stage 1: Raw Material & API Sourcing'
    elif any(k in t for k in ['label','packaging','barcode','ndc','seal']):
        return 'Stage 3: Secondary Packaging & Labeling'
    elif any(k in t for k in ['temperature','excursion','cold-chain','transit']):
        return 'Stage 4: Cold-Chain & Downstream Logistics'
    else:
        return 'Stage 2: Manufacturing & Formulation (cGMP)'
```

#### 1-10-100 Quality Cost Principle Table

| Stage | Cost to Intervene | Avoided Recall Loss | ROI Multiple |
|-------|------------------|--------------------|----|
| Stage 1 — Sourcing | $2,500 | $420,000 | 168× |
| Stage 2 — cGMP Formulation | $6,500 | $310,000 | 48× |
| Stage 3 — Packaging | $1,500 | $85,000 | 56× |
| Stage 4 — Cold-Chain | $3,000 | $120,000 | 40× |

**Interactive Intercept Calculator:** User inputs batch qty + unit price → app computes unmitigated loss vs. intercept cost → displays net value saved and ROI multiple dynamically.

#### K-Means NLP Defect Clustering

```python
vec      = TfidfVectorizer(max_features=50, stop_words="english")
tfidf    = vec.fit_transform(recalls_df["reason_for_recall"])
km       = KMeans(n_clusters=4, random_state=42, n_init=10)
clusters = km.fit_predict(tfidf)

# Cluster → Archetype mapping:
{0: "Sterile & Particulate (Class I)",
 1: "Dissolution & Potency (Class II)",
 2: "Packaging & Labeling (Class III)",
 3: "Chemical Impurities / NDMA (Class II)"}
```

#### Isolation Forest — Batch Yield Anomaly

```python
iso = IsolationForest(contamination=0.035, random_state=42)
df_ano["yield_variance"] = (produced_qty - planned_qty) / planned_qty * 100
iso.fit(df_ano[["yield_variance", "batch_qty"]])
# Output: -1 = Anomaly (3.5% flagged) | 1 = Normal
```

All flagged batches go to a quarantine watchlist table with action: "🚨 HOLD — Initiate In-Process Assay Re-Check"

### Tab 3 — New Batch Recall Risk Predictor

```python
# Feature set
X_b = concat([
    df_b[["batch_qty", "yield_variance", "unit_price", "shelf_life_months"]],
    get_dummies(df_b[["dosage_form", "manufacturer_id"]])
])
y_b = df_b["recall_flag"].astype(int)

clf_batch = RandomForestClassifier(n_estimators=60, max_depth=7, random_state=42)
clf_batch.fit(X_tr_b, y_tr_b)
```

**Risk Banners:**
- > 60% probability → 🚨 CRITICAL RECALL HAZARD DETECTED
- 30–60% → ⚠️ MODERATE RECALL RISK
- < 30% → ✅ LOW RECALL RISK

**Suspected Root-Cause Stage** is derived from top feature importance and input values — not hardcoded.

**Value Saved** = `(batch_qty × unit_price) + freight + legal fees + EPA disposal - intercept cost`

### AI Insight Card

**Title:** "Reverse Logistics, Recall Root-Cause Staging & Batch Predictive AI"

Dynamic bullets:
- `_recall_pct` — recall-driven % of all RMAs
- `stage_counts` — events per supply chain stage (from actual recall data)
- `acc_b` — Random Forest batch recall accuracy
- `auc_b` — ROC-AUC score
- `len(df_b)` — actual training batch count

---

## 7. Shared UI Components

### `ai_insight(title, bullets, icon, color)` function

Renders a premium dark-themed AI insight card with:
- Gradient header with icon + title
- Scrollable bullet list (each bullet can be HTML-formatted)
- Expandable/collapsible via `st.expander`

### `fmt_curr(value, compact, decimals, code)` function

Formats a float as currency:
```python
fmt_curr(1_500_000)              → "$1.50M"
fmt_curr(85000, compact=False)   → "$85,000"
fmt_curr(450, decimals=2)        → "$450.00"
```

### `show_fig(fig)` function

Renders a matplotlib figure to Streamlit with dark background preservation and `st.pyplot(fig)`.

### Global CSS

Applied via `st.markdown(GLOBAL_CSS, unsafe_allow_html=True)`. Key classes:
- `.section-header` — violet gradient page title bar
- `.section-desc` — slate-coloured subtitle below header

---

## 8. Data Schema Reference

### `inventory` (Core Table)

| Column | Type | Description |
|--------|------|-------------|
| `batch_number` | str | Unique batch identifier |
| `product_id` | str | FK → products |
| `warehouse_id` | str | FK → warehouses |
| `quantity_on_hand` | int | Current physical stock |
| `expiry_date` | date | Batch expiry date |
| `manufacturing_date` | date | Batch manufacturing date |
| `unit_price` | float | Price per unit (USD) |
| `inventory_value_usd` | float | quantity × unit_price |
| `days_to_expiry` | int | Computed: expiry_date − today |
| `shelf_life_days` | int | Total shelf life in days |
| `generic_name` | str | Drug name (joined from products) |
| `rag_status` | str | Computed RAG zone label |

### `monthly_demand` (Engine 3)

| Column | Type | Description |
|--------|------|-------------|
| `warehouse_id` | str | FK → warehouses |
| `product_id` | str | FK → products |
| `month` | date | Month start date |
| `quantity_demanded_units` | int | Total units ordered |
| `quantity_dispatched_units` | int | Total units shipped |

### `returns` (Engine 4)

| Column | Type | Description |
|--------|------|-------------|
| `return_id` | str | Unique RMA ID |
| `return_no` | str | Human-readable RMA number |
| `fp_batch_id` | str | FK → finished_product_batches |
| `warehouse_id` | str | Return destination DC |
| `return_reason` | str | recall / expired / damaged / overstock / quality_issue |
| `quantity` | int | Units returned |
| `return_date` | date | Date of return receipt |

---

## 9. Models Reference

| Model | Engine | Library | Type | Key Parameters |
|-------|--------|---------|------|---------------|
| Random Forest (Expiry) | Engine 1 | scikit-learn | Classification | n_estimators=120, max_depth=9, class_weight="balanced" |
| Gradient Boosting (Expiry) | Engine 1 | scikit-learn | Classification | n_estimators=100, learning_rate=0.08 |
| Logistic Regression (Expiry) | Engine 1 | scikit-learn | Classification | max_iter=400, class_weight="balanced" |
| XGBoost (Demand) | Engine 3 | xgboost | Regression | Per-horizon; n_estimators=200, learning_rate=0.05 |
| K-Means (Recall Clustering) | Engine 4 | scikit-learn | Clustering | n_clusters=4, n_init=10, random_state=42 |
| Isolation Forest (Anomaly) | Engine 4 | scikit-learn | Anomaly Detection | contamination=0.035, random_state=42 |
| Random Forest (Recall Risk) | Engine 4 | scikit-learn | Classification | n_estimators=60, max_depth=7 |
| Random Forest (Disposal) | Engine 4 | scikit-learn | Multi-class | n_estimators=100, max_depth=8 |

---

## 10. Regulatory Compliance Mapping

| Regulation | Applicable Engine | Implementation |
|-----------|-----------------|----------------|
| FDA 21 CFR §211.142 | Engine 1, 4 | Red zone quarantine action cards |
| FDA 21 CFR §211.150 | Engine 1, 2 | FEFO dispatch logic |
| FDA 21 CFR §211.160 | Engine 4 | Expired stock compliance alert |
| DSCSA (Drug Supply Chain Security Act) | Engine 1 | 180-day RSL threshold |
| EPA RCRA §264 | Engine 4 | 8% destruction cost applied; disposal method AI |
| DEA Title 21 CFR | Engine 4 | Witnessed incineration flag for high-value/parenteral |
| ICH Q10 (Pharmaceutical Quality System) | Engine 1, 4 | Batch anomaly detection pre-release hold |

---

## 11. Known Limitations

| Limitation | Impact | Workaround |
|-----------|--------|-----------|
| Synthetic data fallback | Forecasts/models trained on simulated data may not reflect client's actual patterns | Upload real Excel data |
| XGBoost needs ≥ 12 months history | Shorter history → poor generalization | Pre-load 12+ months before deployment |
| Isolation Forest contamination fixed at 3.5% | May over/under-flag for different manufacturing environments | Expose as a configurable parameter in future |
| K-Means cluster labels are manually mapped | If recall text patterns shift, re-mapping needed | Retrain and remap when new data volume > 500 events |
| No real-time data feed | Dashboard shows snapshot at upload time | Re-upload updated Excel to refresh |
| Single-file upload | All data in one Excel; schema must match template exactly | Validate against PharmaTrace_Data_Template.xlsx |

---

## 12. Future Enhancements

| Priority | Enhancement | Engine Affected |
|----------|------------|----------------|
| High | Real-time ERP (SAP/Oracle) API connector | All |
| High | Live IoT cold-chain temperature dashboard | Engine 4 |
| High | Automated Red-zone email/SMS alerts | Engine 1 |
| Medium | NLP query interface ("Show expiring batches in WH003") | All |
| Medium | PDF board report auto-generation | All |
| Medium | Configurable Isolation Forest contamination rate | Engine 4 |
| Low | Multi-language support | UI |
| Low | Mobile-responsive layout optimization | UI |
| Research | Reinforcement learning for dynamic FEFO optimization | Engine 2 |

---

*PharmaTrace AI | ISB AMPBA Capstone 2026 | Sponsor: Innodatatics Inc.*
*Complete Technical Documentation — Final Submission*
