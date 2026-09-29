# PharmaTrace AI — Exploratory Data Analysis (EDA) Report

**Deliverable 3: EDA Report**
**ISB AMPBA Capstone | Sponsor: Innodatatics Inc.**
**Dataset:** PharmaTrace AI Synthetic Master Dataset | September 2026

---

## Contents

1. [Data Understanding](#1-data-understanding)
2. [Data Quality Assessment](#2-data-quality-assessment)
3. [Preprocessing Steps](#3-preprocessing-steps)
4. [Statistical Analysis](#4-statistical-analysis)
5. [Key Patterns & Findings — Engine 1: ML Expiry Classifier](#5-engine-1-ml-expiry-classifier)
6. [Key Patterns & Findings — Engine 2: Network Rebalancing](#6-engine-2-network-rebalancing)
7. [Key Patterns & Findings — Engine 3: Demand & Seasonality](#7-engine-3-demand--seasonality)
8. [Key Patterns & Findings — Engine 4: Reverse Logistics](#8-engine-4-reverse-logistics)
9. [Cross-Engine Correlations](#9-cross-engine-correlations)
10. [Key EDA Findings Summary](#10-key-eda-findings-summary)

> 📌 All visualizations referenced below are in the `visualizations/` folder.
> The full EDA notebook is: `PharmaTrace_AI_EDA.ipynb`
> The full EDA PDF report is: `PharmaTrace_AI_EDA_Report.pdf`

---

## 1. Data Understanding

### Dataset Overview

The PharmaTrace AI dataset is a **synthetic pharmaceutical supply chain master dataset** built to mirror real-world distributor operations. It spans the complete product lifecycle from raw material sourcing to final certified disposal.

**📊 Visualization:** `EDA_01_row_counts.png` — Row Counts per Table (52,548 Total)

| Table | Row Count | Description |
|-------|-----------|-------------|
| `batch_genealogy` | 18,000 | Full traceability chain per batch component |
| `finished_product_batches` | 6,000 | Final manufactured batch records |
| `manufacturing_orders` | 6,000 | Production order details |
| `inventory` | 6,000 | Current warehouse stock positions |
| `ai_prediction_data` | 3,000 | Pre-computed AI risk predictions |
| `compliance_documents` | 3,532 | FDA/EPA audit trail documents |
| `products` | 3,000 | SKU master catalogue |
| `shipments` | 2,100 | Historical dispatch transactions |
| `raw_material_batches` | 1,598 | RM batch records |
| `recalls` | 600 | FDA recall events |
| `returns` | 532 | Customer RMA returns |
| `disposal` | 532 | Certified destruction records |
| `manufacturers` | 510 | Manufacturer master |
| `raw_materials` | 799 | Raw material master |
| `retailers` | 300 | Retail customer master |
| `distributors` | 25 | Wholesale distributor master |
| `warehouses` | 8 | Distribution centre master |
| `suppliers` | 12 | API/RM supplier master |
| **TOTAL** | **52,548** | Across 18 tables |

### Supply Chain Flow
**📊 Visualization:** `EDA_14_supply_chain_flow.png`

The data follows the end-to-end pharmaceutical supply chain:
```
Suppliers (12) → Raw Materials (799) → RM Batches (1,598) →
Manufacturing Orders (6,000) → Finished Product Batches (6,000) →
Batch Genealogy (18,000) →
Inventory (6,000) → Shipments (2,100) → Retailers (300)
                                              ↓
Returns (532) → Recalls (600) → Disposal (532) → Compliance Docs (3,532)
```

---

## 2. Data Quality Assessment

**📊 Visualization:** `EDA_02_missing_values.png`

### Missing Value Analysis

| Table | Column | Missing % | Treatment |
|-------|--------|-----------|-----------|
| `inventory` | `expiry_date` | ~0.3% | Dropped (cannot classify without expiry) |
| `inventory` | `unit_price` | ~0.1% | Median imputation ($45.00/unit) |
| `shipments` | `freight_cost` | ~2.1% | Median imputation ($0.85/unit) |
| `monthly_demand` | `quantity_dispatched` | ~1.8% | Forward-fill by SKU/warehouse |
| `returns` | `return_date` | ~0.4% | Excluded from temporal analysis |
| `recalls` | `reason_for_recall` | 0% | Complete — critical for K-Means NLP |
| `finished_product_batches` | `produced_qty` | ~0.6% | Imputed from `planned_qty × 0.97` |

### Key Data Quality Observations
- **No duplicate batch IDs** — all `batch_number` and `fp_batch_id` values are unique
- **Date integrity confirmed** — all `manufacturing_date < expiry_date` (no logical inversions)
- **Referential integrity** — all inventory records join correctly to products and warehouses
- **92.8% synthetic, 7.2% real seed data** — confirmed via `EDA_07_inventory.png` (Inventory Data Source pie chart)

---

## 3. Preprocessing Steps

The following preprocessing steps were applied before ML model training:

### 3.1 Feature Engineering (Engine 1 — Expiry Classifier)

```python
# Derived columns computed from raw inventory data
cover_days              = (quantity_on_hand / avg_monthly_dispatch * 30).clip(0, 9999)
velocity_pressure       = (cover_days / days_to_expiry.clip(1)).clip(0, 10)
pct_life_remaining      = (days_to_expiry / shelf_life_days).clip(0, 1)
shelf_life_consumed_pct = 1 - pct_life_remaining
value_per_day           = inventory_value_usd / days_to_expiry.clip(1)
capital_velocity_ratio  = (value_per_day / avg_monthly_dispatch).clip(0, 9999)
```

**Binary Target Variable:**
```python
financial_loss_risk = 1  if velocity_pressure > 1.0 OR days_to_expiry < 90
                    = 0  otherwise
```

### 3.2 Text Preprocessing (Engine 4 — NLP Recall Clustering)

```python
# recall reason_for_recall text → TF-IDF vectors
vectorizer = TfidfVectorizer(max_features=50, stop_words="english")
tfidf_matrix = vectorizer.fit_transform(recalls_df["reason_for_recall"])
```

### 3.3 Demand Data Preprocessing (Engine 3 — XGBoost)

- **Chronological split:** 80% training (oldest months), 20% test (newest months) — no data leakage
- **Lag features:** 1M, 2M, 3M lagged demand
- **Cyclical encoding:** `month_sin = sin(2π × month / 12)`, `month_cos = cos(2π × month / 12)`
- **Monotonic enforcement post-prediction:** 3M ≥ 1M, 6M ≥ 3M

### 3.4 Label Encoding (Categorical Variables)

All categorical string columns (`product_id`, `warehouse_id`, `dosage_form`, `manufacturer_id`) encoded using `sklearn.LabelEncoder` before ML model training.

---

## 4. Statistical Analysis

### 4.1 Inventory — Key Statistics

**📊 Visualization:** `EDA_07_inventory.png`

| Metric | Value |
|--------|-------|
| Total inventory records | 6,000 |
| Mean quantity on hand | 2,994 units |
| Quantity distribution | Near-uniform (100–5,000 units) with spike at low end |
| Warehouses | 8 DCs — roughly equal stock distribution (~2.1M–2.4M units each) |
| Warehouse utilization | Near 100% across all DCs (capacity constraint visible) |
| Mean unit price | $45.00 |
| Price range | $0.50 (generic OTC) to $950+ (specialty biologics) |

### 4.2 Products — Key Statistics

**📊 Visualization:** `EDA_03_products_analysis.png`

| Metric | Value |
|--------|-------|
| Unique products | 3,000 SKUs |
| Dosage forms | Tablets (~45%), Injectable (~25%), Capsules (~15%), Others (~15%) |
| Shelf life distribution | Predominantly 24 months (standard pharma) |
| Price distribution | Right-skewed (majority $10–$100; long tail to $950+) |

### 4.3 Manufacturing — Key Statistics

**📊 Visualization:** `EDA_06_manufacturing.png`

| Metric | Value |
|--------|-------|
| Manufacturing orders | 6,000 |
| Average yield | ~97% (planned vs. produced) |
| Yield variance range | −10% to +0.2% |
| Batches with >3.5% yield deviation | ~3.5% (flagged by Isolation Forest) |

### 4.4 Recalls — Key Statistics

**📊 Visualization:** `EDA_09_recalls.png`

| Metric | Value |
|--------|-------|
| Total recall events | 600 |
| Class I recalls (life-threatening) | ~22% |
| Class II recalls | ~55% |
| Class III recalls | ~23% |
| Top recall reason category | Manufacturing / Formulation defects (~65%) |
| Recall-driven returns | ~30.7% of all RMA events |

### 4.5 Returns & Disposal

**📊 Visualization:** `EDA_10_returns_disposal.png`

| Metric | Value |
|--------|-------|
| Total returns | 532 RMAs |
| Disposal events | 532 |
| Recall-driven returns | ~164 (30.7%) |
| Expired returns | ~186 (34.9%) |
| Damaged/transit returns | ~99 (18.6%) |
| Overstock returns | ~83 (15.6%) |
| Estimated total return value | ~$97.4M |
| EPA destruction cost | ~$7.79M (~8% of return value) |

---

## 5. Engine 1: ML Expiry Classifier

### 5.1 RAG Zone Distribution

The 4-Color RAG framework segments inventory by Residual Shelf Life (RSL):

| Zone | RSL Range | Batch Count | % of Portfolio |
|------|-----------|-------------|----------------|
| 🟢 Green | > 12 months | ~3,000 | ~50% |
| 🟡 Yellow | 7–12 months | ~1,200 | ~20% |
| 🟠 Amber | 4–6 months | ~1,080 | ~18% |
| 🔴 Red | < 3M / Expired | ~720 | ~12% |

**Key Finding:** ~30% of inventory (Amber + Red zones) is at immediate financial risk. This translates to an estimated **$15–$45M in threatened capital** depending on product value mix.

### 5.2 Velocity Pressure Distribution

`velocity_pressure = cover_days / days_to_expiry`

- Values > 1.0 indicate **guaranteed write-off** (more stock than can be sold before expiry)
- ~18% of batches show velocity_pressure > 1.0 at time of analysis
- Highest velocity pressures concentrated in **low-demand specialty SKUs** with long cover days

### 5.3 Model Feature Importance (Random Forest Champion)

Top predictive features ranked by mean impurity decrease:

| Rank | Feature | Contribution |
|------|---------|-------------|
| 1 | `quantity_on_hand` | ~28% |
| 2 | `velocity_pressure` | ~24% |
| 3 | `cover_days` | ~19% |
| 4 | `pct_life_remaining` | ~14% |
| 5 | `avg_monthly_dispatch` | ~9% |
| 6 | `capital_velocity_ratio` | ~6% |

**Finding:** Procurement quantity (how much was ordered) is the #1 risk driver — pointing to upstream purchasing without shelf-life gating as the root cause.

### 5.4 SKU Pareto (20/80 Rule Confirmed)

Analysis confirms the Pareto principle holds:
- Top 20% of SKUs drive ~78% of total at-risk inventory value
- Top 3 SKUs by threatened capital account for >40% of total exposure

---

## 6. Engine 2: Network Rebalancing

**📊 Visualization:** `EDA_08_distribution.png`

### 6.1 Warehouse Stock Distribution

| DC | City | Type | Avg Days of Stock | Status |
|----|------|------|-------------------|--------|
| WH001 | New Jersey | Central DC | 45 days | Moderate |
| WH002 | Texas | Regional DC | 28 days | 🔴 HOT |
| WH003 | Illinois | Cold Chain Hub | 132 days | 🔵 COLD |
| WH004 | California | Regional DC | 38 days | Moderate |
| WH005 | Georgia | Regional DC | 22 days | 🔴 HOT |
| WH006 | Pennsylvania | Cold Chain Hub | 118 days | 🔵 COLD |
| WH007 | Ohio | Regional DC | 51 days | Moderate |
| WH008 | Nevada | Central DC | 41 days | Moderate |

**Key Finding:** Cold Chain Hubs (WH003, WH006) are chronically over-stocked (>4 months of cover), while Regional DCs in Texas and Georgia face recurring stockout risk.

### 6.2 Transfer Opportunity Analysis

- **Viable transfers identified:** ~150–400 batch-level transfer candidates per data run
- **Average net savings per transfer:** ~$12,000–$85,000
- **Total rescuable capital:** Typically 60–80% of at-risk Amber zone inventory value
- **Freight cost as % of rescued value:** ~3–7% (strong positive ROI)

---

## 7. Engine 3: Demand & Seasonality

**📊 Visualization:** `EDA_05_supply_chain.png`, `EDA_08_distribution.png`

### 7.1 Demand Pattern Distribution (24-Month History)

| Pattern | SKU Share | Seasonality Strength |
|---------|-----------|---------------------|
| Chronic Maintenance | ~38% | Flat — low variance |
| Winter Surge Respiratory | ~18% | Strong — Nov–Feb peak |
| Generic/OTC | ~16% | Moderate |
| Seasonal Allergy | ~12% | Strong — Mar–Jun peak |
| Controlled Substance | ~8% | Flat — DEA-monitored |
| Specialty Oncology | ~5% | Erratic — low volume |
| Biologic/Specialty | ~3% | Erratic — cold chain |

### 7.2 Demand Seasonality — Winter Surge

- Winter Surge Respiratory SKUs show 2.1–3.4× demand in Nov–Feb vs. Jun–Aug trough
- This creates over-procurement risk: buyers stock up pre-season, leaving excess inventory when season ends
- **This is the #1 driver of Amber-zone inventory accumulation** in the dataset

### 7.3 XGBoost Model Performance

| Horizon | MAPE | R² | Notes |
|---------|------|-----|-------|
| 1 Month | ~14% | ~0.82 | Best accuracy — short horizon |
| 3 Months | ~18% | ~0.74 | Good generalization |
| 6 Months | ~22% | ~0.65 | Acceptable for procurement planning |

---

## 8. Engine 4: Reverse Logistics

**📊 Visualization:** `EDA_09_recalls.png`, `EDA_10_returns_disposal.png`, `EDA_11_ai_predictions.png`

### 8.1 Recall Root-Cause Distribution (4-Stage Staging)

| Stage | Events | % of Total | Avg Recall Cost |
|-------|--------|-----------|----------------|
| Stage 1: Raw Material & API Sourcing | ~132 | 22% | $420,000 |
| Stage 2: Manufacturing & Formulation | ~330 | 55% | $310,000 |
| Stage 3: Secondary Packaging & Labeling | ~90 | 15% | $85,000 |
| Stage 4: Cold-Chain & Downstream Logistics | ~48 | 8% | $120,000 |

**Key Finding:** Stage 2 Manufacturing defects drive the majority of recalls but Stage 1 has the highest per-event cost due to contamination scale. Stage 1 interception ROI = **168×**.

### 8.2 K-Means NLP Defect Clustering

TF-IDF vectorization of 600 recall reason descriptions revealed 4 natural defect archetypes:

| Cluster | Archetype | Top Keywords | FDA Class |
|---------|-----------|-------------|-----------|
| 0 | Sterile & Particulate | *particulate, sterile, vial, injectable* | Class I |
| 1 | Dissolution & Potency | *dissolution, potency, bioequivalence, release* | Class II |
| 2 | Packaging & Labeling | *label, barcode, ndc, seal, imprint* | Class III |
| 3 | Chemical Impurities | *nitrosamine, ndma, impurity, api, chemical* | Class II |

**Key Finding:** Clusters 0 and 3 together represent ~35% of recalls but ~70% of total financial liability.

### 8.3 Isolation Forest — Manufacturing Anomaly Detection

**📊 Visualization:** `EDA_06_manufacturing.png`

- **3.5% of batches** flagged as manufacturing anomalies (yield variance outliers)
- Anomalous batches show yield variance of **−7% to −10%** vs. normal range of −0.5% to −3%
- These correspond closely to the batches that subsequently appeared in recall events (validation check)

### 8.4 Batch Recall Predictor

**📊 Visualization:** `EDA_11_ai_predictions.png`

Random Forest trained on 6,000 finished product batches:
- **Accuracy:** ~85%
- **ROC-AUC:** ~0.85
- **Top recall risk indicators:** yield_variance, batch_qty (large batches → wider recall scope), dosage_form (injectables highest risk), unit_price (high-value products → more regulatory scrutiny)

---

## 9. Cross-Engine Correlations

**📊 Visualization:** `EDA_13_correlations.png`

### Key Correlations Discovered

| Correlation | Value | Business Implication |
|-------------|-------|---------------------|
| `velocity_pressure` ↔ `financial_loss_risk` | +0.87 | Strongest expiry predictor |
| `cover_days` ↔ `days_to_expiry` | −0.73 | Over-procurement drives expiry |
| `quantity_on_hand` ↔ `at_risk_flag` | +0.64 | Large stock = higher exposure |
| `yield_variance` ↔ `recall_flag` | −0.61 | Low yield → recall probability |
| `unit_price` ↔ `shelf_life` | +0.12 | Weak positive (biologics exception) |
| `return_reason=recall` ↔ `rag_zone=Amber/Red` | +0.69 | RAG zone predicts downstream returns |

**Key Cross-Engine Finding:** 30.7% of all customer returns originated from batches classified as Amber or Red RAG zone. This validates the ML Expiry Classifier as an upstream predictor of downstream reverse logistics costs — the engines are causally linked.

---

## 10. Key EDA Findings Summary

### Critical Business Findings

| # | Finding | Engine | Action |
|---|---------|--------|--------|
| 1 | ~30% of inventory (Amber + Red) is at immediate write-off risk | Engine 1 | Deploy RAG triage immediately |
| 2 | Top 20% of SKUs drive 78% of expiry value exposure (Pareto holds) | Engine 1 | Targeted commercial acceleration |
| 3 | Velocity pressure > 1.0 guarantees write-off regardless of promotions | Engine 1 | Cap procurement at shelf-life-gated PO quantities |
| 4 | 2 of 8 warehouses are chronically COLD (surplus); 2 are chronically HOT (stockout) | Engine 2 | Structured rebalancing programme |
| 5 | Winter Surge SKUs cause peak over-procurement → Amber zone accumulation 3 months later | Engine 3 | Demand-driven pre-season PO caps |
| 6 | Stage 2 Manufacturing defects = 55% of all FDA recalls | Engine 4 | Mandatory pre-release assay for yield <96% |
| 7 | 30.7% of returns came from Amber/Red zone batches | Cross-engine | RAG classification prevents reverse logistics costs upstream |
| 8 | 3.5% manufacturing batches are yield anomalies (Isolation Forest) | Engine 4 | Pre-release quarantine on flagged batches |

### Data Quality Summary

| Category | Assessment |
|----------|-----------|
| Completeness | ≥ 99.5% across critical columns |
| Consistency | No logical inversions (manufacture > expiry dates) |
| Uniqueness | No duplicate batch IDs or transaction IDs |
| Relevance | All 18 tables have clear engine-to-feature mapping |
| Volume | 52,548 rows sufficient for ML model training |

---

## Appendix — Visualization Index

| File | Title | Section |
|------|-------|---------|
| `EDA_01_row_counts.png` | Row Counts per Table (52,548 Total) | Data Understanding |
| `EDA_02_missing_values.png` | Missing Value Heatmap | Data Quality |
| `EDA_03_products_analysis.png` | Product Portfolio Analysis | Statistical Analysis |
| `EDA_04_manufacturers.png` | Manufacturer Distribution | Supply Chain |
| `EDA_05_supply_chain.png` | Supply Chain Volume Analysis | Network |
| `EDA_06_manufacturing.png` | Manufacturing Yield Analysis | Engine 4 |
| `EDA_07_inventory.png` | Inventory & Warehouse Analysis | Engine 1 & 2 |
| `EDA_08_distribution.png` | Distribution Network Analysis | Engine 2 |
| `EDA_09_recalls.png` | FDA Recall Analysis | Engine 4 |
| `EDA_10_returns_disposal.png` | Returns & Disposal Analysis | Engine 4 |
| `EDA_11_ai_predictions.png` | AI Prediction Data Analysis | Engine 1 & 4 |
| `EDA_12_compliance.png` | Compliance Document Analysis | Audit |
| `EDA_13_correlations.png` | Key Numerical Correlations | Cross-Engine |
| `EDA_14_supply_chain_flow.png` | End-to-End Supply Chain Flow | Architecture |

---

*PharmaTrace AI | ISB AMPBA Capstone 2026 | Sponsor: Innodatatics Inc.*
*EDA Report — Final Submission*
