# PharmaTrace AI — End-to-End Project Lifecycle Presentation

**Deliverable 5: End-to-End Project Lifecycle Presentation**
**ISB AMPBA Capstone | Sponsor: Innodatatics Inc.**
**Format:** This document serves as the complete narrative script for the final presentation

---

# SLIDE 1 — Title Slide

**PharmaTrace AI**
*Pharmaceutical Supply Chain Intelligence Platform*

4 Strategic AI Engines | Expiry Risk | Network Rebalancing | Demand Forecasting | Reverse Logistics

ISB AMPBA Capstone 2026 | Sponsor: Innodatatics Inc.
Live Dashboard: https://cap-pharmatrace.streamlit.app

---

# SLIDE 2 — Business Problem & Objectives

## The Problem: Pharmaceutical Supply Chains Lose Billions to Preventable Failures

| Problem | Annual Impact |
|---------|--------------|
| Inventory expiry write-offs | 3–8% of warehouse value (~$1.5M–$40M/year for mid-sized distributor) |
| Inefficient stock distribution (HOT/COLD imbalance) | 15–25% unfulfilled hospital orders during stockouts |
| Inaccurate demand forecasting | Over-procurement leads to excess inventory → expiry cascade |
| Recall failures caught too late | FDA Class I recalls cost $10M–$600M per event |
| Certified disposal non-compliance | EPA/DEA violations: $25,000–$70,000/day in fines |

## Business Objectives
1. **Predict** which batches will expire before being sold (ML Expiry Classifier)
2. **Rebalance** network to move surplus stock to demand-deficit nodes (Network Engine)
3. **Forecast** future demand accurately to guide procurement (Demand Engine)
4. **Prevent** regulatory recalls and manage returns end-to-end (Reverse Logistics Engine)

---

# SLIDE 3 — Requirements & Scope

## What Was Required

**Functional Requirements**
- Real-time batch expiry risk classification with RAG colour zoning
- Inter-warehouse transfer recommendation with freight cost comparison
- XGBoost demand forecasting at 1M / 3M / 6M horizons
- FDA recall root-cause staging and financial prevention ROI model
- Executive dashboard accessible without data science knowledge

**Non-Functional Requirements**
- Deployable on Streamlit Cloud (zero-infrastructure)
- Upload-based: works with any client's Excel data file
- All AI insight cards must be dynamic (data-driven, not static text)
- Load time < 10 seconds on standard internet connection

**Out of Scope**
- Direct ERP/WMS API integration
- Real-time IoT sensor feeds
- Mobile application

---

# SLIDE 4 — Research & Approach

## Literature & Industry Framework

| Framework | Application in PharmaTrace |
|-----------|---------------------------|
| **FEFO (First Expiry First Out)** | Core inventory dispatch logic — earliest expiry batch dispatched first |
| **4-Color RAG (Red/Amber/Yellow/Green)** | Clinical shelf-life triage adapted from hospital pharmacy standards |
| **1-10-100 Quality Cost Principle** | Financial value saved calculator in Reverse Logistics Engine |
| **FDA 21 CFR Part 211** | Regulatory compliance mapping for storage, expiry, and disposal |
| **EPA RCRA §264** | Hazardous pharmaceutical waste certified destruction requirements |
| **DEA Title 21 CFR** | Controlled substance witnessed destruction protocol |

## ML Methodology Decisions

| Decision | Rationale |
|----------|-----------|
| Random Forest as champion | Handles class imbalance, non-linear interactions between shelf life and velocity |
| XGBoost for demand | Best-in-class gradient boosting for time-series structured tabular data |
| K-Means + TF-IDF for recalls | Unsupervised — no labelled defect archetype data available |
| Isolation Forest for anomalies | Detects rare yield deviations without needing labelled "bad batch" examples |
| Model tournament approach | Prevents overfitting to a single algorithm; F1 determines champion |

---

# SLIDE 5 — Data & Data Architecture

## Data Sources

| Source | Description | Volume |
|--------|-------------|--------|
| Synthetic master dataset | 30+ tables covering full pharma supply chain | 15,000+ inventory rows |
| FDA recall database simulation | 3,000 FDA recall events with reason-for-recall text | 3,000 rows |
| Monthly demand history | 24-month shipment dispatch history per SKU/warehouse | 24 × SKUs × Warehouses |
| Freight cost matrix | Inter-warehouse freight cost per unit shipped | 8 × 8 matrix |
| IoT cold-chain logs | Simulated temperature telemetry | 10,000+ readings |

## Key Data Tables

```
inventory          → core batch-level stock data (quantity, expiry, value)
products           → SKU master (dosage form, shelf life, price)
warehouses         → DC network (8 distribution centres)
shipments          → historical outbound dispatch transactions
monthly_demand     → aggregated monthly demand/dispatch per SKU/WH
returns            → customer RMA return records
recalls            → FDA recall events
finished_product_batches → manufacturing batch genealogy
manufacturing_orders    → production order quantities
```

## Data Quality & Preprocessing
- Null expiry dates → excluded from ML features (defensive `dropna`)
- Negative DTE → retained as "expired" class (Red zone)
- Zero velocity → floored at 5% of base to avoid divide-by-zero
- Missing freight costs → defaulted to $0.85/unit median
- Missing product prices → defaulted to $45.00/unit median

---

# SLIDE 6 — Solution Architecture

> 📌 **Use diagram:** `07_complete_documentation/PharmaTrace_Solution_Architecture.jpg`

```
┌─────────────────────────────────────────────────────────────────┐
│                    PHARMATRACE AI PLATFORM                       │
│                    (Streamlit Application)                        │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  DATA INGESTION LAYER                                            │
│  ┌─────────────────┐  ┌──────────────────┐  ┌───────────────┐  │
│  │  Uploaded Excel │  │  Pre-loaded      │  │  Cached PKL   │  │
│  │  (User's data)  │  │  Synthetic Data  │  │  (XGBoost)    │  │
│  └─────────────────┘  └──────────────────┘  └───────────────┘  │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  4 STRATEGIC AI ENGINES                                          │
│                                                                  │
│  ┌──────────────────┐  ┌──────────────────────────────────────┐ │
│  │ Engine 1         │  │ Engine 2                             │ │
│  │ ML Expiry        │  │ Network Rebalancing & Transfers      │ │
│  │ Classifier       │  │ HOT/COLD Detection + ML Transfer     │ │
│  │ RAG + RF/GBM/LR  │  │ Recommender + Freight Analysis       │ │
│  └──────────────────┘  └──────────────────────────────────────┘ │
│                                                                  │
│  ┌──────────────────┐  ┌──────────────────────────────────────┐ │
│  │ Engine 3         │  │ Engine 4                             │ │
│  │ Demand &         │  │ Reverse Logistics &                  │ │
│  │ Seasonality      │  │ Certified Disposal                   │ │
│  │ XGBoost 1M/3M/6M │  │ K-Means + IsoForest + RF Recall      │ │
│  └──────────────────┘  └──────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  PRESENTATION LAYER                                              │
│  Executive KPIs | AI Insight Cards | Interactive Simulators      │
│  Charts | Download Manifests | Prescriptive Action Registers     │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  DEPLOYMENT                                                      │
│  Streamlit Cloud → https://cap-pharmatrace.streamlit.app         │
└─────────────────────────────────────────────────────────────────┘
```

---

# SLIDE 7 — Model / AI Approach

## Engine 1: ML Expiry Risk Classifier

**Problem:** Will this batch expire before being fully sold?

**Feature Engineering:**
- `cover_days` = Stock on Hand / Monthly Dispatch Velocity × 30
- `velocity_pressure` = Cover Days / Days to Expiry (>1.0 = guaranteed write-off)
- `pct_life_remaining` = Days to Expiry / Total Shelf Life Days
- `capital_velocity_ratio` = Inventory Value per Day / Monthly Dispatch Velocity

**Model Tournament:**
```
Logistic Regression  → Baseline linear classifier
Random Forest        → Non-linear, handles feature interactions
Gradient Boosting    → Sequential error correction
→ Champion selected by Weighted F1 on 25% holdout test set
```

**RAG Zone Assignment:**
```
🟢 Green  → RSL > 12 months  → Monitor only
🟡 Yellow → RSL 7–12 months  → Velocity Sprint required
🟠 Amber  → RSL 4–6 months   → 180-day distributor cliff (URGENT)
🔴 Red    → RSL < 3 months   → Certified disposal mandatory
```

## Engine 2: Network Rebalancing

**HOT/COLD Detection:**
- HOT: Days of Stock < 30 (stockout risk)
- COLD: Days of Stock > 120 (surplus / write-off risk)

**Transfer Scoring:**
```
Net Savings = (Batch Value Rescued) − (Freight Cost)
              Where: Freight Cost = Transfer Qty × Freight Rate per Unit
Transfer approved if: Net Savings > 0 AND DTE > Lead Time (4 days)
```

## Engine 3: XGBoost Demand Forecasting

**Architecture:**
- Separate XGBoost model per horizon (1M, 3M, 6M)
- 80/20 chronological split (no data leakage)
- Monotonic enforcement: 3M ≥ 1M, 6M ≥ 3M
- Features: lag demand, rolling averages, month-of-year, product type

**7 Clinical Demand Patterns:**
```
1. Winter Surge Respiratory    → Peak Nov–Feb
2. Seasonal Allergy            → Peak Mar–Jun
3. Chronic Maintenance         → Flat/Stable year-round
4. Controlled Substance        → Stable, DEA-monitored
5. Specialty Oncology          → Erratic, low-volume
6. Biologic/Specialty          → Cold-chain, high-value
7. Generic/OTC                 → Price-elastic, high-volume
```

## Engine 4: Reverse Logistics AI

**K-Means NLP on 3,000 FDA Recall Events:**
```
TF-IDF vectorization (50 features) → K-Means (k=4)
→ Cluster 0: Sterile & Particulate (Class I — life threatening)
→ Cluster 1: Dissolution & Potency (Class II)
→ Cluster 2: Packaging & Labeling (Class III)
→ Cluster 3: Chemical Impurities / NDMA (Class II)
```

**Isolation Forest — Manufacturing Anomaly:**
```
Features: yield_variance, batch_qty
Contamination: 3.5% (expected abnormal batch rate)
Output: Normal / 🚨 Anomaly — triggers pre-release quarantine
```

**Recall Probability Predictor:**
```
Random Forest (60 trees, max_depth=7)
Features: batch_qty, yield_variance, unit_price, shelf_life_months, dosage_form, manufacturer_id
Target: Will this batch be recalled? (Binary)
Output: Probability % + Suspected Root-Cause Stage + Recommended Action
```

---

# SLIDE 8 — Development Process

## Timeline

| Phase | Work Done | Duration |
|-------|-----------|---------|
| Data Architecture | Schema design, 30+ table synthetic dataset generation | Week 1–2 |
| Engine 1 | RAG matrix, ML classifier, recovery playbook | Week 3–4 |
| Engine 2 | HOT/COLD detection, transfer recommender, freight analysis | Week 5 |
| Engine 3 | XGBoost pipeline, pattern classifier, procurement workorder | Week 6–7 |
| Engine 4 | RMA tracking, recall staging, K-Means, Isolation Forest, recall predictor | Week 8–9 |
| Dashboard | UI/UX, AI insight cards, simulators, alignment audit | Week 10 |
| Deployment | Streamlit Cloud, GitHub CI/CD, submission packaging | Week 11 |

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Application | Python 3.11, Streamlit 1.32+ |
| ML Models | scikit-learn, XGBoost |
| Data | pandas, numpy, openpyxl |
| Visualization | matplotlib, seaborn |
| Deployment | Streamlit Cloud, GitHub |
| Containerization | Docker, Docker Compose |

---

# SLIDE 9 — Key Features & Modules

## Engine 1 — ML Expiry Classifier
- ✅ 4-color RAG zone dashboard with real-time KPIs
- ✅ Batch-level action planning matrix (Dispatch / Transfer / Liquidate / Dispose)
- ✅ Demand Scenario Analysis (Pessimistic / Base / Optimistic)
- ✅ SKU Pareto — identify 20% of products driving 80% of write-off risk
- ✅ Warehouse Risk Concentration heatmap
- ✅ Prescriptive Recovery Playbook with quantified dollar recovery
- ✅ AI Insight card — dynamic recommendations driven by live data

## Engine 2 — Network Rebalancing & Transfers
- ✅ Geographic HOT/COLD warehouse detection with KPI cards
- ✅ Batch-level transfer manifest (origin DC → destination DC)
- ✅ Transfer cost vs. manufacturing cost comparison
- ✅ Top 3 executive transfer directive cards
- ✅ Velocity-to-Save simulator (interactive)
- ✅ Capital reclamation vs. baseline loss AI insight
- ✅ Certified disposal routing for unsalvageable stock

## Engine 3 — Demand & Seasonality
- ✅ XGBoost 1M / 3M / 6M forecast per SKU
- ✅ Clinical demand pattern classification (7 pattern types)
- ✅ Monthly seasonality trend visualization
- ✅ Top growing / declining SKU detection
- ✅ Procurement Action Plan with net order quantities
- ✅ Warehouse intelligence (capacity, volume, demand share)
- ✅ Distributor demand intelligence
- ✅ Dynamic AI insights for demand patterns and warehouse supply

## Engine 4 — Reverse Logistics & Certified Disposal
- ✅ RMA return tracking with financial leakage radar
- ✅ Supply chain recall staging (4 stages: Sourcing / Formulation / Packaging / Logistics)
- ✅ 1-10-100 Quality Cost Principle table (ROI of early interception)
- ✅ Interactive intercept value calculator (Stage 1–4 simulator)
- ✅ K-Means NLP defect archetype clustering
- ✅ Isolation Forest manufacturing anomaly detection
- ✅ New-batch recall probability simulator (interactive)
- ✅ EPA/DEA certified disposal method recommender
- ✅ 1-click certified disposal audit manifest download

---

# SLIDE 10 — Testing & Validation

## ML Model Validation

| Engine | Validation Method | Key Metric | Result |
|--------|------------------|-----------|--------|
| Engine 1 | 75/25 stratified train/test split | Weighted F1 Score | Champion > 90% |
| Engine 3 | 80/20 chronological split (no leakage) | MAPE | < 20% on holdout |
| Engine 4 Recall Predictor | 75/25 random split | Accuracy + ROC-AUC | ~85% / ~0.85 |
| Engine 4 Isolation Forest | Contamination parameter validation | Flag rate | 3.5% of batches |

## Dashboard Validation
- All AI insight bullets verified as dynamic (data-driven, no hardcoded static text)
- Zero hardcoded batch counts or financial figures in insight cards
- Verified RAG zone key matching (`🔴 Red (<3M / Expired)` not just `'Red'`)
- All engine pages tested with both synthetic and real uploaded data
- Mobile and tablet responsiveness checked

## Regulatory Compliance Cross-Check
- FDA 21 CFR §211.142 (storage) — Red zone quarantine action validated
- FDA 21 CFR §211.150 (distribution) — FEFO logic verified
- EPA RCRA §264 (disposal) — 8% destruction cost correctly applied
- DEA Title 21 CFR — witnessed destruction flag for Schedule II drugs validated

---

# SLIDE 11 — Deployment

## Streamlit Cloud
- **URL:** https://cap-pharmatrace.streamlit.app
- **Auto-deployment:** Every push to `main` branch auto-deploys (GitHub → Streamlit Cloud)
- **Build time:** ~2 minutes on clean build
- **No secrets / environment variables required**
- **Data:** Synthetic dataset pre-loaded; real data uploaded by user at runtime

## GitHub Repository
- **URL:** https://github.com/Siddharth0711/cap-pharmatrace
- **Branch strategy:** Single `main` branch (capstone project)
- **CI/CD:** GitHub Actions + Streamlit Cloud webhook

## Alternative Deployment Options
- **Docker:** `docker-compose up --build` → http://localhost:8501
- **Local:** `streamlit run streamlit_app.py` → http://localhost:8501
- **Binder:** One-click cloud notebook (for presentation/demo only)

---

# SLIDE 12 — Results & Outcomes

## Business Impact Delivered

| Metric | Value |
|--------|-------|
| At-risk inventory identified | Dynamic — computed from uploaded data |
| Network transfer savings | Dynamic — based on freight matrix |
| Recall prevention ROI | Up to 168× at Stage 1 (Sourcing) |
| Models trained | 10+ across 4 engines |
| Dashboard pages | 8 navigation pages |
| Interactive simulators | 4 (Velocity-to-Save, Intercept Value, Batch Recall, Disposal Recommender) |
| AI Insight cards | 7 dynamic insight cards across all engines |

## Technical Outcomes
- **5,413 lines** of production Python code
- **10+ ML models** (supervised + unsupervised)
- **30+ data tables** handled by a single upload
- **Deployed live** — accessible from any browser globally

---

# SLIDE 13 — Challenges & Learnings

| Challenge | How Resolved |
|-----------|-------------|
| Synthetic data needed to look clinically real | Designed 30-table schema with FK relationships; seeded distributions from real pharma patterns |
| XGBoost demand MAPE spiked on low-volume SKUs | Added monotonic enforcement; log-transform on target for low-demand products |
| Streamlit performance with 15,000+ row datasets | Applied `@st.cache_data` decorators on all heavy computations |
| K-Means labelling — clusters had no natural names | Mapped cluster centroids manually to pharmaceutical defect archetypes |
| Static AI insight values detected in audit | Replaced all hardcoded fallback numbers with dynamic computed variables |
| "LP Cost Optimizer" references confused reviewers | Removed all internal engine cross-references; each engine is self-contained |

---

# SLIDE 14 — Future Enhancements

| Priority | Enhancement |
|----------|------------|
| 🔴 High | Real-time ERP/SAP data connector (API bridge) |
| 🔴 High | IoT cold-chain temperature integration (live excursion alerts) |
| 🟠 Medium | Automated email/WhatsApp alerts for Red-zone batch crossings |
| 🟠 Medium | Natural Language query interface ("Show me all batches expiring this month") |
| 🟡 Low | Mobile-optimized responsive layout |
| 🟡 Low | Multi-language support (Hindi, regional languages for Indian market) |
| 🟡 Low | PDF auto-report generation for board meetings |
| 🔵 Research | Reinforcement learning for dynamic FEFO routing optimization |

---

*PharmaTrace AI | ISB AMPBA Capstone 2026 | Sponsor: Innodatatics Inc.*
*End of Presentation*
