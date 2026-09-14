# 🚀 AI-Enabled Exoplanet Detection System — Full Implementation Plan

> **Source Document**: College_Project.pdf — *AI-Enabled Exoplanet Detection Pipeline* (PRD v1.0, June 2026)  
> **Goal**: Autonomously detect and classify exoplanet transit signals in noisy NASA TESS light curves.  
> **Target**: ≥90% macro-F1 across 4 signal classes | ≤90 min runtime for a full 25k-target TESS sector
>
> **Scope note**: This document remains the spec for the *scientific pipeline* (Stages 1–6, the multi-agent design, and acceptance criteria below). For the productization plan — turning this pipeline into LunaVisionAI as a commercial multi-tenant SaaS (architecture, orchestration, billing, roadmap) — see [`SAAS_ARCHITECTURE_PLAN.md`](./SAAS_ARCHITECTURE_PLAN.md).

---

## 📋 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Directory Structure](#2-directory-structure)
3. [Dataset Making Process](#3-dataset-making-process)
4. [Backend Pipeline — Step-by-Step](#4-backend-pipeline--step-by-step)
5. [Agent Modeling — Step-by-Step](#5-agent-modeling--step-by-step)
6. [Frontend Dashboard — Step-by-Step](#6-frontend-dashboard--step-by-step)
7. [Technology Stack](#7-technology-stack)
8. [Implementation Roadmap & Phases](#8-implementation-roadmap--phases)
9. [Testing & Validation Strategy](#9-testing--validation-strategy)
10. [Risks & Mitigations](#10-risks--mitigations)
11. [Acceptance Criteria (Go/No-Go Checklist)](#11-acceptance-criteria-gono-go-checklist)

---

## 1. Project Overview

### Problem Statement
Manual review of 20,000–30,000 light curves per TESS sector is infeasible. Exoplanet transits manifest as periodic, box-shaped dips in stellar flux of only 10⁻⁴ to 10⁻² depth — far below typical photometric noise floors. The system must automatically:

- Ingest and preprocess raw TESS FITS files from NASA's MAST archive
- Detect periodic dip signals using BLS/TLS periodograms
- Classify each signal into one of **4 classes**: `Transit`, `Eclipsing Binary`, `Blend`, `Other Variability`
- Estimate orbital parameters (period, depth, duration) with Bayesian uncertainties
- Produce publication-quality visualizations and a ranked candidate catalog

### Signal Taxonomy

| Class | Shape Feature | Physical Origin | Discriminating Trait |
|---|---|---|---|
| **Transit** | Flat-bottomed dip | Planet occults host star | Even depth; no secondary eclipse |
| **Eclipsing Binary** | V- or U-shaped dip | Stellar companion eclipse | Secondary eclipse ~50% depth |
| **Blend (3rd-body)** | Diluted dip | Background EB in aperture | Odd-even depth mismatch; centroid shift |
| **Starspot/Variability** | Sinusoidal/asymmetric | Stellar rotation/activity | No periodicity or non-periodic shape |

---

## 2. Directory Structure

```
exoplanet_pipeline/
├── data/
│   ├── raw/               # Downloaded TESS FITS files
│   ├── processed/         # Detrended light curves (.npz)
│   └── catalogs/          # TOI, TIC, ExoFOP reference catalogs
├── models/                # Trained model artifacts (.pkl, .pt)
├── src/
│   ├── acquisition.py     # MAST download & FITS parsing
│   ├── preprocessing.py   # Detrending, normalization, outlier removal
│   ├── detection.py       # TLS/BLS periodogram wrapper
│   ├── features.py        # Feature extraction for ML
│   ├── classifier.py      # Train, predict, evaluate ensemble model
│   ├── fitting.py         # batman + emcee parameter estimation
│   ├── visualization.py   # All plotting functions
│   ├── report.py          # PDF report generation
│   └── agents/            # Multi-agent orchestration layer
│       ├── orchestrator.py
│       ├── acquisition_agent.py
│       ├── preprocessing_agent.py
│       ├── detection_agent.py
│       ├── classification_agent.py
│       ├── fitting_agent.py
│       └── reporting_agent.py
├── frontend/
│   ├── app.py             # Streamlit or FastAPI + React dashboard
│   ├── pages/             # Dashboard pages/views
│   └── static/            # CSS, JS assets
├── notebooks/             # EDA and result exploration notebooks
├── tests/                 # Unit and integration tests
├── outputs/               # Candidate catalogs, plots, reports
├── requirements.txt       # Full dependency list
└── run_pipeline.py        # CLI entry point for full pipeline
```

---

## 3. Dataset Making Process

> **IMPORTANT**: The quality of your training dataset directly determines whether you hit ≥90% macro-F1. Follow every step carefully.

### Step 3.1 — Define Label Sources

Collect labeled data from four complementary sources:

| Source | Labels | Approx. Size | Role |
|---|---|---|---|
| **ExoFOP-TESS Curated Set** | Planet, EB, FP, BEB | ~4,000 | Primary training |
| **NASA Exoplanet Archive** | Confirmed planets | ~5,500 | Positive augmentation |
| **TESS SPOC TCE Catalog** | TCEs with dispositions | ~10,000 | Feature pre-training |
| **Synthetic Injection-Recovery** | Generated transits | ~2,000 | Minority class boost |

### Step 3.2 — Download Reference Catalogs

```python
# Download TOI catalog
from astroquery.mast import Catalogs
toi = Catalogs.query_criteria(catalog="Tic", Tmag=[6, 14])

# Download ExoFOP labels via astroquery
# Alternatively, download CSV manually from https://exofop.ipac.caltech.edu/tess/
```

1. **ExoFOP-TESS**: Download the cumulative TOI catalog from exofop.ipac.caltech.edu
2. **NASA Exoplanet Archive**: Download confirmed planets table (CSV) from exoplanetarchive.ipac.caltech.edu
3. **TESS SPOC TCE Catalog**: Obtain from MAST bulk download; `dvt.fits` files contain Threshold Crossing Events
4. **TIC Catalog cross-match**: Use `astroquery.mast.Catalogs.query_object()` to get stellar parameters for each TIC ID

### Step 3.3 — Assign Class Labels

Map raw disposition strings to unified integer labels:

```python
label_map = {
    'PC': 0,        # Planet Candidate → Transit
    'KP': 0,        # Known Planet → Transit
    'CP': 0,        # Confirmed Planet → Transit
    'EB': 1,        # Eclipsing Binary
    'BEB': 2,       # Background Eclipsing Binary → Blend
    'FP': 3,        # False Positive → Other
    'V': 3,         # Variable Star → Other
}
```

### Step 3.4 — Download Raw TESS Light Curves

```python
import lightkurve as lk

# Example: download all 2-min cadence data for a TIC ID
search_result = lk.search_lightcurve("TIC 123456789", author="SPOC", cadence="2min")
lc = search_result.download_all()
```

- For bulk download: use `astroquery.mast.Observations.download_products()` with FITS filters
- Filter: `QUALITY` bitmask flags `1, 2, 4, 8, 16, 512` must be zero (good cadences only)
- Download both `SAP_FLUX` (Simple Aperture Photometry) and `PDCSAP_FLUX` (PDC corrected)
- Cache all FITS files locally under `data/raw/sector_XX/`

### Step 3.5 — Synthetic Transit Injection (Class Imbalance Boost)

```python
import batman
import numpy as np

def inject_transit(time, flux, period, t0, rp, a, inc=87.5):
    params = batman.TransitParams()
    params.t0 = t0
    params.per = period
    params.rp = rp          # Rp/R* ratio
    params.a = a            # semi-major axis / R*
    params.inc = inc
    params.ecc = 0
    params.w = 90
    params.u = [0.3, 0.1]
    params.limb_dark = "quadratic"
    m = batman.TransitModel(params, time)
    return flux * m.light_curve(params)
```

- Generate 2,000 synthetic transit signals by injecting `batman`-modeled dips into real non-variable TESS light curves
- Vary `rp` in range [0.01, 0.15], `period` in [1, 30] days, `inc` in [85, 90] degrees
- Label all injected targets as class `0` (Transit)

### Step 3.6 — Preprocess All Light Curves for ML

Apply the full preprocessing pipeline (detailed in Section 4) to every raw light curve:
1. Normalize to unit median flux
2. Sigma-clip outliers (3σ, 5 iterations)
3. Fill small gaps (≤3 cadences) with linear interpolation
4. Detrend using `wotan` bi-weight filter
5. Save processed arrays as `.npz` files in `data/processed/`

### Step 3.7 — Extract 35+ Features Per Candidate (Feature Matrix)

Run the BLS/TLS periodogram on each processed light curve and compute features (see Section 4, Stage 4). Store as a pandas DataFrame:

```python
features_df.to_csv("data/catalogs/feature_matrix.csv", index=False)
```

Columns include: `tic_id`, `period`, `depth`, `duration`, `SDE`, `SNR`, `odd_even_ratio`, `centroid_shift`, `transit_count`, `stellar_Teff`, `stellar_logg`, `stellar_rad`, `secondary_depth`, and 20+ more.

### Step 3.8 — Class Balancing with SMOTE

```python
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

over = SMOTE(sampling_strategy=0.5, random_state=42)
under = RandomUnderSampler(sampling_strategy=0.8, random_state=42)
pipeline = Pipeline([('o', over), ('u', under)])
X_resampled, y_resampled = pipeline.fit_resample(X_train, y_train)
```

### Step 3.9 — Train/Val/Test Split

```python
from sklearn.model_selection import train_test_split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
```

Split: **80% train / 10% validation / 10% held-out test**. Stratify on class label to maintain distribution.

---

## 4. Backend Pipeline — Step-by-Step

The backend is structured as a **6-stage sequential processor** with feedback loops.

---

### Stage 1 — Data Acquisition Module (DAM) [`src/acquisition.py`]

**Goal**: Bulk-download raw TESS FITS files from MAST archive.

#### Steps:
1. **Accept CLI input**: sector number, camera, CCD (e.g., `--sector 1 --camera 1 --ccd 1`)
2. **Query MAST archive**:
   ```python
   from astroquery.mast import Observations
   obs = Observations.query_criteria(
       obs_collection="TESS", 
       sequence_number=sector,
       target_classification="STAR"
   )
   products = Observations.get_product_list(obs)
   filtered = Observations.filter_products(products, productSubGroupDescription="LC")
   ```
3. **Download FITS files** to `data/raw/sector_XX/` with resume support using local file existence check
4. **Parse FITS**: Extract `TIME`, `SAP_FLUX`, `PDCSAP_FLUX`, `QUALITY` columns using `astropy.io.fits`
5. **Apply QUALITY bitmask filter**: Remove cadences where `QUALITY & (1|2|4|8|16|512) != 0`
6. **Log statistics**: total targets queried, downloaded, skipped, failed
7. **Cache metadata**: Save a JSON manifest per sector listing all downloaded TIC IDs and file paths

---

### Stage 2 — Preprocessing & Detrending Module (PPM) [`src/preprocessing.py`]

**Goal**: Clean and normalize light curves; remove systematic trends while preserving transit signals.

#### Steps:
1. **Load FITS** and extract `PDCSAP_FLUX` as the primary flux array
2. **Normalize**: Divide all flux values by the median → unit median flux
   ```python
   flux_norm = flux / np.nanmedian(flux)
   ```
3. **Outlier rejection**: Apply iterative sigma-clipping
   ```python
   from astropy.stats import sigma_clip
   clipped = sigma_clip(flux_norm, sigma=3, maxiters=5)
   flux_clean = flux_norm[~clipped.mask]
   ```
4. **Gap filling**: Identify gaps >3 cadences; flag them; fill gaps ≤3 cadences with `np.interp()`
5. **Detrend with wotan**:
   ```python
   from wotan import flatten
   flat_flux, trend = flatten(time, flux_clean, method='biweight', window_length=0.75)
   ```
6. **Optional GP detrending** (for heavily variable stars): Use `celerite2` to model stellar variability as a Gaussian Process and subtract it
7. **Quality gating**:
   - Skip if usable data < 13.5 days
   - Skip if noise floor σ > 5× expected photon noise
   - Flag crowded apertures using TIC contamination ratio
8. **Save** detrended flux to `data/processed/sector_XX/TIC_XXXXXXX.npz`

---

### Stage 3 — Periodicity Detection Module (PDM) [`src/detection.py`]

**Goal**: Identify statistically significant periodic dip signals using BLS/TLS periodogram.

#### Steps:
1. **Run TLS (Transit Least Squares)** on detrended flux:
   ```python
   from transitleastsquares import transitleastsquares
   model = transitleastsquares(time, flat_flux)
   results = model.power(
       minimum_period=0.5,
       maximum_period=27,
       oversampling_factor=5,
       duration_grid_step=1.05
   )
   ```
2. **Extract peak period**: `results.period`, `results.T0`, `results.depth`, `results.duration`
3. **Compute SDE** (Signal Detection Efficiency): `results.SDE`
4. **Apply SDE threshold**: Keep only candidates with `SDE ≥ 7`
5. **Compute FAP** (False Alarm Probability): Use TLS analytical FAP → keep FAP < 0.01
6. **Extract secondary eclipse period**: Check for secondary dip at phase 0.5 (EB discriminator)
7. **Odd/even transit depth comparison**: Compare alternating transit depths to flag EBs
8. **Build candidate list**: For each target passing thresholds, record: `{tic_id, period, t0, depth, duration, SDE, FAP}`
9. **Serialize**: Save candidate list as `outputs/candidates_sector_XX.csv`

---

### Stage 4 — Feature Engineering Module (FEM) [`src/features.py`]

**Goal**: Extract 35+ statistical, morphological, and astrophysical features per candidate for ML input.

#### Feature Categories:

| Category | Features |
|---|---|
| **Transit Geometry** | period, depth (δ), duration (T14), impact parameter proxy, transit count |
| **Signal Quality** | SDE, SNR, FAP, phase coverage fraction |
| **Morphological** | flat bottom score, ingress/egress asymmetry, limb-darkening slope |
| **Odd/Even** | odd transit depth, even transit depth, odd/even ratio |
| **Secondary Eclipse** | secondary depth, secondary depth ratio |
| **Centroid** | centroid shift during transit (arcsec), PRF contamination score |
| **Stellar** | Teff, log g, R★, Gaia magnitude, TIC contamination ratio, spectral type |
| **Time-Series Stats** | RMS scatter, skewness, kurtosis, autocorrelation, power spectrum slope |
| **tsfresh extras** | abs_energy, fft_coefficient, change_quantiles, mean_abs_change |

#### Steps:
1. **Load** detrended flux and candidate parameters
2. **Phase-fold** the light curve at the candidate period
3. **Compute geometric features** from phase-folded transit shape
4. **Cross-match TIC/Gaia** for stellar parameters using `astroquery`
5. **Compute centroid shift**: Load pixel-level TESS data and measure centroid motion during vs. out of transit
6. **Run tsfresh** on the phase-folded flux array for automated statistical features
7. **Assemble feature vector** as a pandas Series; concatenate into the master feature DataFrame
8. **Handle missing values**: Impute with median for missing stellar params; flag with a boolean mask column

---

### Stage 5 — ML Classification Module (CLM) [`src/classifier.py`]

**Goal**: Classify each candidate into Transit / Eclipsing Binary / Blend / Other with calibrated confidence.

#### Steps:

**5a. Training Phase** (offline, run once):
1. Load feature matrix + labels from Step 3.7–3.9
2. Apply SMOTE balancing (Step 3.8)
3. **Train XGBoost**:
   ```python
   import xgboost as xgb
   xgb_model = xgb.XGBClassifier(
       n_estimators=500, max_depth=6, learning_rate=0.05,
       use_label_encoder=False, eval_metric='mlogloss',
       early_stopping_rounds=20
   )
   xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
   ```
4. **Train LightGBM**:
   ```python
   import lightgbm as lgb
   lgb_model = lgb.LGBMClassifier(
       n_estimators=500, learning_rate=0.05, num_leaves=63
   )
   lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                 callbacks=[lgb.early_stopping(20)])
   ```
5. **Hyperparameter optimization** with Optuna (5-fold CV):
   ```python
   import optuna
   study = optuna.create_study(direction='maximize')
   study.optimize(objective_fn, n_trials=100)
   ```
6. **Stack with meta-learner**:
   ```python
   from sklearn.linear_model import LogisticRegression
   meta_X = np.column_stack([xgb_proba, lgb_proba])
   meta_model = LogisticRegression().fit(meta_X, y_train)
   ```
7. **Calibrate probabilities** with Platt scaling:
   ```python
   from sklearn.calibration import CalibratedClassifierCV
   calibrated = CalibratedClassifierCV(meta_model, method='sigmoid', cv='prefit')
   calibrated.fit(meta_X_val, y_val)
   ```
8. **Save models**: `joblib.dump(calibrated, 'models/ensemble_v1.0.pkl')`
9. **SHAP analysis**: Compute SHAP values to verify physically meaningful features dominate

**5b. Inference Phase** (per sector run):
1. Load saved ensemble model
2. Transform new candidate features with the same scaler
3. Predict class probabilities: `proba = calibrated.predict_proba(X_new)`
4. Assign final class label: `label = np.argmax(proba, axis=1)`
5. Flag low-confidence candidates: `max_proba < 0.65` → send to active learning queue

**5c. (Optional) 1-D CNN Model**:
- Input: 200-point phase-folded flux array
- Architecture: 5 × Conv1D(ReLU) → GlobalAveragePooling → Dense(128, dropout=0.3) → Dense(4, softmax)
- Trained on ExoFOP labeled dataset using PyTorch
- Combined with XGBoost/LightGBM via the meta-learner

---

### Stage 6 — Parameter Estimation & Reporting (PER) [`src/fitting.py`, `src/report.py`]

**Goal**: Fit precise orbital parameters with Bayesian uncertainties and produce output catalogs.

#### Steps:
1. **Select transit candidates** (class 0 predictions)
2. **Phase-fold light curve** at TLS-detected period
3. **Fit batman transit model**:
   ```python
   import batman
   params = batman.TransitParams()
   params.t0 = t0_init; params.per = period_init
   params.rp = rp_init; params.a = a_init
   params.inc = inc_init; params.ecc = 0; params.w = 90
   params.u = [0.3, 0.1]; params.limb_dark = "quadratic"
   m = batman.TransitModel(params, time_folded)
   ```
4. **Run MCMC with emcee**:
   ```python
   import emcee
   sampler = emcee.EnsembleSampler(nwalkers=32, ndim=6, log_prob_fn=log_probability)
   sampler.run_mcmc(p0, nsteps=1000, progress=True)
   ```
5. **Extract credible intervals**: Use 16th, 50th, 84th percentiles of posterior chains
6. **Compute SNR**: `SNR = delta * sqrt(N_transit * T14 / sigma)`
7. **Compute planetary radius**: `Rp = rp * R_star` using TIC stellar radius
8. **Compute habitable zone score**: Compare period to stellar luminosity-based HZ boundaries
9. **Serialize results**: Append to master CSV and FITS catalog
10. **Generate PDF report**: Using Jinja2 templates, auto-fill 3-page summary per hackathon spec

---

## 5. Agent Modeling — Step-by-Step

The pipeline is redesigned as a **Multi-Agent System (MAS)** where each stage is encapsulated in an autonomous agent. Agents communicate via a message queue and are orchestrated by a central coordinator.

### 5.1 Agent Architecture Overview

```
User / CLI
    |
    v
[Orchestrator Agent]
    |         |          |            |              |           |
    v         v          v            v              v           v
[Acquire] [Preproc] [Detect]  [Feature Ext.] [Classify]  [Fitting]
                                                  |
                                         [Active Learning] <-> [Label UI]
                                                  |
                                            [Reporting]
```

### 5.2 Orchestrator Agent [`src/agents/orchestrator.py`]

**Role**: Master coordinator — dispatches work, monitors agent health, handles failures, and enforces pipeline sequencing.

#### Steps:
1. **Parse CLI arguments**: sector, camera, CCD, output directory, skip flags
2. **Initialize message queue**: Use Python `multiprocessing.Queue` or Redis for inter-agent comms
3. **Spawn agent processes**: Each agent runs as an independent process or thread
4. **Dispatch Stage 1**: Send `{action: 'acquire', sector: 1, camera: 1, ccd: 1}` to Acquisition Agent
5. **Monitor completion**: Listen for `{status: 'done', stage: 'acquisition', ...}` messages
6. **Handle agent failures**: Implement retry logic (max 3 retries), fallback strategies
7. **Control flow**: Trigger next agent only when current stage reports success
8. **Log pipeline state**: Write timestamped stage start/end events to `outputs/pipeline_run.log`
9. **Enforce timeout**: Kill stalled agents after configurable timeout (default: 30 min per stage)

#### Agent Interface Contract:
```python
class BaseAgent:
    def receive(self, message: dict) -> None: ...
    def process(self) -> dict: ...
    def report_done(self) -> dict: ...
    def report_error(self, exc: Exception) -> dict: ...
```

### 5.3 Acquisition Agent [`src/agents/acquisition_agent.py`]

**Responsibilities**:
- Query MAST for sector targets
- Manage download queue with concurrency limit (max 8 parallel downloads)
- Implement exponential backoff on HTTP 429 (rate-limit) responses
- Report progress (downloaded / total) to Orchestrator every 30 seconds
- Write `data/raw/sector_XX/manifest.json` on completion

### 5.4 Preprocessing Agent [`src/agents/preprocessing_agent.py`]

**Responsibilities**:
- Consume list of FITS files from the manifest
- Process in parallel batches using `joblib.Parallel(n_jobs=-1)`
- Apply quality gating rules (skip bad targets early)
- Report skipped count and reason codes to Orchestrator
- Write `.npz` files and a processing report

### 5.5 Detection Agent [`src/agents/detection_agent.py`]

**Responsibilities**:
- Run TLS periodogram on each processed light curve
- Filter by SDE threshold (>=7) and FAP (<0.01)
- Collect odd/even transit metrics and secondary eclipse depth
- Return ranked candidate list to Orchestrator
- Expose intermediate BLS power spectrum data for visualization

### 5.6 Feature Extraction Agent [`src/agents/feature_agent.py`]

**Responsibilities**:
- Accept candidate list from Detection Agent
- Parallelized feature computation per candidate
- Cross-match TIC/Gaia catalogs (cache results to avoid repeat API calls)
- Handle missing stellar parameters gracefully with imputation
- Output: `feature_matrix.csv` with all features and target labels

### 5.7 Classification Agent [`src/agents/classification_agent.py`]

**Responsibilities**:
- Load pre-trained ensemble model from `models/`
- Run inference on feature matrix
- Assign class labels and calibrated probabilities
- Separate high-confidence (>=0.65) vs. low-confidence (<0.65) candidates
- Emit low-confidence candidates to Active Learning Agent
- Output: labeled candidates with probabilities appended to feature matrix

### 5.8 Parameter Fitting Agent [`src/agents/fitting_agent.py`]

**Responsibilities**:
- Process only Transit-class candidates (class 0)
- Run batman+emcee MCMC fit for each
- Handle MCMC non-convergence: fall back to scipy.optimize least-squares
- Compute physical parameters (Rp, habitable zone proximity)
- Write parameter posteriors and credible intervals to catalog

### 5.9 Reporting Agent [`src/agents/reporting_agent.py`]

**Responsibilities**:
- Aggregate all candidate data into master DataFrame
- Generate all 7 required visualizations per transit candidate
- Produce ranked CSV catalog sorted by SNR descending
- Produce FITS binary table version of catalog
- Render 3-page PDF report using Jinja2 template
- Generate per-target HTML summary pages

### 5.10 Active Learning Agent [`src/agents/active_learning_agent.py`]

**Responsibilities**:
- Collect low-confidence candidates (max_proba < 0.65)
- Present them to the human labeling interface (Streamlit app)
- Receive expert labels from the frontend
- Append newly labeled samples to training set
- Trigger automatic fine-tuning of the ensemble model
- Track annotation history and compute inter-annotator agreement

### 5.11 Inter-Agent Communication Protocol

```python
# Standard message envelope
{
    "sender": "detection_agent",
    "recipient": "feature_agent",
    "action": "process_candidates",
    "payload": {
        "candidates": [...],      # list of candidate dicts
        "sector": 1,
        "timestamp": "2026-06-01T12:00:00Z"
    },
    "status": "request"           # request | done | error
}
```

---

## 6. Frontend Dashboard — Step-by-Step

The frontend is a **Streamlit-based interactive dashboard** with an active-learning labeling interface.

### 6.1 Dashboard Architecture

```
frontend/
├── app.py                   # Main Streamlit entry point
├── pages/
│   ├── 1_Overview.py        # Sector-level summary statistics
│   ├── 2_Candidate_Table.py # Searchable/filterable candidate catalog
│   ├── 3_Target_Detail.py   # Per-target deep dive with all 7 plots
│   ├── 4_Label_Review.py    # Active learning labeling interface
│   └── 5_Model_Metrics.py   # Confusion matrix, PR curves, SHAP plots
├── components/
│   ├── light_curve_plot.py  # Reusable plotly/bokeh plot components
│   ├── corner_plot.py       # MCMC posterior corner plot
│   └── candidate_card.py    # Summary card per candidate
└── static/
    └── styles.css           # Custom CSS for branding
```

### 6.2 Step-by-Step Frontend Implementation

#### Step 6.1 — Setup & Installation
```bash
pip install streamlit plotly bokeh pandas numpy
streamlit run frontend/app.py
```

#### Step 6.2 — Overview Page (`1_Overview.py`)
Display sector-level statistics:
- **KPI Cards**: Total targets processed, candidates detected, confirmed transits, EBs, blends
- **Pie chart**: Class distribution of all candidates
- **Timeline plot**: Sector data coverage with quality-flagged gaps highlighted
- **Noise histogram**: Distribution of RMS scatter across all processed stars

#### Step 6.3 — Candidate Table Page (`2_Candidate_Table.py`)
Interactive sortable/filterable catalog:
```python
import streamlit as st
import pandas as pd

df = pd.read_csv("outputs/candidates_sector_01.csv")
st.dataframe(
    df[['tic_id','period','depth','SNR','predicted_class','confidence']],
    column_config={
        "confidence": st.column_config.ProgressColumn(min_value=0, max_value=1),
        "predicted_class": st.column_config.SelectboxColumn(
            options=["Transit","EB","Blend","Other"])
    }
)
```
- **Filter sidebar**: Filter by class, SNR range, period range, stellar Teff
- **Download button**: Export filtered view as CSV
- **Row click**: Navigate to Target Detail page for the selected TIC ID

#### Step 6.4 — Target Detail Page (`3_Target_Detail.py`)
Display all 7 required visualizations for a selected TIC ID:

| Plot | Description |
|---|---|
| 1. **Raw Light Curve** | Full time series with transit epochs and quality flags highlighted |
| 2. **Detrended Light Curve** | After preprocessing, showing noise reduction |
| 3. **BLS/TLS Periodogram** | Power spectrum with peak period annotation and FAP threshold |
| 4. **Phase-Folded Light Curve** | With batman model overlay and 1σ uncertainty band |
| 5. **MCMC Corner Plot** | Posterior distributions for P, δ, T14, T0, b |
| 6. **Odd/Even Transit Comparison** | Side-by-side comparison for EB discrimination |
| 7. **Centroid Motion Plot** | Pixel-level centroid shift during transit events |

```python
# Interactive Plotly light curve
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter(x=time, y=flux, mode='lines', name='PDCSAP Flux'))
fig.add_trace(go.Scatter(x=transit_times, y=transit_flux, mode='markers',
                          marker=dict(color='red', size=5), name='Transit Epochs'))
st.plotly_chart(fig, use_container_width=True)
```

#### Step 6.5 — Label Review Page (`4_Label_Review.py`)
Active learning interface for expert vetting:

```python
# Load low-confidence candidates
low_conf = df[df['confidence'] < 0.65]
for _, row in low_conf.iterrows():
    st.subheader(f"TIC {row['tic_id']} — Confidence: {row['confidence']:.2f}")
    show_phase_folded(row['tic_id'])
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("Transit", key=f"t_{row['tic_id']}"):
            save_label(row['tic_id'], 0)
    with col2:
        if st.button("EB", key=f"e_{row['tic_id']}"):
            save_label(row['tic_id'], 1)
    with col3:
        if st.button("Blend", key=f"b_{row['tic_id']}"):
            save_label(row['tic_id'], 2)
    with col4:
        if st.button("Other", key=f"o_{row['tic_id']}"):
            save_label(row['tic_id'], 3)
```

- Labels are saved to `data/catalogs/human_labels.csv`
- When batch size >= 50 new labels, trigger model re-training automatically

#### Step 6.6 — Model Metrics Page (`5_Model_Metrics.py`)
Display model performance diagnostics:
- **Confusion Matrix**: Interactive heatmap using plotly
- **Precision-Recall Curves**: Per class, with AUC annotations
- **ROC Curves**: Per class
- **SHAP Summary Plot**: Top 20 features by importance (beeswarm plot)
- **Calibration Curve**: Reliability diagram to check probability calibration
- **Training History**: Loss and F1 curves per epoch/round

---

## 7. Technology Stack

| Package | Version | Purpose |
|---|---|---|
| `lightkurve` | >=2.4 | TESS FITS access, light curve utilities |
| `transitleastsquares` | >=1.0 | TLS periodogram for transit detection |
| `batman-package` | >=2.4 | Mandel-Agol transit model |
| `emcee` | >=3.1 | MCMC Bayesian parameter estimation |
| `wotan` | >=1.10 | Light curve detrending |
| `celerite2` | >=0.2 | Gaussian Process systematics correction |
| `xgboost` | >=1.7 | Gradient boosted classification |
| `lightgbm` | >=3.3 | Gradient boosted classification |
| `scikit-learn` | >=1.3 | ML utilities, calibration |
| `imbalanced-learn` | >=0.11 | SMOTE oversampling |
| `astropy` / `astroquery` | >=5.3 / >=0.4 | FITS I/O, MAST API |
| `tsfresh` | >=0.20 | Time-series feature extraction |
| `bokeh` / `plotly` | >=3.3 / >=5.17 | Interactive visualization |
| `torch` (optional) | >=2.0 | 1-D CNN classification |
| `optuna` | >=3.4 | Hyperparameter optimization |
| `shap` | >=0.44 | Model explainability |
| `streamlit` | >=1.30 | Interactive web dashboard |
| `Jinja2` | >=3.1 | PDF report templating |
| `joblib` | >=1.3 | Parallelization |

### Installation
```bash
conda create -n exoplanet python=3.10
conda activate exoplanet
pip install lightkurve transitleastsquares batman-package emcee wotan celerite2 xgboost lightgbm scikit-learn imbalanced-learn astropy astroquery tsfresh bokeh plotly optuna shap streamlit Jinja2 joblib
```

---

## 8. Implementation Roadmap & Phases

| Phase | Name | Key Deliverables | Duration | Priority |
|---|---|---|---|---|
| **Phase 0** | Setup & Data Ingestion | Env setup, MAST download, EDA on 1 sector | Day 1 | P0 |
| **Phase 1** | Preprocessing & BLS | Detrending, TLS search, candidate list | Days 1-2 | P0 |
| **Phase 2** | Feature Engineering | 35+ features extracted per candidate | Day 2 | P0 |
| **Phase 3** | ML Classifier | Trained ensemble, >=90% F1, SMOTE | Days 2-3 | P0 |
| **Phase 4** | Parameter Fitting | batman + emcee MCMC, credible intervals | Day 3 | P0 |
| **Phase 5** | Visualization & Report | Plots, catalog CSV, 3-page PDF report | Days 3-4 | P0 |
| **Phase 6 (Bonus)** | Enhanced Features | Centroid analysis, injection-recovery, active learning | Day 4 | P1 |

### Detailed Day-by-Day Schedule

| Day | Time | Task |
|---|---|---|
| Day 1 | Morning | Environment setup, install all dependencies, test imports |
| Day 1 | Afternoon | Acquisition Agent: MAST query, FITS download for Sector 1, EDA notebook |
| Day 2 | Morning | Preprocessing Agent: detrending, quality gating, save .npz files |
| Day 2 | Afternoon | Detection Agent: TLS periodogram, candidate list; Feature Extraction Agent start |
| Day 3 | Morning | Feature Extraction Agent complete; train XGBoost/LightGBM ensemble |
| Day 3 | Afternoon | Optuna hyperparameter search; Platt calibration; evaluate on test set |
| Day 4 | Morning | Parameter Fitting Agent: batman + emcee MCMC for transit candidates |
| Day 4 | Afternoon | Reporting Agent: plots, CSV catalog, FITS table, 3-page PDF |
| Day 4 | Evening | Streamlit dashboard; active learning interface; integration tests |

---

## 9. Testing & Validation Strategy

### Unit Tests (`tests/`)

| Module | Test |
|---|---|
| **DAM** | Mock MAST API responses; test QUALITY bitmask filtering |
| **PPM** | Test sigma-clipping on synthetic noisy arrays; test detrending on known sinusoid |
| **PDM** | Validate TLS recovers injected transit period within 0.1% |
| **FEM** | Test feature determinism; test NaN handling |
| **CLM** | Test calibrated probabilities sum to 1; test on known labeled subset |
| **PER** | Test batman model generates valid transit shape; test MCMC convergence |

### Integration Tests

1. **End-to-end run** on TESS Sector 1, Camera 1, CCD 1 (reference sector with many confirmed planets)
2. **Benchmark against TOI catalog**: require >=80% of confirmed planets recovered at SDE >= 7
3. **False positive rate**: must be <2% on background non-variable stars
4. **Runtime check**: full sector must complete in <=90 minutes on 8-core CPU

### Model Validation

- **Confusion matrix** on held-out test set for all 4 classes
- **Precision-Recall curves** for transit class at multiple thresholds
- **SHAP analysis**: physically meaningful features (depth, SDE, odd/even ratio) must rank in top 10
- **Calibration plot**: reliability diagram must lie within 5% of the diagonal

---

## 10. Risks & Mitigations

| Risk | Impact | Severity | Mitigation |
|---|---|---|---|
| MAST download bandwidth limits | Sector download incomplete | HIGH | Pre-cache data; use MAST bulk download tools; implement resume |
| Severe class imbalance | Classifier biased to majority class | HIGH | SMOTE + class-weighted loss; injection-recovery augmentation |
| Poor detrending masking real transits | Missed detections | MEDIUM | Compare multiple detrending algorithms; inject-recover test |
| MCMC non-convergence | Unreliable parameter uncertainties | MEDIUM | Set max iteration limits; fall back to least-squares fit |
| Unrecognized signal morphologies | Misclassification | LOW | Out-of-distribution detector; flag uncertain cases for review |

---

## 11. Acceptance Criteria (Go/No-Go Checklist)

The system is submission-ready when **ALL** of the following are met:

- [ ] Pipeline processes a full TESS sector (25k+ targets) end-to-end without crashing
- [ ] BLS/TLS detection recovers >=80% of known TOIs present in the processed sector at SDE >= 7
- [ ] Classifier achieves **>=90% macro-F1** on held-out test set across all 4 classes
- [ ] Period estimates for confirmed planets within **5%** of catalog values
- [ ] Transit depth and duration estimates within **10%** of published values
- [ ] Every candidate receives a calibrated confidence score between 0 and 1
- [ ] All **7 required visualizations** generated per transit candidate
- [ ] Output CSV catalog includes all required columns with no null values for transit candidates
- [ ] 3-page PDF report generated automatically by the pipeline
- [ ] Runtime for a full sector <= **90 minutes** on 8-core CPU without GPU

---

*Document prepared based on College_Project.pdf — AI-Enabled Exoplanet Detection System PRD v1.0 (June 2026)*
