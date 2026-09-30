# Comprehensive Multi-File CSV Data Analysis — LunaVisionAI Exoplanet Pipeline

This analysis covers **10 CSV files** constituting a complete exoplanet detection and classification pipeline built around **TESS (Transiting Exoplanet Survey Satellite)** data.

> **Update 2026-09-19 — read before using the numbers below.** This report describes the CSVs as they were at analysis time. Since then the pipeline was re-run on much more data and several data-quality issues were fixed, so specific counts and values here are out of date:
> - `feature_matrix.csv` (40 rows here) is stale: the real light-curve set grew from 27 to **2,360 downloaded / 2,120 usable** (1,423 planet-side, 697 non-planet) and features are being regenerated. Earlier `depth_ppm` values (~99%) were wrong (TLS depth-semantics bug, fixed); `label` was mostly -1 for real targets (ID-prefix mismatch, fixed); the file had appended/misaligned rows (schema-drift bug, fixed).
> - `synthetic_injections.csv` (20 rows here) now holds **199** detectable injections on label-3 hosts (≥3 transits, SNR ≥ 8); previously some were non-transiting or unrecoverable.
> - `preprocessing_report.csv` (48 rows here) now covers ~2,560 light curves; the noise gate and sigma-clip were both fixed (see `SAAS_ARCHITECTURE_PLAN.md` Appendix A.2).
> - Catalog files (`exofop_toi_labeled.csv`, `nasa_confirmed_labeled.csv`, `unified_labels.csv`) are unchanged in structure; the light curves downloaded cover only a labeled subset, not every catalog row.

---

## 1. Overall Project Overview

This dataset collection forms a **machine-learning pipeline for exoplanet transit detection and classification** using TESS light curves. The pipeline ingests raw catalog data (NASA confirmed planets, ExoFOP TOIs), performs light-curve preprocessing, extracts time-series features, injects synthetic transits for training, and produces a unified label set and feature matrix for ML model training.

| # | File | Rows | Columns | Main Purpose | Potential Key |
|---|------|-----:|--------:|--------------|---------------|
| 1 | `exofop_toi_labeled.csv` | 8,149 | 18 | ExoFOP TOI catalog with curated labels | `tic_id` + `toi` |
| 2 | `feature_matrix.csv` | 40 | 71 | Engineered features for ML training/inference | `tic_id` |
| 3 | `nasa_confirmed_labeled.csv` | 6,367 | 14 | NASA Exoplanet Archive confirmed planets with labels | `tic_id` + `pl_name` |
| 4 | `preprocessing_report.csv` | 48 | 8 | Light-curve preprocessing pipeline status log | `ic_id` (tic_id) |
| 5 | `spoc_tce_sector01.csv` | 4 | 14 | SPOC Threshold Crossing Events for Sector 1 | `tic_id` + `tce_ext` |
| 6 | `synthetic_injections.csv` | 20 | 10 | Synthetic transit injection parameters | `tic_id` (SYN_*) |
| 7 | `tic_stellar_params.csv` | 501 | 100 | Full TIC stellar parameter catalog subset | `tic_id` |
| 8 | `unified_labels.csv` | 10,965 | 3 | Merged label set from all sources | `tic_id` |
| 9 | `PS_2026.09.14_15.07.20.csv` | 6,367 | 13 | NASA Planetary Systems raw download | `tic_id` + `pl_name` |
| 10 | `tois_list.csv` | 8,149 | 63 | Full ExoFOP TESS TOI list (raw download) | `TIC ID` + `TOI` |

---

## 2. File-by-File Analysis

---

### File 1 — `exofop_toi_labeled.csv`

**Source**: ExoFOP (Exoplanet Follow-up Observing Program) TESS Object of Interest catalog, post-processed with binary labels.

| Aspect | Value |
|--------|-------|
| Rows | 8,149 |
| Columns | 18 |
| Duplicate rows | 0 |
| Total missing | 1,068 |
| Numerical features | 14 |
| Categorical features | 2 (tfop_disposition, sectors) |
| Date/time features | 0 |
| Identifier columns | tic_id, toi |
| Possible target | `label` |

#### Feature Dictionary

| # | Feature Name | Meaning | Data Type | Feature Type | Example Values | Unique | Missing | Missing % | Unit | Role | Source |
|---|---|---|---|---|---|---:|---:|---:|---|---|---|
| 1 | `tic_id` | TESS Input Catalog unique identifier for the host star | int64 | Identifier | 231663901, 149603524 | 7,519 | 0 | 0% | — | Identifier | ExoFOP |
| 2 | `toi` | TESS Object of Interest designation number (star.planet) | float64 | Identifier | 101.01, 102.01 | 8,149 | 0 | 0% | — | Identifier | ExoFOP |
| 3 | `ra` | Right Ascension of the host star (sexagesimal) | object | Numerical (encoded) | "21:14:56.88" | ~7,519 | 0 | 0% | hh:mm:ss | Input | ExoFOP |
| 4 | `dec` | Declination of the host star (sexagesimal) | object | Numerical (encoded) | "-55:52:18.71" | ~7,519 | 0 | 0% | dd:mm:ss | Input | ExoFOP |
| 5 | `period` | Orbital period of the planet candidate | float64 | Numerical | 1.43, 4.41, 3.55 | ~8,036 | 0 | 0% | days | Input | ExoFOP |
| 6 | `depth_ppm` | Transit depth in parts per million | float64 | Numerical | 18960.7, 14711.9 | ~6,306 | 0 | 0% | ppm | Input | ExoFOP |
| 7 | `duration_hr` | Transit duration | float64 | Numerical | 1.62, 3.63, 3.49 | ~6,195 | ~4 | 0.05% | hours | Input | ExoFOP |
| 8 | `epoch_bjd` | Mid-transit time in Barycentric Julian Date | float64 | Numerical | 2458326.01 | ~8,147 | 0 | 0% | BJD | Input | ExoFOP |
| 9 | `stellar_Teff` | Stellar effective temperature | float64 | Numerical | 5600, 6280, 6351 | ~4,674 | ~175 | 2.15% | Kelvin | Input | TIC |
| 10 | `stellar_logg` | Surface gravity of host star (log10) | float64 | Numerical | 4.49, 4.32, 4.23 | ~2,115 | ~360 | 4.4% | log10(cm/s²) | Input | TIC |
| 11 | `stellar_rad` | Stellar radius | float64 | Numerical | 0.89, 1.21, 1.4 | ~2,800 | ~180 | 2.2% | R_Sun | Input | TIC |
| 12 | `stellar_mass` | Stellar mass | float64 | Numerical | 1.05, 1.28, 1.27 | ~1,502 | ~330 | 4% | M_Sun | Input | TIC |
| 13 | `tess_mag` | TESS-band apparent magnitude | float64 | Numerical | 12.41, 9.71, 11.52 | ~7,252 | 0 | 0% | mag | Input | TIC |
| 14 | `planet_rad_earth` | Estimated planet radius | float64 | Numerical | 13.19, 15.06 | ~7,595 | ~50 | 0.6% | R_Earth | Input | ExoFOP |
| 15 | `planet_snr` | Signal-to-noise ratio of the transit detection | float64 | Numerical | 151.7, 3217.2, 59.5 | ~3,252 | 0 | 0% | — | Input | ExoFOP |
| 16 | `tfop_disposition` | TFOPWG follow-up disposition code | object | Categorical | KP, CP, FP, PC, APC, FA | 6 | 0 | 0% | — | Metadata | ExoFOP |
| 17 | `label` | Binary classification target (0=planet, 3=false positive) | int64 | Binary/Target | 0, 3 | 2 | 0 | 0% | — | **Target** | Derived |
| 18 | `sectors` | TESS observation sectors (comma-separated) | object | Categorical (multi-value) | "1,27,67" | ~2,670 | 0 | 0% | — | Metadata | ExoFOP |

**Data Quality Issues:**
- **Confirmed**: `label` uses values 0 and 3 (not 0/1) — unusual binary encoding. 0 = planet (CP/KP/PC/APC/VP), 3 = false positive (FP/FA).
- **Confirmed**: `stellar_mass` has ~4% missing values.
- **Possible**: `ra`/`dec` are in sexagesimal string format — need conversion for ML use.
- **Possible**: `sectors` is a multi-value string field — needs parsing/encoding.

---

### File 2 — `feature_matrix.csv`

**Source**: Pipeline-generated feature matrix combining transit search results, light-curve statistics, stellar parameters, and FFT coefficients for ML training.

| Aspect | Value |
|--------|-------|
| Rows | 40 |
| Columns | 71 |
| Duplicate rows | 2 (SYN_0000=SYN_0012, SYN_0003=SYN_0019) |
| Total missing | Variable (many sparse columns) |
| Numerical features | 68 |
| Categorical features | 1 (status) |
| Identifier columns | tic_id |
| Possible target | `label` |

#### Feature Dictionary (grouped by category)

**Identifiers & Metadata (3 features)**

| # | Feature | Meaning | Type | Example | Missing % | Unit | Role |
|---|---------|---------|------|---------|-----------|------|------|
| 1 | `tic_id` | TIC ID or synthetic injection ID | object | SYN_0000, TIC_144065872 | 0% | — | Identifier |
| 2 | `label` | Classification label (-1 = unlabeled/synthetic planet) | int64 | -1 | 0% | — | Target |
| 3 | `sector` | TESS sector observed (-1 for synthetic) | int64 | -1, 1 | 0% | — | Metadata |
| 4 | `status` | Processing pipeline status | object | "done" | 0% | — | Metadata |

**Observation Parameters (3 features)**

| # | Feature | Meaning | Type | Unit | Role |
|---|---------|---------|------|------|------|
| 5 | `baseline_days` | Duration of observation baseline | float64 | days | Input |
| 6 | `n_cadences` | Number of photometric data points | int64 | count | Input |

**Transit Search Results (13 features)**

| # | Feature | Meaning | Unit | Role |
|---|---------|---------|------|------|
| 7 | `period` | Detected orbital period from BLS search | days | Input |
| 8 | `depth` | Normalized transit depth (fractional flux loss) | — | Input |
| 9 | `depth_ppm` | Transit depth in parts per million | ppm | Input |
| 10 | `duration_hr` | Transit duration | hours | Input |
| 11 | `transit_count` | Number of transits detected in the light curve | count | Input |
| 12 | `SDE` | Signal Detection Efficiency from BLS algorithm | — | Input |
| 13 | `SNR` | Signal-to-noise ratio of transit detection | — | Input |
| 14 | `FAP` | False Alarm Probability of the detection | — | Input |
| 15 | `t0` | Mid-transit time reference (first transit) | BTJD | Input |
| 16 | `rp_rs` | Planet-to-star radius ratio | — | Input |
| 17 | `phase_coverage` | Fraction of orbital phase with data coverage | — | Input |

**Transit Shape Diagnostics (8 features)**

| # | Feature | Meaning | Role |
|---|---------|---------|------|
| 18 | `flat_bottom_score` | Flatness metric of transit bottom (U vs V shape) | Input |
| 19 | `phase_folded_std` | Standard deviation of phase-folded light curve | Input |
| 20 | `ingress_egress_asym` | Asymmetry between ingress and egress duration | Input |
| 21 | `secondary_depth` | Depth of secondary eclipse | Input |
| 22 | `secondary_ratio` | Ratio of secondary to primary eclipse depth | Input |
| 23 | `odd_depth` | Transit depth of odd-numbered transits | Input |
| 24 | `even_depth` | Transit depth of even-numbered transits | Input |
| 25 | `odd_even_ratio` | Ratio of odd to even transit depths | Input |

**Centroid & Habitability (2 features)**

| # | Feature | Meaning | Role |
|---|---------|---------|------|
| 26 | `centroid_proxy` | Proxy for centroid shift during transit (false-positive indicator) | Input |
| 27 | `hz_proximity` | Proximity to host star's habitable zone | Input |

**Stellar Parameters (8 features)**

| # | Feature | Meaning | Unit | Role |
|---|---------|---------|------|------|
| 28 | `stellar_Teff` | Effective temperature | K | Input |
| 29 | `stellar_logg` | Surface gravity | log10(cm/s²) | Input |
| 30 | `stellar_rad` | Stellar radius | R_Sun | Input |
| 31 | `stellar_mass` | Stellar mass | M_Sun | Input |
| 32 | `stellar_dist_pc` | Distance to star | parsecs | Input |
| 33 | `stellar_metallicity` | Metallicity [Fe/H] | dex | Input |
| 34 | `tess_mag` | TESS-band magnitude | mag | Input |
| 35 | `hz_in_zone` | Whether planet is in habitable zone (binary) | — | Input |

**Light-Curve Statistics (14 features)**

| # | Feature | Meaning | Role |
|---|---------|---------|------|
| 36 | `rms` | Root-mean-square of detrended flux | Input |
| 37 | `rms_raw` | RMS of raw (pre-detrended) flux | Input |
| 38 | `skewness` | Skewness of flux distribution | Input |
| 39 | `kurtosis` | Kurtosis of flux distribution | Input |
| 40 | `autocorr_lag1` | Autocorrelation at lag 1 | Input |
| 41 | `autocorr_lag10` | Autocorrelation at lag 10 | Input |
| 42 | `flux_range` | Range of flux values (max - min) | Input |
| 43 | `flux_percentile_5` | 5th percentile of flux | Input |
| 44 | `flux_percentile_95` | 95th percentile of flux | Input |
| 45 | `above_3sigma_frac` | Fraction of data points >3σ above mean | Input |
| 46 | `below_3sigma_frac` | Fraction of data points >3σ below mean | Input |

**Time-Series Feature (tsfresh) Statistics (7 features)**

| # | Feature | Meaning | Role |
|---|---------|---------|------|
| 47 | `tsf_mean` | Mean of light-curve flux | Input |
| 48 | `tsf_variance` | Variance of flux | Input |
| 49 | `tsf_abs_energy` | Absolute energy (sum of squared values) | Input |
| 50 | `tsf_mean_abs_change` | Mean absolute change between consecutive values | Input |
| 51 | `tsf_maximum` | Maximum flux value | Input |
| 52 | `tsf_minimum` | Minimum flux value | Input |
| 53 | `tsf_median` | Median flux value | Input |

**FFT Coefficients (20 features: tsf_fft_coeff_0 through tsf_fft_coeff_9, each with _real and _abs)**

| # | Feature Pattern | Meaning | Role |
|---|-----------------|---------|------|
| 54–71 | `tsf_fft_coeff_{k}_real`, `tsf_fft_coeff_{k}_abs` | Real and absolute components of k-th FFT coefficient (k=0..9) | Input |

**Data Quality Issues:**
- **Confirmed**: Only 40 rows — very small training set (20 synthetic + 20 real TIC targets).
- **Confirmed**: 2 duplicate rows (SYN_0000 duplicates SYN_0012, SYN_0003 duplicates SYN_0019).
- **Confirmed**: All labels are -1 — these are pre-labeled "planet" class; no false-positive examples yet in this file.
- **Confirmed**: Many columns have missing values (FAP, flat_bottom_score, secondary fields, centroid_proxy, hz_proximity).
- **Possible**: `depth` and `depth_ppm` are near-redundant (depth ≈ 1 - depth_ppm/1e6).
- **Confirmed**: `stellar_metallicity`, `tess_mag`, `hz_in_zone`, `hz_proximity` are entirely or mostly missing for synthetic rows.

---

### File 3 — `nasa_confirmed_labeled.csv`

**Source**: NASA Exoplanet Archive Planetary Systems table, filtered and labeled.

| Aspect | Value |
|--------|-------|
| Rows | 6,367 |
| Columns | 14 |
| Duplicate rows | 0 |
| Total missing | ~14,700 |
| Possible target | `label` |

#### Feature Dictionary

| # | Feature | Meaning | Type | Unique | Missing % | Unit | Role |
|---|---------|---------|------|-------:|----------:|------|------|
| 1 | `pl_name` | Planet name | object | 6,367 | 0% | — | Identifier |
| 2 | `hostname` | Host star name | object | 4,809 | 0% | — | Identifier |
| 3 | `tic_id` | TIC identifier (numeric, some missing for non-TESS) | float64/int | 5,766 | ~9.4% | — | Identifier |
| 4 | `period` | Orbital period | float64 | 5,875 | ~3.7% | days | Input |
| 5 | `depth_ppm` | Transit depth | float64 | 813 | ~85.7% | ppm | Input |
| 6 | `duration_hr` | Transit duration | float64 | 3,023 | ~46% | hours | Input |
| 7 | `stellar_Teff` | Stellar effective temperature | float64 | 3,283 | ~8.3% | K | Input |
| 8 | `stellar_logg` | Surface gravity | float64 | 3,091 | ~13.3% | log10(cm/s²) | Input |
| 9 | `stellar_rad` | Stellar radius | float64 | 3,564 | ~5% | R_Sun | Input |
| 10 | `stellar_mass` | Stellar mass | float64 | 2,568 | ~5.6% | M_Sun | Input |
| 11 | `label` | Binary label (all 0 = confirmed planet) | int64 | 1 | 0% | — | **Target** |
| 12 | `tfop_disposition` | TFOP disposition (all CP) | object | 1 | 0% | — | Metadata |
| 13 | `discovery_method` | Planet discovery method | object | 10 | 0% | — | Metadata |
| 14 | `facility` | Discovery facility/telescope | object | 137 | 0% | — | Metadata |

**Data Quality Issues:**
- **Confirmed**: `label` is constant (all 0) — this is a positive-only set.
- **Confirmed**: `tfop_disposition` is constant (all "CP") — no discriminative power.
- **Confirmed**: `depth_ppm` is 85.7% missing — many non-transit discoveries (radial velocity, microlensing).
- **Confirmed**: ~9.4% of rows lack `tic_id` (non-TESS targets like microlensing events).
- **Possible**: `tic_id` format differs from other files (numeric here vs. "TIC_" prefix elsewhere).

---

### File 4 — `preprocessing_report.csv`

**Source**: Light-curve detrending pipeline status report.

| Aspect | Value |
|--------|-------|
| Rows | 48 |
| Columns | 8 |
| Duplicate rows | 0 |
| Total missing | 42 (7 rows × 5 columns gated) |

#### Feature Dictionary

| # | Feature | Meaning | Type | Missing % | Unit | Role |
|---|---------|---------|------|----------:|------|------|
| 1 | `ic_id` | TIC/Synthetic ID (note: column name typo for "tic_id") | object | 0% | — | Identifier |
| 2 | `status` | Pipeline status: "done" or "gated" | object | 0% | — | Metadata |
| 3 | `reason` | Status reason ("ok" or rejection reason) | object | 0% | — | Metadata |
| 4 | `out_path` | File path to detrended light curve (.npz) | object | 14.6% | — | Metadata |
| 5 | `n_cadences` | Number of flux measurements after processing | float64 | 14.6% | count | Metadata |
| 6 | `baseline_days` | Observation duration | float64 | 14.6% | days | Metadata |
| 7 | `rms_raw` | RMS of raw light curve | float64 | 14.6% | — | Metadata |
| 8 | `rms_detrended` | RMS of detrended light curve | float64 | 14.6% | — | Metadata |

**Data Quality Issues:**
- **Confirmed**: Column name `ic_id` is likely a typo for `tic_id`.
- **Confirmed**: 7 rows (14.6%) are "gated" (quality-rejected) with null numeric fields.
- **Confirmed**: `out_path` contains absolute paths from a macOS filesystem — not portable.

---

### File 5 — `spoc_tce_sector01.csv`

**Source**: SPOC (Science Processing Operations Center) Threshold Crossing Events from TESS Sector 1.

| Aspect | Value |
|--------|-------|
| Rows | 4 |
| Columns | 14 |
| Duplicate rows | 0 |
| Total missing | 2 (snr column) |

#### Feature Dictionary

| # | Feature | Meaning | Type | Missing % | Unit | Role |
|---|---------|---------|------|----------:|------|------|
| 1 | `tic_id` | TIC identifier | int64 | 0% | — | Identifier |
| 2 | `tce_ext` | TCE extension label (e.g., TCE_1, TCE_2) | object | 0% | — | Identifier |
| 3 | `period` | Detected orbital period | float64 | 0% | days | Input |
| 4 | `epoch_btjd` | Mid-transit epoch in TESS Barycentric Julian Date | float64 | 0% | BTJD | Input |
| 5 | `depth_ppm` | Transit depth | float64 | 0% | ppm | Input |
| 6 | `duration_hr` | Transit duration | float64 | 0% | hours | Input |
| 7 | `snr` | Signal-to-noise ratio | float64 | 50% | — | Input |
| 8 | `impact` | Impact parameter (transit chord position) | float64 | 0% | — | Input |
| 9 | `rad_ratio` | Planet-to-star radius ratio (Rp/Rs) | float64 | 0% | — | Input |
| 10 | `stellar_Teff` | Stellar effective temperature | float64 | 0% | K | Input |
| 11 | `tess_mag` | TESS magnitude | float64 | 0% | mag | Input |
| 12 | `label` | Classification label (all -1 = unclassified) | int64 | 0% | — | Target |
| 13 | `source` | Source pipeline identifier | object | 0% | — | Metadata |
| 14 | `sector` | TESS sector number | int64 | 0% | — | Metadata |

**Data Quality Issues:**
- **Confirmed**: Only 4 rows — minimal data, single star (TIC 265591866).
- **Confirmed**: All labels are -1 (unclassified).
- **Confirmed**: `snr` has 50% missing.
- **Possible**: Same `tic_id` (265591866) appears with different `stellar_Teff` values (6242 and 7381) — suggests multiple stellar parameter sources or pipeline runs.

---

### File 6 — `synthetic_injections.csv`

**Source**: Synthetic transit injection log — parameters used to create artificial transit signals in real light curves for ML training.

| Aspect | Value |
|--------|-------|
| Rows | 20 |
| Columns | 10 |
| Duplicate rows | 0 |
| Total missing | 0 |

#### Feature Dictionary

| # | Feature | Meaning | Type | Unit | Role |
|---|---------|---------|------|------|------|
| 1 | `tic_id` | Synthetic ID (SYN_XXXX) | object | — | Identifier |
| 2 | `host_path` | File path to host star's raw light curve | object | — | Metadata |
| 3 | `period` | Injected orbital period | float64 | days | Input |
| 4 | `rp` | Injected planet radius (relative to star) | float64 | R_star | Input |
| 5 | `a` | Injected semi-major axis | float64 | R_star | Input |
| 6 | `inc` | Injected orbital inclination | float64 | degrees | Input |
| 7 | `t0` | Injected mid-transit time | float64 | BTJD | Input |
| 8 | `depth_ppm` | Resulting transit depth | float64 | ppm | Input |
| 9 | `label` | Classification label (all 0 = planet) | int64 | — | Target |
| 10 | `npz_path` | File path to injected light curve | object | — | Metadata |

**Data Quality Issues:**
- **Confirmed**: All labels are 0 (planet) — no false positives.
- **Confirmed**: `host_path` and `npz_path` contain macOS absolute paths.

---

### File 7 — `tic_stellar_params.csv`

**Source**: Full TESS Input Catalog (TIC) extract with ~100 photometric, astrometric, and derived stellar parameters.

| Aspect | Value |
|--------|-------|
| Rows | 501 |
| Columns | 100 |
| Duplicate rows | 0 |
| Total missing | ~6,000+ |

This file contains the most extensive stellar characterization data. The 100 columns span:

- **Identifiers (10)**: `tic_id`, `version`, `HIP`, `TYC`, `UCAC`, `TWOMASS`, `SDSS`, `ALLWISE`, `GAIA`, `APASS`, `KIC`, `objID`
- **Object Classification (2)**: `objType`, `typeSrc`
- **Astrometry (12)**: `ra`, `dec`, `POSflag`, `pmRA`, `e_pmRA`, `pmDEC`, `e_pmDEC`, `PMflag`, `plx`, `e_plx`, `PARflag`, coordinates (galactic, ecliptic)
- **Photometry (36)**: Magnitudes in B, V, u, g, r, i, z, J, H, K, W1-W4, GAIA, TESS bands with errors and flags
- **Stellar Parameters (22)**: `stellar_Teff`, `stellar_logg`, `stellar_rad`, `stellar_mass`, `rho`, luminosity, distance, metallicity with errors and flags
- **Contamination (3)**: `numcont`, `tic_contratio`
- **Quality Flags (15+)**: `disposition`, `duplicate_id`, `priority`, `EBVflag`, `distflag`, `TeffFlag`, `VmagFlag`, `BmagFlag`, `starchareFlag`, `wdflag`, `raddflag`, `gaiaqflag`, `splists`

**Data Quality Issues:**
- **Confirmed**: Extensive missing values across photometric bands (not all stars observed in all surveys).
- **Confirmed**: Some columns are near-constant for this subset (e.g., `objType` = STAR).
- **Possible**: Error columns (`e_*`) may have systematic patterns of missingness correlated with data source.

---

### File 8 — `unified_labels.csv`

**Source**: Merged label set combining all sources into a single lookup table.

| Aspect | Value |
|--------|-------|
| Rows | 10,965 |
| Columns | 3 |
| Duplicate rows | 0 |
| Total missing | 0 |

#### Feature Dictionary

| # | Feature | Meaning | Type | Unique | Role |
|---|---------|---------|------|-------:|------|
| 1 | `tic_id` | TIC identifier (numeric) | int64 | 10,965 | Identifier |
| 2 | `label` | Binary label (0 = planet/candidate, presumably 1/3 = false positive) | int64 | 2–3 | **Target** |
| 3 | `source` | Origin of the label | object | ~5 | Metadata |

Sources observed: `nasa_confirmed`, `exofop_toi`, `spoc_tce`, and possibly `synthetic`.

**Data Quality Issues:**
- **Confirmed**: No missing values.
- **Possible**: Label distribution is heavily imbalanced (mostly 0 from NASA confirmed).

---

### File 9 — `PS_2026.09.14_15.07.20.csv`

**Source**: NASA Exoplanet Archive "Planetary Systems" table, raw download dated 2026-09-14.

| Aspect | Value |
|--------|-------|
| Rows | 6,367 |
| Columns | 13 |
| Duplicate rows | 0 |
| Total missing | ~14,500 |

#### Feature Dictionary

| # | Feature | Meaning | Type | Unit | Role |
|---|---------|---------|------|------|------|
| 1 | `pl_name` | Planet name | object | — | Identifier |
| 2 | `hostname` | Host star name | object | — | Identifier |
| 3 | `tic_id` | TIC ID (format: "TIC 123456789") | object | — | Identifier |
| 4 | `pl_orbper` | Orbital period | float64 | days | Input |
| 5 | `pl_trandep` | Transit depth | float64 | — | Input |
| 6 | `pl_trandur` | Transit duration | float64 | hours | Input |
| 7 | `st_teff` | Stellar effective temperature | float64 | K | Input |
| 8 | `st_logg` | Stellar surface gravity | float64 | log10(cm/s²) | Input |
| 9 | `st_rad` | Stellar radius | float64 | R_Sun | Input |
| 10 | `st_mass` | Stellar mass | float64 | M_Sun | Input |
| 11 | `discoverymethod` | Discovery method | object | — | Metadata |
| 12 | `disc_facility` | Discovery facility | object | — | Metadata |
| 13 | `default_flag` | Whether this is the default parameter set | int64 | — | Metadata |

**Data Quality Issues:**
- **Confirmed**: `tic_id` format is "TIC 123456789" (string with prefix) — differs from other files.
- **Confirmed**: `pl_trandep` is ~85% missing (non-transit discoveries).
- **Possible**: This is the raw source for `nasa_confirmed_labeled.csv` (same row count, matching columns after renaming).

---

### File 10 — `tois_list.csv`

**Source**: Full ExoFOP TESS TOI list, raw download with all available columns.

| Aspect | Value |
|--------|-------|
| Rows | 8,149 |
| Columns | 63 |
| Duplicate rows | 0 |
| Total missing | ~36,000+ |

This is the most feature-rich file. Key column groups:

- **Identifiers**: `TIC ID`, `TOI`, `Planet Name`, `Pipeline Signal ID`
- **TFOP Priority**: `Master`, `SG1A`, `SG1B`, `SG2`, `SG3`, `SG4`, `SG5` (priority levels 1–5)
- **Atmosphere Metrics**: `ESM` (Emission Spectroscopy Metric), `TSM` (Transmission Spectroscopy Metric)
- **Predicted Properties**: `Predicted Mass (M_Earth)`, `Predicted RV Semi-amplitude (m/s)`
- **Follow-up Counts**: `Time Series Observations`, `Spectroscopy Observations`, `Imaging Observations`
- **Dispositions**: `TESS Disposition`, `TFOPWG Disposition`
- **Stellar Properties**: `TESS Mag`, `Stellar Distance (pc)`, `Stellar Eff Temp (K)`, `Stellar log(g)`, `Stellar Radius`, `Stellar Mass`, `Stellar Metallicity`
- **Planetary Properties**: `Period (days)`, `Duration (hours)`, `Depth (mmag)`, `Depth (ppm)`, `Planet Radius (R_Earth)`, `Planet Insolation`, `Planet Equil Temp`, `Planet SNR`
- **Astrometry**: `RA`, `Dec`, `PM RA`, `PM Dec`
- **Timing**: `Epoch (BJD)`, `Date TOI Alerted`, `Date TOI Updated`, `Date Modified`
- **Other**: `Source`, `Detection`, `Sectors`, `Comments`

**Data Quality Issues:**
- **Confirmed**: `Planet Name` column is 100% empty (NaN) — all are unconfirmed candidates.
- **Confirmed**: `Stellar Metallicity` is 82.5% missing.
- **Confirmed**: `Stellar log(g) err` is 29.4% missing.
- **Confirmed**: Column names contain spaces and special characters — need sanitization.
- **Possible**: `Previous CTOI` is 90% missing.

---

## 3. Cross-File Feature Comparison

### Common Features Across Files

| Concept | exofop_toi_labeled | feature_matrix | nasa_confirmed_labeled | spoc_tce | synthetic_injections | tic_stellar_params | unified_labels | PS_raw | tois_list |
|---------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **TIC ID** | `tic_id` | `tic_id` | `tic_id` | `tic_id` | `tic_id` (SYN) | `tic_id` | `tic_id` | `tic_id` (prefixed) | `TIC ID` |
| **Label** | `label` | `label` | `label` | `label` | `label` | — | `label` | — | — |
| **Period** | `period` | `period` | `period` | `period` | `period` | — | — | `pl_orbper` | `Period (days)` |
| **Depth (ppm)** | `depth_ppm` | `depth_ppm` | `depth_ppm` | `depth_ppm` | `depth_ppm` | — | — | `pl_trandep` | `Depth (ppm)` |
| **Duration** | `duration_hr` | `duration_hr` | `duration_hr` | `duration_hr` | — | — | — | `pl_trandur` | `Duration (hours)` |
| **Stellar Teff** | `stellar_Teff` | `stellar_Teff` | `stellar_Teff` | `stellar_Teff` | — | `stellar_Teff` | — | `st_teff` | `Stellar Eff Temp (K)` |
| **Stellar logg** | `stellar_logg` | `stellar_logg` | `stellar_logg` | — | — | `stellar_logg` | — | `st_logg` | `Stellar log(g)` |
| **Stellar Radius** | `stellar_rad` | `stellar_rad` | `stellar_rad` | — | — | `stellar_rad` | — | `st_rad` | `Stellar Radius` |
| **Stellar Mass** | `stellar_mass` | `stellar_mass` | `stellar_mass` | — | — | `stellar_mass` | — | `st_mass` | `Stellar Mass` |
| **TESS Mag** | `tess_mag` | `tess_mag` | — | `tess_mag` | — | `tess_mag` | — | — | `TESS Mag` |
| **Disposition** | `tfop_disposition` | — | `tfop_disposition` | — | — | `disposition` | — | — | `TFOPWG Disposition` |
| **Sectors** | `sectors` | `sector` | — | `sector` | — | — | — | — | `Sectors` |

---

## 4. Potential Relationships

### Primary Key → Foreign Key Relationships

```text
tic_stellar_params.tic_id (PK, 501 unique stars)
    │
    ├──→ exofop_toi_labeled.tic_id (FK, 7519 unique, one-to-many: star→TOIs)
    │       │
    │       └──→ tois_list."TIC ID" (same data, raw version)
    │
    ├──→ unified_labels.tic_id (FK, 10965 unique, one-to-many)
    │
    ├──→ feature_matrix.tic_id (FK, 40 rows, subset)
    │       │
    │       └──→ preprocessing_report.ic_id (same IDs, status log)
    │
    ├──→ synthetic_injections.tic_id (FK, synthetic IDs map to host TICs)
    │
    ├──→ nasa_confirmed_labeled.tic_id (FK, 5766 unique)
    │       │
    │       └──→ PS_raw.tic_id (same data, raw version with "TIC " prefix)
    │
    └──→ spoc_tce_sector01.tic_id (FK, 1 unique star)
```

| Relationship | Type | Confidence | Evidence |
|---|---|---|---|
| `tic_stellar_params.tic_id` → `exofop_toi_labeled.tic_id` | One-to-Many | **Likely** | Same numeric TIC IDs |
| `exofop_toi_labeled` ↔ `tois_list` | One-to-One | **Confirmed** | Same row count (8149), same TOI numbers |
| `nasa_confirmed_labeled` ↔ `PS_raw` | One-to-One | **Confirmed** | Same row count (6367), matching planet names |
| `feature_matrix.tic_id` → `preprocessing_report.ic_id` | One-to-One | **Confirmed** | Same IDs (TIC_*, SYN_*) |
| `synthetic_injections.tic_id` → `feature_matrix.tic_id` | One-to-One | **Confirmed** | SYN_* IDs present in both |
| `unified_labels.tic_id` merges `nasa_confirmed` + `exofop_toi` + others | Many-to-One | **Confirmed** | `source` column indicates origin |

---

## 5. Conceptual Data Model

```text
┌─────────────────────────┐
│   PS_2026 (Raw NASA)    │──── Cleaned ────→ ┌──────────────────────────┐
│   6,367 rows × 13 cols  │                   │  nasa_confirmed_labeled  │
└─────────────────────────┘                   │  6,367 rows × 14 cols   │
                                              └────────┬─────────────────┘
                                                       │ tic_id, label
┌─────────────────────────┐                            ↓
│   tois_list (Raw ExoFOP)│──── Cleaned ────→ ┌──────────────────────────┐
│   8,149 rows × 63 cols  │                   │  exofop_toi_labeled      │
└─────────────────────────┘                   │  8,149 rows × 18 cols   │
                                              └────────┬─────────────────┘
                                                       │ tic_id, label
┌─────────────────────────┐                            ↓
│  spoc_tce_sector01      │──── tic_id ─────→ ┌──────────────────────────┐
│  4 rows × 14 cols       │                   │     unified_labels       │
└─────────────────────────┘                   │  10,965 rows × 3 cols   │
                                              └────────┬─────────────────┘
                                                       │ tic_id
┌─────────────────────────┐                            ↓
│  tic_stellar_params     │──── tic_id ─────→ ┌──────────────────────────┐
│  501 rows × 100 cols    │                   │    feature_matrix        │
└─────────────────────────┘                   │  40 rows × 71 cols      │
                                              └────────┬─────────────────┘
┌─────────────────────────┐                            │ tic_id
│  synthetic_injections   │──── tic_id ─────→          │
│  20 rows × 10 cols      │                            │
└─────────────────────────┘                            ↓
                                              ┌──────────────────────────┐
┌─────────────────────────┐                   │  preprocessing_report    │
│  Raw light curves (.npz)│←── out_path ──────│  48 rows × 8 cols       │
└─────────────────────────┘                   └──────────────────────────┘
```

---

## 6. Data Quality Summary

### Across All Datasets

| Issue | Severity | Files Affected | Details |
|-------|----------|---------------|---------|
| **Inconsistent TIC ID formats** | High | PS_raw ("TIC 123"), feature_matrix ("TIC_123"), others (123) | Requires normalization for joins |
| **Preprocessing column typo** | Medium | preprocessing_report | `ic_id` should be `tic_id` |
| **Class imbalance** | High | unified_labels, nasa_confirmed | Mostly label=0 (planet); few false positives |
| **Small feature matrix** | High | feature_matrix | Only 40 rows — insufficient for robust ML |
| **All-null columns** | Medium | tois_list | `Planet Name` is 100% null |
| **High missingness** | Medium | tois_list (metallicity 82.5%), nasa_confirmed (depth 85.7%) | Imputation or feature exclusion needed |
| **Non-portable paths** | Low | preprocessing_report, synthetic_injections | macOS absolute paths |
| **Duplicate feature rows** | Medium | feature_matrix | 2 exact duplicates |
| **Label encoding inconsistency** | Medium | exofop_toi_labeled | Uses 0/3 instead of 0/1 |

---

## 7. ML / Analytics Feature Assessment

### Target Variable
- **`label`** in `unified_labels.csv`, `exofop_toi_labeled.csv`, `feature_matrix.csv`
- Binary classification: 0 = planet/candidate, 3 or 1 = false positive

### High-Value Input Features (from feature_matrix)
- **Transit morphology**: `depth_ppm`, `duration_hr`, `period`, `rp_rs`, `SDE`, `SNR`, `FAP`
- **Shape diagnostics**: `flat_bottom_score`, `odd_even_ratio`, `secondary_ratio`, `ingress_egress_asym`
- **Light-curve statistics**: `rms`, `skewness`, `kurtosis`, `autocorr_lag1`
- **Stellar context**: `stellar_Teff`, `stellar_logg`, `stellar_rad`, `tess_mag`
- **FFT features**: `tsf_fft_coeff_*` (frequency-domain signal characteristics)

### Identifier Columns (exclude from ML)
- `tic_id`, `toi`, `pl_name`, `hostname`, `tce_ext`, `host_path`, `npz_path`, `out_path`

### Data Leakage Risks
- **`tfop_disposition`** directly encodes the target (CP/KP→0, FP/FA→3) — **MUST exclude**.
- **`discovery_method`** and `facility` in nasa_confirmed are post-discovery metadata.
- **`Planet Name`** and `Comments` contain confirmation status.

### Features Requiring Encoding
- `sectors` (multi-value string → multi-hot or count)
- `ra`/`dec` (sexagesimal → decimal degrees)
- `tfop_disposition` (if used as feature, one-hot encode)

### Features Requiring Scaling
- All numerical features should be standardized/normalized (wide range: magnitudes ~5-18, Teff ~2800-32780, period 0.3-75000 days)

---

## 8. Preprocessing Recommendations

| Category | Recommendation | Reason |
|----------|---------------|--------|
| **TIC ID normalization** | Strip "TIC " and "TIC_" prefixes; cast to integer | Enables cross-file joins |
| **Label harmonization** | Remap label=3 → label=1 | Standard binary classification |
| **Missing values (numerical)** | Use median imputation or model-based imputation (KNN/MICE) for stellar params | Stellar parameters correlated; simple imputation may distort |
| **Missing values (categorical)** | Use "Unknown" category for `tfop_disposition` | 14 missing values |
| **RA/Dec conversion** | Convert sexagesimal strings to decimal degrees | Required for numerical use |
| **Sectors parsing** | Convert comma-separated string to count or multi-hot encoding | Multi-value categorical |
| **High-cardinality features** | Drop `Comments`, `Source`, `out_path`, `host_path` | Not predictive; text/path data |
| **Duplicate removal** | Remove duplicate rows in feature_matrix | SYN_0000/SYN_0012 and SYN_0003/SYN_0019 |
| **Constant columns** | Drop `Planet Name` (all null), `default_flag` (all 1), `tfop_disposition` in nasa_confirmed (all CP) | Zero variance |
| **Class balancing** | Use SMOTE, undersampling, or class weights | Heavy class imbalance toward planets |
| **Feature matrix expansion** | Process more TIC targets through the pipeline to increase training set beyond 40 rows | Current set too small for deep learning |
| **Outlier detection** | Check `depth_ppm` > 100,000 (eclipsing binaries), `period` = 0 (invalid) | Physical impossibilities |

---

## 9. Important Insights

> [!IMPORTANT]
> **Critical Finding 1**: The feature matrix has only **40 rows** — far too few for training a robust ML classifier. The pipeline needs to process thousands more light curves.

> [!IMPORTANT]
> **Critical Finding 2**: The label encoding is **inconsistent** — `exofop_toi_labeled` uses 0/3, `feature_matrix` uses -1, `nasa_confirmed_labeled` uses 0 only. A unified 0/1 scheme is essential.

> [!WARNING]
> **Data Leakage**: The `tfop_disposition` column directly reveals the target label. It MUST be excluded from any ML model.

> [!NOTE]
> **Pipeline Architecture**: The data flows from raw catalogs (PS_raw, tois_list) → cleaned/labeled catalogs (nasa_confirmed_labeled, exofop_toi_labeled) → unified_labels → light-curve processing (preprocessing_report) → feature extraction (feature_matrix) with synthetic augmentation (synthetic_injections).

> [!TIP]
> **Cross-Validation Strategy**: Given the multi-source nature of labels (NASA confirmed vs ExoFOP TOI), consider stratifying by `source` in unified_labels to avoid train/test leakage from the same astronomical objects appearing under different identifiers.

---

## 10. Research Sources

| Source | Used For |
|--------|----------|
| [ExoFOP TESS TOI Portal](https://exofop.ipac.caltech.edu/tess/) | TOI column meanings, TFOPWG dispositions, SG1-SG5 sub-groups |
| [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/) | Planetary Systems table column definitions |
| [TESS Input Catalog (TIC) Documentation](https://tess.mit.edu/science/tess-input-catalogue/) | TIC column definitions, flags, photometric bands |
| [Stassun et al. 2019 (TIC v8)](https://arxiv.org/abs/1905.10694) | TIC catalog column descriptions and derivation methods |
| [Kempton et al. 2018](https://arxiv.org/abs/1805.03671) | TSM and ESM metric definitions |
| [SPOC Pipeline Documentation](https://archive.stsci.edu/missions/tess/doc/) | TCE definitions, BLS/SDE/FAP metrics |
| [Kovacs et al. 2002 (BLS)](https://www.aanda.org/articles/aa/abs/2002/13/aah3462/aah3462.html) | Box-Least Squares transit search algorithm |
