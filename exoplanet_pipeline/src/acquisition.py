"""
acquisition.py
==============
Implements Steps 3.2 – 3.5 from the Implementation Plan:

  Step 3.2  – Download / parse reference catalogs
              a) ExoFOP-TESS TOI catalog          (already in data/raw)
              b) NASA Exoplanet Archive CSV        (already in data/raw)
              c) TESS SPOC TCE catalog via MAST
              d) TIC cross-match for stellar params

  Step 3.3  – Assign unified class labels (0=Transit, 1=EB, 2=Blend, 3=Other)

  Step 3.4  – Download raw TESS 2-min light curves (PDCSAP) via lightkurve

  Step 3.5  – Synthetic transit injection using batman to boost minority class

All processed outputs are saved to  data/processed/
All catalog outputs are saved to    data/catalogs/
"""

import os
import sys
import json
import warnings
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import astropy.io.fits as fits

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).resolve().parents[1]   # …/exoplanet_pipeline/
RAW_DIR        = BASE_DIR / "data" / "raw"
PROCESSED_DIR  = BASE_DIR / "data" / "processed"
CATALOGS_DIR   = BASE_DIR / "data" / "catalogs"

for d in [RAW_DIR, PROCESSED_DIR, CATALOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Known raw-file names (already downloaded by the user)
TOI_CSV         = RAW_DIR / "tois_list.csv"           # ExoFOP full TOI table
PS_CSV          = next(RAW_DIR.glob("PS_*.csv"), None) # NASA confirmed planets

# ─────────────────────────────────────────────────────────────────────────────
# LABEL MAP  (Step 3.3)
# ─────────────────────────────────────────────────────────────────────────────
LABEL_MAP = {
    # TESS dispositions
    "PC":  0,   # Planet Candidate  → Transit
    "KP":  0,   # Known Planet      → Transit
    "CP":  0,   # Confirmed Planet  → Transit
    "APC": 0,   # Ambiguous PC      → Transit (keep as positive)
    "EB":  1,   # Eclipsing Binary
    "BEB": 2,   # Background EB     → Blend
    "SB2": 1,   # Double-lined SB   → treat as EB
    "V":   3,   # Variable Star     → Other
    "FA":  3,   # False Alarm       → Other
    "FP":  3,   # False Positive    → Other
    "NEB": 3,   # Nearby EB         → Other
    "O":   3,   # Other
    "U":   3,   # Undecided         → Other (conservative)
}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3.2a + 3.3  – Parse ExoFOP TOI catalog
# ─────────────────────────────────────────────────────────────────────────────
def parse_exofop_toi() -> pd.DataFrame:
    """
    Load the ExoFOP-TESS TOI list (tois_list.csv), clean it, and assign
    a unified integer class label using LABEL_MAP.

    Returns
    -------
    pd.DataFrame with columns:
        tic_id, toi, ra, dec, period, depth_ppm, duration_hr,
        stellar_Teff, stellar_logg, stellar_rad, stellar_mass,
        tess_mag, tfop_disposition, label
    """
    log.info("Parsing ExoFOP TOI catalog: %s", TOI_CSV)
    df = pd.read_csv(TOI_CSV, dtype={"TIC ID": str})

    # Standardise column names to snake_case
    df.columns = [c.strip() for c in df.columns]
    rename = {
        "TIC ID":                       "tic_id",
        "TOI":                          "toi",
        "RA":                           "ra",
        "Dec":                          "dec",
        "Period (days)":               "period",
        "Depth (ppm)":                 "depth_ppm",
        "Duration (hours)":            "duration_hr",
        "Stellar Eff Temp (K)":        "stellar_Teff",
        "Stellar log(g) (cm/s^2)":     "stellar_logg",
        "Stellar Radius (R_Sun)":      "stellar_rad",
        "Stellar Mass (M_Sun)":        "stellar_mass",
        "TESS Mag":                    "tess_mag",
        "TFOPWG Disposition":          "tfop_disposition",
        "Epoch (BJD)":                 "epoch_bjd",
        "Planet Radius (R_Earth)":     "planet_rad_earth",
        "Planet SNR":                  "planet_snr",
        "Sectors":                     "sectors",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # Assign labels
    df["tfop_disposition"] = df["tfop_disposition"].fillna("U").str.strip().str.upper()
    df["label"] = df["tfop_disposition"].map(LABEL_MAP).fillna(3).astype(int)

    keep_cols = [c for c in [
        "tic_id", "toi", "ra", "dec", "period", "depth_ppm", "duration_hr",
        "epoch_bjd", "stellar_Teff", "stellar_logg", "stellar_rad",
        "stellar_mass", "tess_mag", "planet_rad_earth", "planet_snr",
        "tfop_disposition", "label", "sectors",
    ] if c in df.columns]

    df = df[keep_cols].drop_duplicates(subset=["tic_id", "toi"])
    log.info("  → %d TOI entries | label distribution: %s",
             len(df), df["label"].value_counts().to_dict())

    out = CATALOGS_DIR / "exofop_toi_labeled.csv"
    df.to_csv(out, index=False)
    log.info("  Saved → %s", out)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3.2b + 3.3  – Parse NASA Exoplanet Archive confirmed planets
# ─────────────────────────────────────────────────────────────────────────────
def parse_nasa_confirmed() -> pd.DataFrame:
    """
    Load the NASA Exoplanet Archive confirmed-planets CSV (PS_*.csv),
    filter to TESS-discovered planets, and label all as class 0 (Transit).

    Returns
    -------
    pd.DataFrame with columns: pl_name, hostname, tic_id, period, depth_ppm,
        duration_hr, stellar_Teff, stellar_logg, stellar_rad, label
    """
    if PS_CSV is None:
        log.warning("NASA PS CSV not found in %s – skipping.", RAW_DIR)
        return pd.DataFrame()

    log.info("Parsing NASA Exoplanet Archive: %s", PS_CSV.name)
    df = pd.read_csv(PS_CSV, comment="#", dtype=str)
    df.columns = [c.strip() for c in df.columns]

    rename = {
        "pl_name":       "pl_name",
        "hostname":      "hostname",
        "tic_id":        "tic_id",
        "pl_orbper":     "period",
        "pl_trandep":    "depth_ppm",      # transit depth in ppm
        "pl_trandur":    "duration_hr",    # hours
        "st_teff":       "stellar_Teff",
        "st_logg":       "stellar_logg",
        "st_rad":        "stellar_rad",
        "st_mass":       "stellar_mass",
        "discoverymethod": "discovery_method",
        "disc_facility": "facility",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # Keep only default parameter rows if column exists
    if "default_flag" in df.columns:
        df = df[df["default_flag"].astype(str) == "1"]

    # All confirmed planets are class 0
    df["label"] = 0
    df["tfop_disposition"] = "CP"

    keep_cols = [c for c in [
        "pl_name", "hostname", "tic_id", "period", "depth_ppm", "duration_hr",
        "stellar_Teff", "stellar_logg", "stellar_rad", "stellar_mass",
        "label", "tfop_disposition", "discovery_method", "facility",
    ] if c in df.columns]

    df = df[keep_cols].drop_duplicates()
    log.info("  → %d confirmed planets (all label=0)", len(df))

    out = CATALOGS_DIR / "nasa_confirmed_labeled.csv"
    df.to_csv(out, index=False)
    log.info("  Saved → %s", out)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3.2c  – Download TESS SPOC TCE catalog from MAST
# ─────────────────────────────────────────────────────────────────────────────
def download_tce_catalog(sector: int = 1, max_records: int = 500) -> pd.DataFrame:
    """
    Query the MAST TESS DV (Data Validation) catalog for Threshold Crossing
    Events (TCEs) in a given sector.  Results are saved to data/catalogs/.

    This uses astroquery's Mast service, which exposes the DV results table
    (equivalent to the SPOC dvt.fits catalog).

    Parameters
    ----------
    sector      : TESS sector number (default 1)
    max_records : maximum rows to retrieve (default 500 for speed)

    Returns
    -------
    pd.DataFrame of TCE entries with columns:
        tic_id, toi_id, sector, period, epoch, depth_ppm, duration_hr, label
    """
    log.info("Querying MAST for SPOC TCE catalog — Sector %d …", sector)
    try:
        from astroquery.mast import Catalogs, Observations
        from astroquery.mast import MastMissions

        # Query TESS SPOC DV summary table for this sector
        # The table name on MAST is 'tess_dv_summary'
        results = MastMissions.query_criteria(
            mission="tess",
            select_cols=["tic_id", "tce_plnt_num", "tce_period", "tce_time0bk",
                         "tce_depth", "tce_duration", "tce_dikco_msky",
                         "tce_model_snr", "tce_sde"],
            sector_number=sector,
            limit=max_records,
        )
        tce_df = results.to_pandas() if hasattr(results, "to_pandas") else pd.DataFrame(results)

    except Exception as e:
        log.warning("MastMissions query failed (%s). Falling back to Observations API.", e)
        tce_df = _download_tce_via_observations(sector, max_records)

    if tce_df is None or len(tce_df) == 0:
        log.warning("No TCE records retrieved for sector %d.", sector)
        return pd.DataFrame()

    # Standardise
    col_map = {
        "tic_id":        "tic_id",
        "tce_plnt_num":  "tce_num",
        "tce_period":    "period",
        "tce_time0bk":   "epoch_btjd",
        "tce_depth":     "depth_ppm",
        "tce_duration":  "duration_hr",
        "tce_model_snr": "snr",
        "tce_sde":       "sde",
    }
    tce_df = tce_df.rename(columns={k: v for k, v in col_map.items() if k in tce_df.columns})
    tce_df["sector"]  = sector
    tce_df["label"]   = -1   # Unknown — will be resolved via TOI cross-match
    tce_df["source"]  = "SPOC_TCE"

    out = CATALOGS_DIR / f"spoc_tce_sector{sector:02d}.csv"
    tce_df.to_csv(out, index=False)
    log.info("  → %d TCE rows saved → %s", len(tce_df), out)
    return tce_df


def _download_tce_via_observations(sector: int, max_records: int) -> pd.DataFrame:
    """
    Fallback: search MAST Observations for TESS SPOC DV products in a sector,
    download dvt.fits files, and parse TCE parameters from named TCE_N extensions.
    """
    log.info("  Fallback: searching MAST Observations for SPOC DV in sector %d …", sector)
    try:
        from astroquery.mast import Observations

        obs = Observations.query_criteria(
            obs_collection="TESS",
            sequence_number=sector,
            dataproduct_type="timeseries",
            calib_level=3,
        )
        if len(obs) == 0:
            return pd.DataFrame()

        products  = Observations.get_product_list(obs[:20])   # limit for speed
        dv_prods  = Observations.filter_products(products, extension="dvt.fits", productType="SCIENCE")

        if len(dv_prods) == 0:
            return pd.DataFrame()

        # Download all at once
        manifest = Observations.download_products(
            dv_prods[:max_records],
            download_dir=str(RAW_DIR / f"sector_{sector:02d}_dvt"),
        )
        # Parse each downloaded file
        return _parse_dvt_files(list(manifest["Local Path"]))

    except Exception as e:
        log.error("  Fallback also failed: %s", e)
        return pd.DataFrame()


def _parse_dvt_files(file_paths: list) -> pd.DataFrame:
    """
    Parse a list of TESS dvt.fits files.  TCE parameters live in named
    extensions  TCE_1, TCE_2, … (header keywords: TPERIOD, TEPOCH, TDEPTH,
    TDUR, TSNR, IMPACT, RADRATIO).  Stellar params are in the PRIMARY header.

    Returns
    -------
    pd.DataFrame with one row per TCE extension per file.
    """
    import glob as _glob
    records = []
    for fpath in file_paths:
        try:
            with fits.open(str(fpath)) as hdul:
                hdr0   = hdul[0].header
                tic_id = str(hdr0.get("TICID", ""))
                teff   = hdr0.get("TEFF",    np.nan)
                tmag   = hdr0.get("TESSMAG", np.nan)
                for ext in hdul[1:]:
                    if not ext.name.startswith("TCE_"):
                        continue
                    h = ext.header
                    records.append({
                        "tic_id":        tic_id,
                        "tce_ext":       ext.name,
                        "period":        h.get("TPERIOD",  np.nan),
                        "epoch_btjd":    h.get("TEPOCH",   np.nan),
                        "depth_ppm":     h.get("TDEPTH",   np.nan),
                        "duration_hr":   h.get("TDUR",     np.nan),
                        "snr":           h.get("TSNR",     np.nan),
                        "impact":        h.get("IMPACT",   np.nan),
                        "rad_ratio":     h.get("RADRATIO", np.nan),
                        "stellar_Teff":  teff,
                        "tess_mag":      tmag,
                        "label":         -1,
                        "source":        "SPOC_TCE",
                    })
        except Exception as ex:
            log.debug("  Parse error %s: %s", fpath, ex)
    return pd.DataFrame(records)


def parse_downloaded_dvt_fits(sector: int = 1) -> pd.DataFrame:
    """
    Parse all already-downloaded dvt.fits files for a given sector.
    Saves result to data/catalogs/spoc_tce_sectorXX.csv.
    """
    import glob as _glob
    pattern  = str(RAW_DIR / f"sector_{sector:02d}_dvt" / "**" / "*dvt*.fits")
    dvt_files = _glob.glob(pattern, recursive=True)
    log.info("Parsing %d local dvt.fits files for sector %d…", len(dvt_files), sector)
    df = _parse_dvt_files(dvt_files)
    if not df.empty:
        out = CATALOGS_DIR / f"spoc_tce_sector{sector:02d}.csv"
        df.to_csv(out, index=False)
        log.info("  → %d TCE rows saved → %s", len(df), out)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3.2d  – TIC Catalog cross-match for stellar parameters
# ─────────────────────────────────────────────────────────────────────────────
def crossmatch_tic(tic_ids: list, batch_size: int = 100) -> pd.DataFrame:
    """
    Query the TESS Input Catalog (TIC v8.2) on MAST for a list of TIC IDs
    to retrieve stellar parameters: Teff, logg, R★, M★, distance, contamination.

    Parameters
    ----------
    tic_ids   : list of TIC IDs (strings or ints)
    batch_size: number of IDs per MAST request (default 100)

    Returns
    -------
    pd.DataFrame indexed by tic_id with stellar columns
    """
    from astroquery.mast import Catalogs

    log.info("TIC cross-match for %d targets (batch_size=%d)…", len(tic_ids), batch_size)
    all_rows = []
    tic_ids  = [str(t) for t in tic_ids if pd.notna(t)]

    for start in range(0, len(tic_ids), batch_size):
        batch = tic_ids[start : start + batch_size]
        try:
            # Build a comma-separated ID list query string
            id_list = ",".join(batch)
            result = Catalogs.query_criteria(
                catalog="TIC",
                ID=id_list,
            )
            if result is not None and len(result) > 0:
                all_rows.append(result.to_pandas())
        except Exception as e:
            # Fallback: query one by one (slower but reliable)
            for tic in batch:
                try:
                    r = Catalogs.query_object(f"TIC {tic}", catalog="TIC", radius=0.0001)
                    if r is not None and len(r) > 0:
                        all_rows.append(r[:1].to_pandas())
                except Exception as e2:
                    log.debug("  TIC %s lookup failed: %s", tic, e2)

        if (start // batch_size + 1) % 5 == 0:
            log.info("  … processed %d / %d TIC IDs", start + batch_size, len(tic_ids))

    if not all_rows:
        log.warning("TIC cross-match returned no results.")
        return pd.DataFrame()

    tic_df = pd.concat(all_rows, ignore_index=True)
    tic_df = tic_df.rename(columns={
        "ID":         "tic_id",
        "Teff":       "stellar_Teff",
        "logg":       "stellar_logg",
        "rad":        "stellar_rad",
        "mass":       "stellar_mass",
        "d":          "stellar_dist_pc",
        "contratio":  "tic_contratio",
        "lumclass":   "lum_class",
        "gaiamag":    "gaia_mag",
        "Tmag":       "tess_mag",
    })
    tic_df["tic_id"] = tic_df["tic_id"].astype(str)
    tic_df = tic_df.drop_duplicates(subset=["tic_id"])

    out = CATALOGS_DIR / "tic_stellar_params.csv"
    tic_df.to_csv(out, index=False)
    log.info("  → %d TIC rows saved → %s", len(tic_df), out)
    return tic_df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3.4  – Download raw TESS 2-min light curves via lightkurve
# ─────────────────────────────────────────────────────────────────────────────

# QUALITY bitmask bits to reject (per the PRD: flags 1,2,4,8,16,512)
BAD_QUALITY_BITS = 1 | 2 | 4 | 8 | 16 | 512


def download_light_curves(
    tic_ids: list,
    sector: int = None,
    output_dir: Path = None,
    max_targets: int = None,
) -> dict:
    """
    Download SPOC 2-minute cadence TESS light curves for a list of TIC IDs.

    For each target:
      - Downloads PDCSAP_FLUX and SAP_FLUX from MAST via lightkurve
      - Applies QUALITY bitmask filter (removes bad cadences)
      - Saves a compressed numpy archive (.npz) per target to output_dir

    Parameters
    ----------
    tic_ids    : list of TIC ID strings
    sector     : restrict to a specific TESS sector (None = all available)
    output_dir : where to save .npz files (default: data/processed/lc_raw/)
    max_targets: cap on number of targets (for testing)

    Returns
    -------
    dict mapping tic_id → npz file path (for successfully saved targets)
    """
    import lightkurve as lk

    if output_dir is None:
        output_dir = PROCESSED_DIR / "lc_raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    tic_ids = [str(t) for t in tic_ids if pd.notna(t)]
    if max_targets:
        tic_ids = tic_ids[:max_targets]

    log.info("Downloading light curves for %d targets (sector=%s)…", len(tic_ids), sector)

    manifest = {}
    stats = {"downloaded": 0, "skipped": 0, "failed": 0}

    for i, tic_id in enumerate(tic_ids):
        npz_path = output_dir / f"TIC_{tic_id}.npz"

        # ── Resume: skip if already downloaded ──────────────────────────────
        if npz_path.exists():
            manifest[tic_id] = str(npz_path)
            stats["skipped"] += 1
            continue

        try:
            query_str = f"TIC {tic_id}"
            search_kwargs = dict(author="SPOC", cadence="2min")
            if sector is not None:
                search_kwargs["sector"] = sector

            sr = lk.search_lightcurve(query_str, **search_kwargs)
            if len(sr) == 0:
                log.debug("  TIC %s: no SPOC 2-min data found.", tic_id)
                stats["failed"] += 1
                continue

            lc_coll = sr.download_all(quality_bitmask="none")   # we apply our own mask
            if lc_coll is None or len(lc_coll) == 0:
                stats["failed"] += 1
                continue

            # Stitch multiple sectors if available
            lc = lc_coll.stitch()

            # Apply custom QUALITY bitmask
            good_mask = (lc["quality"].value & BAD_QUALITY_BITS) == 0
            lc = lc[good_mask]

            if len(lc) < 100:
                log.debug("  TIC %s: too few good cadences (%d).", tic_id, len(lc))
                stats["failed"] += 1
                continue

            time        = lc.time.value.astype(np.float64)
            sap_flux    = lc["sap_flux"].value.astype(np.float64)
            pdcsap_flux = lc["pdcsap_flux"].value.astype(np.float64)
            quality     = lc["quality"].value.astype(np.int32)

            np.savez_compressed(
                npz_path,
                tic_id      = tic_id,
                time        = time,
                sap_flux    = sap_flux,
                pdcsap_flux = pdcsap_flux,
                quality     = quality,
                sector      = sector if sector else -1,
            )
            manifest[tic_id] = str(npz_path)
            stats["downloaded"] += 1

        except Exception as e:
            log.warning("  TIC %s: download error — %s", tic_id, e)
            stats["failed"] += 1

        if (i + 1) % 50 == 0:
            log.info("  Progress: %d / %d  |  %s", i + 1, len(tic_ids), stats)

    log.info("Download complete: %s", stats)

    # Save manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    log.info("Manifest saved → %s", manifest_path)
    return manifest


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3.5  – Synthetic transit injection for minority class augmentation
# ─────────────────────────────────────────────────────────────────────────────

def inject_synthetic_transits(
    npz_paths: list,
    n_inject: int = 200,
    output_dir: Path = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Inject batman-generated synthetic transit signals into real, non-variable
    TESS light curves to augment the minority (Transit) class.

    For each injected target:
      - Randomly samples transit parameters from physically motivated ranges
      - Injects the batman model into a real (non-variable) host light curve
      - Saves the injected flux as a new .npz file
      - Returns a catalog row with the injected parameters and label=0

    Parameters
    ----------
    npz_paths  : list of Path objects to real (class=Other) .npz light curves
    n_inject   : number of synthetic transits to generate
    output_dir : where to save injected .npz files
    seed       : random seed for reproducibility

    Returns
    -------
    pd.DataFrame with one row per injection:
        tic_id, period, rp, a, inc, depth_ppm, label, npz_path
    """
    try:
        import batman
    except ImportError:
        log.error("batman-package not installed. Run: pip install batman-package")
        return pd.DataFrame()

    rng = np.random.default_rng(seed)

    if output_dir is None:
        output_dir = PROCESSED_DIR / "lc_injected"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not npz_paths:
        log.warning("No host light curves provided for injection.")
        return pd.DataFrame()

    log.info("Injecting %d synthetic transits into %d host light curves…",
             n_inject, len(npz_paths))

    records = []
    for i in range(n_inject):
        # ── Pick a random host light curve ──────────────────────────────────
        host_path = rng.choice(npz_paths)
        data      = np.load(host_path, allow_pickle=True)
        time      = data["time"].astype(np.float64)
        flux      = data["pdcsap_flux"].astype(np.float64)

        # Remove NaNs
        valid = np.isfinite(time) & np.isfinite(flux)
        time  = time[valid]
        flux  = flux[valid]
        if len(time) < 200:
            continue

        # Normalise host flux
        flux = flux / np.nanmedian(flux)

        # ── Sample transit parameters ────────────────────────────────────────
        period = float(rng.uniform(1.0, 30.0))           # days
        rp     = float(rng.uniform(0.01, 0.15))          # Rp/R*
        a      = float(rng.uniform(5.0, 50.0))           # a/R*
        inc    = float(rng.uniform(85.0, 90.0))          # degrees
        t0     = float(rng.uniform(time[0], time[0] + period))

        # ── Build batman model ───────────────────────────────────────────────
        params = batman.TransitParams()
        params.t0          = t0
        params.per         = period
        params.rp          = rp
        params.a           = a
        params.inc         = inc
        params.ecc         = 0.0
        params.w           = 90.0
        params.u           = [0.3, 0.1]
        params.limb_dark   = "quadratic"

        try:
            m          = batman.TransitModel(params, time)
            lc_model   = m.light_curve(params)
            flux_inj   = flux * lc_model
        except Exception as e:
            log.debug("  Injection %d failed (batman error): %s", i, e)
            continue

        depth_ppm = (rp ** 2) * 1e6

        # ── Save injected light curve ────────────────────────────────────────
        syn_id    = f"SYN_{i:04d}"
        out_path  = output_dir / f"{syn_id}.npz"
        np.savez_compressed(
            out_path,
            tic_id      = syn_id,
            time        = time,
            pdcsap_flux = flux_inj,
            sap_flux    = flux_inj,        # same — host is already clean
            quality     = np.zeros(len(time), dtype=np.int32),
            is_injected = True,
            inj_period  = period,
            inj_rp      = rp,
            inj_a       = a,
            inj_inc     = inc,
            inj_t0      = t0,
            inj_depth_ppm = depth_ppm,
        )

        records.append({
            "tic_id":     syn_id,
            "host_path":  str(host_path),
            "period":     period,
            "rp":         rp,
            "a":          a,
            "inc":        inc,
            "t0":         t0,
            "depth_ppm":  depth_ppm,
            "label":      0,
            "npz_path":   str(out_path),
        })

        if (i + 1) % 50 == 0:
            log.info("  Injected %d / %d", i + 1, n_inject)

    df = pd.DataFrame(records)
    if not df.empty:
        out_csv = CATALOGS_DIR / "synthetic_injections.csv"
        df.to_csv(out_csv, index=False)
        log.info("Injection catalog saved → %s  (%d entries)", out_csv, len(df))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MASTER PIPELINE  – run all steps in order
# ─────────────────────────────────────────────────────────────────────────────
def run_acquisition_pipeline(
    sector: int           = 1,
    max_lc_targets: int   = 100,     # set higher for full-sector runs
    n_synthetic: int      = 200,
    tce_max_records: int  = 500,
):
    """
    Execute Steps 3.2 → 3.5 sequentially and persist all outputs.

    Parameters
    ----------
    sector          : TESS sector to process
    max_lc_targets  : max number of light curves to download (Step 3.4)
    n_synthetic     : number of synthetic injections to create (Step 3.5)
    tce_max_records : max TCE rows to retrieve from MAST (Step 3.2c)
    """
    log.info("=" * 70)
    log.info("ACQUISITION PIPELINE  —  Sector %d", sector)
    log.info("=" * 70)

    # ── Step 3.2a + 3.3 ─────────────────────────────────────────────────────
    toi_df = parse_exofop_toi()

    # ── Step 3.2b + 3.3 ─────────────────────────────────────────────────────
    nasa_df = parse_nasa_confirmed()

    # ── Step 3.2c ────────────────────────────────────────────────────────────
    tce_df = download_tce_catalog(sector=sector, max_records=tce_max_records)

    # ── Step 3.2d – TIC cross-match ─────────────────────────────────────────
    all_tic_ids = list(toi_df["tic_id"].dropna().unique())
    tic_df = crossmatch_tic(all_tic_ids[:500], batch_size=100)  # cap for speed

    # ── Step 3.4 – Download light curves ────────────────────────────────────
    # Use the first max_lc_targets TOI TIC IDs for this sector
    target_ids = list(toi_df["tic_id"].dropna().unique()[:max_lc_targets])
    lc_manifest = download_light_curves(
        tic_ids    = target_ids,
        sector     = sector,
        output_dir = PROCESSED_DIR / "lc_raw",
    )

    # ── Step 3.5 – Synthetic injection ──────────────────────────────────────
    # Use downloaded class=3 (Other / no-transit) light curves as hosts
    npz_files = list((PROCESSED_DIR / "lc_raw").glob("TIC_*.npz"))
    syn_df = inject_synthetic_transits(
        npz_paths  = npz_files,
        n_inject   = n_synthetic,
        output_dir = PROCESSED_DIR / "lc_injected",
    )

    # ── Summary ──────────────────────────────────────────────────────────────
    log.info("=" * 70)
    log.info("ACQUISITION COMPLETE")
    log.info("  ExoFOP TOI entries    : %d", len(toi_df))
    log.info("  NASA confirmed planets: %d", len(nasa_df))
    log.info("  TCE entries           : %d", len(tce_df))
    log.info("  TIC cross-matches     : %d", len(tic_df))
    log.info("  Light curves saved    : %d", len(lc_manifest))
    log.info("  Synthetic injections  : %d", len(syn_df))
    log.info("=" * 70)

    return {
        "toi_df":   toi_df,
        "nasa_df":  nasa_df,
        "tce_df":   tce_df,
        "tic_df":   tic_df,
        "lc_manifest": lc_manifest,
        "syn_df":   syn_df,
    }


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Exoplanet Pipeline — Data Acquisition")
    parser.add_argument("--sector",       type=int, default=1,   help="TESS sector number")
    parser.add_argument("--max-targets",  type=int, default=100, help="Max light curves to download")
    parser.add_argument("--n-synthetic",  type=int, default=200, help="Synthetic injections to create")
    parser.add_argument("--tce-records",  type=int, default=500, help="Max TCE rows from MAST")
    args = parser.parse_args()

    run_acquisition_pipeline(
        sector          = args.sector,
        max_lc_targets  = args.max_targets,
        n_synthetic     = args.n_synthetic,
        tce_max_records = args.tce_records,
    )
