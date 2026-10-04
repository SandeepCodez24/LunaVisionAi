"""
User-supplied light curves -> the raw .npz format preprocessing.py consumes.

Accepts:
  * CSV/TXT with a time column and a flux column (header names are matched loosely;
    a headerless two-column file is read as time, flux)
  * TESS-style light-curve FITS (TIME + PDCSAP_FLUX/SAP_FLUX/FLUX, optional QUALITY, MOM_CENTR1/2)

Everything is validated up front so a bad file fails immediately with a readable
message instead of deep inside the pipeline.
"""
from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

MAX_BYTES = 40 * 1024 * 1024
MIN_CADENCES = 200
MAX_CADENCES = 3_000_000
MIN_BASELINE_D = 2.0
BAD_QUALITY_BITS = 1 | 2 | 4 | 8 | 16 | 512          # same mask acquisition.py applies

_TIME_NAMES = ("time", "btjd", "bjd", "bjd_tdb", "jd", "hjd", "t", "mjd", "bkjd")
_FLUX_NAMES = ("pdcsap_flux", "flux", "sap_flux", "norm_flux", "normalized_flux", "flux_norm",
               "fluxes", "f", "lc", "relative_flux")


class IngestError(ValueError):
    """A user-facing problem with the uploaded file."""


def _pick(cols: dict, names) -> Optional[str]:
    for n in names:
        if n in cols:
            return cols[n]
    return None


def _parse_csv(data: bytes) -> dict:
    text = data.decode("utf-8", errors="replace")
    try:
        df = pd.read_csv(io.StringIO(text), sep=None, engine="python", comment="#")
    except Exception as e:                                           # noqa: BLE001
        raise IngestError(f"Could not read the file as CSV: {e}")
    # A headerless file makes pandas swallow the first data row as column names:
    # if every "column name" is a number, re-read with header=None so no point is lost.
    def _is_number(x):
        try:
            float(x)
            return True
        except (TypeError, ValueError):
            return False
    if len(df.columns) >= 2 and all(_is_number(c) for c in df.columns):
        try:
            df = pd.read_csv(io.StringIO(text), sep=None, engine="python", comment="#", header=None)
        except Exception as e:                                       # noqa: BLE001
            raise IngestError(f"Could not read the file as CSV: {e}")
    cols = {str(c).strip().lower(): c for c in df.columns}
    tcol, fcol = _pick(cols, _TIME_NAMES), _pick(cols, _FLUX_NAMES)
    if (tcol is None or fcol is None) and len(df.columns) >= 2 and all(isinstance(c, int) for c in df.columns):
        tcol, fcol = df.columns[0], df.columns[1]                    # headerless: time, flux
    if tcol is None or fcol is None:
        raise IngestError("Couldn't find a time and a flux column. Name them 'time' and 'flux' "
                          f"(found columns: {', '.join(map(str, df.columns[:6]))}).")
    out = {"time": pd.to_numeric(df[tcol], errors="coerce").to_numpy(float),
           "flux": pd.to_numeric(df[fcol], errors="coerce").to_numpy(float)}
    qcol = _pick(cols, ("quality", "flags", "quality_flag"))
    if qcol is not None:
        out["quality"] = pd.to_numeric(df[qcol], errors="coerce").fillna(0).to_numpy(np.int64)
    for key, names in (("c1", ("mom_centr1", "centroid_col", "centr1")), ("c2", ("mom_centr2", "centroid_row", "centr2"))):
        c = _pick(cols, names)
        if c is not None:
            out[key] = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
    return out


def _parse_fits(data: bytes) -> dict:
    import astropy.io.fits as fits
    try:
        hdul = fits.open(io.BytesIO(data), memmap=False)
    except Exception as e:                                           # noqa: BLE001
        raise IngestError(f"Could not read the file as FITS: {e}")
    with hdul:
        table = next((h for h in hdul if getattr(h, "columns", None) is not None
                      and "TIME" in [n.upper() for n in h.columns.names]), None)
        if table is None:
            raise IngestError("This FITS file has no table with a TIME column.")
        names = {n.upper(): n for n in table.columns.names}
        fcol = next((names[n] for n in ("PDCSAP_FLUX", "SAP_FLUX", "FLUX") if n in names), None)
        if fcol is None:
            raise IngestError("This FITS file has no PDCSAP_FLUX, SAP_FLUX or FLUX column.")
        d = table.data
        out = {"time": np.array(d[names["TIME"]], dtype=float), "flux": np.array(d[fcol], dtype=float)}
        if "QUALITY" in names:
            out["quality"] = np.array(d[names["QUALITY"]], dtype=np.int64)
        for key, nm in (("c1", "MOM_CENTR1"), ("c2", "MOM_CENTR2")):
            if nm in names:
                out[key] = np.array(d[names[nm]], dtype=float)
        sector = hdul[0].header.get("SECTOR")
        if sector is not None:
            out["sector"] = int(sector)
    return out


def parse_light_curve(filename: str, data: bytes) -> dict:
    """Parse + validate. Returns clean arrays (sorted by time, finite, positive flux)."""
    if len(data) > MAX_BYTES:
        raise IngestError(f"File is {len(data) / 1e6:.0f} MB; the limit is {MAX_BYTES // (1024 * 1024)} MB.")
    if not data:
        raise IngestError("The file is empty.")
    name = (filename or "").lower()
    is_fits = name.endswith((".fits", ".fit", ".fits.gz", ".fz")) or data[:6] == b"SIMPLE"
    raw = _parse_fits(data) if is_fits else _parse_csv(data)

    t, f = raw["time"], raw["flux"]
    n = len(t)
    if n > MAX_CADENCES:
        raise IngestError(f"{n:,} rows is too many (limit {MAX_CADENCES:,}).")
    keep = np.isfinite(t) & np.isfinite(f) & (f > 0)
    if "quality" in raw:
        keep &= (raw["quality"] & BAD_QUALITY_BITS) == 0
    if keep.sum() < MIN_CADENCES:
        raise IngestError(f"Only {int(keep.sum())} usable points (finite, positive flux, good quality); "
                          f"need at least {MIN_CADENCES}.")
    t, f = t[keep], f[keep]
    c1 = raw["c1"][keep] if "c1" in raw else np.full(len(t), np.nan)
    c2 = raw["c2"][keep] if "c2" in raw else np.full(len(t), np.nan)
    order = np.argsort(t, kind="stable")
    t, f, c1, c2 = t[order], f[order], c1[order], c2[order]
    uniq = np.concatenate(([True], np.diff(t) > 0))                  # drop duplicate timestamps
    t, f, c1, c2 = t[uniq], f[uniq], c1[uniq], c2[uniq]
    if t[0] > 2.4e6:                                                 # full Julian date -> BTJD
        t = t - 2457000.0
    baseline = float(t[-1] - t[0])
    if baseline < MIN_BASELINE_D:
        raise IngestError(f"The light curve spans only {baseline:.2f} days; need at least {MIN_BASELINE_D:g}. "
                          "Is the time column in days?")
    if np.nanmedian(np.abs(np.diff(f))) == 0:
        raise IngestError("The flux column looks constant.")
    return {"time": t, "flux": f, "c1": c1, "c2": c2, "sector": int(raw.get("sector", -1))}


def save_upload(filename: str, data: bytes, raw_dir: Path) -> str:
    """Parse, validate and store as raw .npz. Returns the new id (UPL_xxxxxxxx)."""
    lc = parse_light_curve(filename, data)
    uid = "UPL_" + hashlib.sha1(data).hexdigest()[:8]                # same file -> same id (idempotent)
    raw_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        raw_dir / f"{uid}.npz",
        tic_id=uid, time=lc["time"], sap_flux=lc["flux"], pdcsap_flux=lc["flux"],
        quality=np.zeros(len(lc["time"]), dtype=np.int32),
        mom_centr1=lc["c1"], mom_centr2=lc["c2"],
        pos_corr1=np.full(len(lc["time"]), np.nan), pos_corr2=np.full(len(lc["time"]), np.nan),
        sector=lc["sector"],
    )
    return uid
