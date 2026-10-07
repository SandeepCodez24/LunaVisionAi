"""
visualization.py — the seven diagnostic plots.

The plotting code itself lives in report.py (it renders the same axes into both the PNGs and the
PDF pages, so the two can never drift apart). This module is the stable, light entry point for
everything else (API, CLI): plot names, where the PNGs live, and one call to render them.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

PLOT_NAMES = ["1_lightcurve", "2_periodogram", "3_fold", "4_corner", "5_odd_even", "6_secondary", "7_centroid"]
PLOT_TITLES = {
    "1_lightcurve": "Light curve", "2_periodogram": "TLS periodogram", "3_fold": "Phase fold + model",
    "4_corner": "MCMC corner", "5_odd_even": "Odd / even depth", "6_secondary": "Secondary eclipse",
    "7_centroid": "Centroid motion",
}
_NAME_RE = re.compile(r"^[1-7]_[a-z_]+$")


def plot_dir(out_dir: Path, tic: str) -> Path:
    return Path(out_dir) / "plots" / tic


def available_plots(out_dir: Path, tic: str) -> list:
    """Names of the PNGs that exist for this target, in display order."""
    d = plot_dir(out_dir, tic)
    return [n for n in PLOT_NAMES if (d / f"{n}.png").exists()]


def plot_path(out_dir: Path, tic: str, name: str) -> Optional[Path]:
    """Path to one plot PNG, or None. `name` is validated, so it can come straight from a URL."""
    if not _NAME_RE.match(name or "") or name not in PLOT_NAMES:
        return None
    p = plot_dir(out_dir, tic) / f"{name}.png"
    return p if p.exists() else None


def render_plots(tic: str, out_dir: Path) -> dict:
    """Render the PNGs for an already-fitted target (needs outputs/fits/<tic>.json). Returns name -> path."""
    from report import report_one                      # heavy import (matplotlib) only when rendering
    report_one(tic, Path(out_dir), pdf=False)
    return {n: plot_path(out_dir, tic, n) for n in available_plots(out_dir, tic)}
