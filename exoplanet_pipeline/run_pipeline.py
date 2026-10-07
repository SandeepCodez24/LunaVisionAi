"""
run_pipeline.py — one entry point for the whole pipeline.

Batch mode chains the stage modules in order; each stage resumes from what is already on disk:

    acquire      acquisition.py    download TESS light curves + catalogs + synthetic injections (network, slow)
    preprocess   preprocessing.py  normalise, sigma-clip, detrend (wotan)
    features     features.py       TLS search + the 65 model features   -> data/catalogs/feature_matrix.csv
    train        classifier.py     stacked classifier + held-out split  -> models/
    fit          fitting.py        batman + emcee for the top candidates -> outputs/fits
    report       report.py         vetting plots, PDFs, ranked catalogue -> outputs/

    python run_pipeline.py                                  # preprocess, features, train, fit, report
    python run_pipeline.py --stages preprocess,features     # just those, in pipeline order
    python run_pipeline.py --stages acquire --sector 1 --max-targets 200
    python run_pipeline.py --top 20 --n-jobs 4              # fit + report only the 20 best candidates

Single-target mode runs one star end to end the way the web app does (fetching it from MAST if it
isn't on disk), and is the reference implementation the API's worker calls:

    python run_pipeline.py --tic 307210830                  # analyse
    python run_pipeline.py --tic 307210830 --report         # ...then fit + vetting report + PDF
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src"))

log = logging.getLogger("run_pipeline")

ALL_STAGES = ["acquire", "preprocess", "features", "train", "fit", "report"]
DEFAULT_STAGES = ["preprocess", "features", "train", "fit", "report"]       # acquire needs the network: opt-in


# ─────────────────────────────────────────────────────────────────────────────
# Stages. Each takes the parsed args and returns a short summary string.
# ─────────────────────────────────────────────────────────────────────────────
def stage_acquire(a) -> str:
    import acquisition
    if a.labeled_targets:
        acquisition.download_light_curves(acquisition.select_labeled_targets(a.labeled_targets), sector=None,
                                          concurrency=a.concurrency, max_sectors=a.max_sectors)
        return f"downloaded up to {a.labeled_targets} labeled targets"
    acquisition.run_acquisition_pipeline(sector=a.sector, max_lc_targets=a.max_targets,
                                         n_synthetic=a.n_synthetic, concurrency=a.concurrency)
    return f"sector {a.sector}, up to {a.max_targets} light curves"


def stage_preprocess(a) -> str:
    import preprocessing
    rep = preprocessing.preprocess_batch(n_jobs=a.n_jobs)
    counts = rep["status"].value_counts().to_dict() if len(rep) else {}
    return f"{len(rep)} light curves: {counts}"


def stage_features(a) -> str:
    import features
    labels = BASE_DIR / "data" / "catalogs" / "unified_labels.csv"
    df = features.extract_features_batch(label_csv=labels if labels.exists() else None,
                                         n_jobs=a.n_jobs, resume=not a.no_resume)
    return f"{int((df['status'] == 'done').sum())}/{len(df)} extracted -> data/catalogs/feature_matrix.csv"


def stage_train(a) -> str:
    import classifier
    res = classifier.train(classifier.CATALOGS_DIR / "feature_matrix.csv", BASE_DIR / "models",
                           use_smote=a.smote, n_jobs=max(a.n_jobs, 1))
    t = res["test"]
    return (f"held-out n={t['n']}: ROC-AUC {t.get('roc_auc', float('nan')):.3f}, "
            f"ECE {t.get('calibration', {}).get('ece', float('nan')):.3f} -> models/")


def stage_fit(a) -> str:
    import fitting
    cands = fitting.select_candidates(min_prob=a.min_prob, top=a.top)
    if cands.empty:
        raise RuntimeError("No candidates to fit. Run the features and train stages first, or lower --min-prob.")
    tab = fitting.fit_batch(cands, n_jobs=a.n_jobs if a.n_jobs > 0 else 1, force=a.force)
    return f"{int((tab['status'] == 'done').sum())}/{len(tab)} fitted"


def stage_report(a) -> str:
    import report
    cat = report.run(pdf=not a.no_pdf, n_jobs=a.n_jobs if a.n_jobs > 0 else 1)
    return f"{len(cat)} candidates, verdicts {cat['verdict'].value_counts().to_dict()} -> outputs/candidate_catalog.csv"


STAGE_FUNCS = {"acquire": stage_acquire, "preprocess": stage_preprocess, "features": stage_features,
               "train": stage_train, "fit": stage_fit, "report": stage_report}


def run_stages(stages: list, args) -> list:
    """Run `stages` in pipeline order, stopping at the first failure. Returns [(stage, seconds, summary)]."""
    done = []
    for name in sorted(stages, key=ALL_STAGES.index):
        log.info("━━ stage: %s", name)
        t0 = time.time()
        try:
            summary = STAGE_FUNCS[name](args)
        except Exception as e:                                           # noqa: BLE001
            log.error("Stage '%s' failed: %s", name, e)
            done.append((name, time.time() - t0, f"FAILED: {e}"))
            raise SystemExit(f"Stopped at stage '{name}': {e}")
        done.append((name, time.time() - t0, summary))
        log.info("✓ %s (%.0f s): %s", name, time.time() - t0, summary)
    return done


def run_target(tics: list, want_report: bool) -> int:
    """Analyse (and optionally fit + report) individual stars. Returns the number of failures."""
    import pipeline_service as ps
    failures = 0
    for tic in tics:
        print(f"\n=== {ps.norm_tic(tic)} ===")
        try:
            r = ps.run_target(tic, lambda stage, state, info: log.info("  %-16s %-6s %s", stage, state, info or ""))
            f = r["features"]
            print(f"  {r['pred']}  P(transit)={r['p_transit'] if r['p_transit'] is None else round(r['p_transit'], 3)}"
                  f"  period={f['period']}  SDE={f['SDE']}  depth_ppm={f['depth_ppm']}  split={r['split']}")
            if want_report:
                s = ps.run_fit_report(tic, lambda stage, state, info: log.info("  %-16s %-6s %s", stage, state, info or ""))
                print(f"  verdict: {s['verdict']}" + (f"  flags: {s['flags']}" if s["flags"] else ""))
                print(f"  report:  {ps.OUTPUTS_DIR / 'reports' / (r['tic_id'] + '.pdf')}")
        except Exception as e:                                           # noqa: BLE001
            failures += 1
            print(f"  FAILED: {e}")
    return failures


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="LunaVisionAI pipeline", formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("Batch mode")[1] if "Batch mode" in __doc__ else None)
    ap.add_argument("--stages", default=",".join(DEFAULT_STAGES),
                    help=f"comma-separated subset of {','.join(ALL_STAGES)} (default: {','.join(DEFAULT_STAGES)})")
    ap.add_argument("--tic", action="append", help="single-target mode: analyse this star (repeatable)")
    ap.add_argument("--report", action="store_true", help="with --tic: also fit the transit model and build the PDF")
    ap.add_argument("--n-jobs", type=int, default=-1, help="parallel workers (-1 = automatic)")
    ap.add_argument("--top", type=int, default=None, help="fit/report only the N best candidates")
    ap.add_argument("--min-prob", type=float, default=0.5, help="minimum model score for a candidate to be fitted")
    ap.add_argument("--force", action="store_true", help="refit even if outputs/fits/<tic>.json exists")
    ap.add_argument("--no-pdf", action="store_true", help="report stage: plots and catalogue only")
    ap.add_argument("--no-resume", action="store_true", help="features stage: re-extract everything")
    ap.add_argument("--smote", action="store_true", help="train stage: SMOTE instead of class weights")
    g = ap.add_argument_group("acquire stage")
    g.add_argument("--sector", type=int, default=1)
    g.add_argument("--max-targets", type=int, default=100)
    g.add_argument("--n-synthetic", type=int, default=200)
    g.add_argument("--concurrency", type=int, default=8)
    g.add_argument("--labeled-targets", type=int, default=0, help="download N labeled targets across all sectors")
    g.add_argument("--max-sectors", type=int, default=2)
    a = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    warnings.filterwarnings("ignore")

    if a.tic:
        return 1 if run_target(a.tic, a.report) else 0

    stages = [s.strip() for s in a.stages.split(",") if s.strip()]
    bad = [s for s in stages if s not in ALL_STAGES]
    if bad:
        ap.error(f"unknown stage(s): {', '.join(bad)}. Choose from: {', '.join(ALL_STAGES)}")
    results = run_stages(stages, a)
    print("\nPipeline finished:")
    for name, secs, summary in results:
        print(f"  {name:<11} {secs:7.0f} s  {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
