"""
classifier.py — Stage 5: ML classification of transit candidates.

Stacked ensemble of gradient-boosted trees -> logistic-regression meta-learner
-> Platt calibration, trained on the feature matrix from features.py.

Design decisions (each one guards against a way these scores can look better
than they are):

* **Group-aware splits.** A synthetic injection shares its host star with real
  targets. Splitting by row would put a star in the test set and its
  synthetic clone in training. Rows are grouped by star (real: own TIC id;
  synthetic: host TIC id) and all splits keep a group on one side only.
* **Synthetics are training-only.** Held-out evaluation uses real targets.
* **Nested calibration.** The meta-learner is fit on out-of-fold base
  probabilities, and the Platt calibrator on out-of-fold *meta* probabilities.
* **Optional backends.** XGBoost / LightGBM are used when importable (they need
  the OpenMP runtime: `brew install libomp` on macOS); scikit-learn's
  HistGradientBoosting is always available and is the fallback.

CLI
---
    python src/classifier.py --features data/catalogs/feature_matrix.csv \
        --out models/ [--smote] [--learning-curve]
"""
from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             classification_report, confusion_matrix, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold

log = logging.getLogger("classifier")

BASE_DIR     = Path(__file__).resolve().parents[1]
CATALOGS_DIR = BASE_DIR / "data" / "catalogs"
SEED         = 42

# Columns that identify a row or describe how it was processed — never features.
NON_FEATURES = {"tic_id", "label", "status", "sector"}

# Columns dropped even though they are numeric: each one tracks how/when a
# target was *collected*, not a transit property, and leaks the label through
# the data-collection process rather than through real signal.
#   t0          - absolute transit epoch (BJD). Differs by ~435 days between
#                 label medians purely because real planet-side (label 0) and
#                 non-planet (label 3) targets were pulled from different
#                 catalog batches/sectors. It topped permutation importance
#                 before this fix (0.020, #1) with zero physical basis for
#                 predicting "is this a transit".
#   n_cadences  - point count after select_search_window's segment trim;
#                 tracks how many sectors got stitched per target (itself a
#                 side effect of --max-sectors and which catalog a target
#                 came from), not transit-vs-not.
LEAKY_FEATURES = {"t0", "n_cadences"}


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def _norm_id(t) -> str:
    t = str(t).strip()
    for pre in ("TIC_", "TIC "):
        if t.startswith(pre):
            t = t[len(pre):]
    return t[:-2] if t.endswith(".0") else t


def load_dataset(features_csv: Path, synthetic_csv: Optional[Path] = None):
    """
    Returns X (DataFrame), y (ndarray), groups (ndarray of star ids),
    is_synthetic (bool ndarray). Rows with failed extraction or unknown label
    (-1) are dropped.
    """
    df = pd.read_csv(features_csv, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    df = df[(df["status"].astype(str).str.strip() == "done") & (df["label"] >= 0)].copy()
    df["tic_id"] = df["tic_id"].astype(str).str.strip()

    is_syn = df["tic_id"].str.startswith("SYN").to_numpy()

    host_of = {}
    synthetic_csv = synthetic_csv or (Path(features_csv).parent / "synthetic_injections.csv")
    if Path(synthetic_csv).exists():
        s = pd.read_csv(synthetic_csv)
        host_of = {r.tic_id: _norm_id(Path(str(r.host_path)).stem) for r in s.itertuples()}
    groups = np.array([host_of.get(t, t) if syn else _norm_id(t)
                       for t, syn in zip(df["tic_id"], is_syn)])

    feature_cols = [c for c in df.columns
                    if c not in NON_FEATURES and c not in LEAKY_FEATURES
                    and pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().any()]
    X = df[feature_cols].astype(float).replace([np.inf, -np.inf], np.nan)
    return X, df["label"].astype(int).to_numpy(), groups, is_syn


def group_split(y, groups, is_syn, test_size=0.25, seed=SEED):
    """
    Hold out ~test_size of the REAL stars. Any group (star) that appears in the
    test set is removed from training entirely, synthetic clones included.
    """
    rng = np.random.default_rng(seed)
    real_groups = np.unique(groups[~is_syn])
    # stratify roughly by class using each real group's label
    test_groups = []
    for cls in np.unique(y[~is_syn]):
        g = np.unique(groups[(~is_syn) & (y == cls)])
        rng.shuffle(g)
        test_groups.extend(g[: max(1, int(round(len(g) * test_size)))])
    test_mask  = (~is_syn) & np.isin(groups, test_groups)
    train_mask = ~np.isin(groups, test_groups)
    assert not (set(groups[train_mask]) & set(groups[test_mask])), "group leakage"
    return np.where(train_mask)[0], np.where(test_mask)[0]


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

_WARNED: set = set()


def _warn_once(msg: str) -> None:
    if msg not in _WARNED:
        _WARNED.add(msg)
        log.warning(msg)


def make_base_learners(use_smote: bool = False, n_jobs: int = 2) -> dict:
    """Base learners; XGBoost/LightGBM only if their OpenMP runtime loads."""
    learners = {
        "hgb": HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=300, max_depth=6, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.15, n_iter_no_change=20,
            class_weight=None if use_smote else "balanced", random_state=SEED),
    }
    try:
        import xgboost as xgb
        learners["xgb"] = xgb.XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, eval_metric="logloss", n_jobs=n_jobs, random_state=SEED)
    except Exception as e:                                   # noqa: BLE001
        _warn_once("XGBoost unavailable — install the OpenMP runtime: `brew install libomp`.")
    try:
        import lightgbm as lgb
        learners["lgbm"] = lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.05, num_leaves=63, subsample=0.8,
            colsample_bytree=0.8, n_jobs=n_jobs, random_state=SEED, verbose=-1,
            class_weight=None if use_smote else "balanced")
    except Exception as e:                                   # noqa: BLE001
        _warn_once("LightGBM unavailable — install the OpenMP runtime: `brew install libomp`.")

    if use_smote:
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        learners = {k: Pipeline([("impute", SimpleImputer(strategy="median")),
                                 ("smote", SMOTE(random_state=SEED)),
                                 ("clf", v)]) for k, v in learners.items()}
    return learners


class StackedClassifier:
    """Base learners -> logistic meta-learner -> Platt calibration (binary)."""

    def __init__(self, use_smote: bool = False, n_splits: int = 5, n_jobs: int = 2):
        self.use_smote, self.n_splits, self.n_jobs = use_smote, n_splits, n_jobs

    def _oof(self, X, y, groups, make_model):
        """Out-of-fold class probabilities from group-stratified CV."""
        oof = np.zeros((len(y), len(self.classes_)))
        cv = StratifiedGroupKFold(self.n_splits, shuffle=True, random_state=SEED)
        for tr, va in cv.split(X, y, groups):
            m = make_model().fit(X.iloc[tr], y[tr])
            oof[va] = m.predict_proba(X.iloc[va])
        return oof

    def fit(self, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray):
        self.classes_ = np.unique(y)
        self.feature_names_ = list(X.columns)
        proto = make_base_learners(self.use_smote, self.n_jobs)
        self.names_ = list(proto)

        oof_cols = []
        for name in self.names_:
            oof = self._oof(X, y, groups, lambda n=name: clone(make_base_learners(
                self.use_smote, self.n_jobs)[n]))
            oof_cols.append(oof if len(self.classes_) > 2 else oof[:, [1]])
        meta_X = np.hstack(oof_cols)

        self.meta_ = LogisticRegression(max_iter=1000).fit(meta_X, y)
        self.base_ = {n: clone(proto[n]).fit(X, y) for n in self.names_}

        self.platt_ = None
        if len(self.classes_) == 2:               # nested: calibrate on OOF meta scores
            cv = StratifiedGroupKFold(self.n_splits, shuffle=True, random_state=SEED + 1)
            meta_oof = np.zeros(len(y))
            for tr, va in cv.split(meta_X, y, groups):
                m = LogisticRegression(max_iter=1000).fit(meta_X[tr], y[tr])
                meta_oof[va] = m.decision_function(meta_X[va])
            self.platt_ = LogisticRegression().fit(meta_oof.reshape(-1, 1), y)
        return self

    def _meta_features(self, X):
        cols = [self.base_[n].predict_proba(X[self.feature_names_]) for n in self.names_]
        return np.hstack([c if len(self.classes_) > 2 else c[:, [1]] for c in cols])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        mf = self._meta_features(X)
        if self.platt_ is not None:
            return self.platt_.predict_proba(self.meta_.decision_function(mf).reshape(-1, 1))
        return self.meta_.predict_proba(mf)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model: StackedClassifier, X, y) -> dict:
    proba = model.predict_proba(X)
    pred = model.classes_[np.argmax(proba, axis=1)]
    out = {"n": int(len(y)),
           "class_counts": {int(c): int((y == c).sum()) for c in model.classes_},
           "confusion_matrix": confusion_matrix(y, pred, labels=model.classes_).tolist(),
           "report": classification_report(y, pred, output_dict=True, zero_division=0)}
    if len(model.classes_) == 2 and len(np.unique(y)) == 2:
        # positive class = 0 ("Transit") when present, else the larger label
        pi = int(np.where(model.classes_ == 0)[0][0]) if 0 in model.classes_ else 1
        yb = (y == model.classes_[pi]).astype(int)
        p1 = proba[:, pi]
        out.update(positive_class=int(model.classes_[pi]),
                   roc_auc=float(roc_auc_score(yb, p1)),
                   pr_auc=float(average_precision_score(yb, p1)),
                   brier=float(brier_score_loss(yb, p1)))
    elif len(np.unique(y)) == len(model.classes_):
        out["roc_auc_ovr"] = float(roc_auc_score(y, proba, multi_class="ovr"))
    return out


def learning_curve(X, y, groups, is_syn, tr_idx, te_idx,
                   fractions=(0.2, 0.4, 0.6, 0.8, 1.0), **kw) -> list:
    """Held-out score vs. training-set size — tells us whether more data helps."""
    rng = np.random.default_rng(SEED)
    tr_groups = np.unique(groups[tr_idx])
    rng.shuffle(tr_groups)
    rows = []
    for f in fractions:
        keep = set(tr_groups[: max(4, int(len(tr_groups) * f))])
        idx = np.array([i for i in tr_idx if groups[i] in keep])
        if len(np.unique(y[idx])) < 2:
            continue
        m = StackedClassifier(**kw).fit(X.iloc[idx], y[idx], groups[idx])
        r = evaluate(m, X.iloc[te_idx], y[te_idx])
        rows.append({"fraction": f, "n_train": int(len(idx)),
                     "roc_auc": r.get("roc_auc"), "pr_auc": r.get("pr_auc")})
        log.info("  learning curve %3.0f%%: n=%d  AUC=%s", f * 100, len(idx), r.get("roc_auc"))
    return rows


def explain(model: StackedClassifier, X, y, top: int = 15) -> list:
    """Feature importances: SHAP on the XGBoost model if available, else permutation."""
    if "xgb" in model.base_:
        try:
            import shap
            sv = shap.TreeExplainer(model.base_["xgb"]).shap_values(X[model.feature_names_])
            imp = np.abs(np.asarray(sv)).mean(axis=0)
            order = np.argsort(imp)[::-1][:top]
            return [{"feature": model.feature_names_[i], "shap": float(imp[i])} for i in order]
        except Exception as e:                               # noqa: BLE001
            log.warning("SHAP failed (%s); using permutation importance.", e)
    hgb = model.base_["hgb"]
    r = permutation_importance(hgb, X[model.feature_names_], y, n_repeats=5,
                               random_state=SEED, n_jobs=1)
    order = np.argsort(r.importances_mean)[::-1][:top]
    return [{"feature": model.feature_names_[i], "permutation": float(r.importances_mean[i])}
            for i in order]


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def train(features_csv: Path, out_dir: Path, use_smote=False, do_learning_curve=False,
          n_jobs: int = 2) -> dict:
    X, y, groups, is_syn = load_dataset(features_csv)
    log.info("Dataset: %d rows (%d real, %d synthetic), %d features, classes %s",
             len(y), int((~is_syn).sum()), int(is_syn.sum()), X.shape[1],
             {int(c): int((y == c).sum()) for c in np.unique(y)})
    if len(np.unique(y[~is_syn])) < 2:
        raise SystemExit("Need at least two real classes to train/evaluate.")

    tr, te = group_split(y, groups, is_syn)
    log.info("Train %d rows (%d synthetic) | held-out test %d real rows",
             len(tr), int(is_syn[tr].sum()), len(te))

    model = StackedClassifier(use_smote=use_smote, n_jobs=n_jobs).fit(
        X.iloc[tr], y[tr], groups[tr])
    result = {"base_learners": model.names_, "n_features": X.shape[1],
              "train_rows": int(len(tr)), "test": evaluate(model, X.iloc[te], y[te]),
              "top_features": explain(model, X.iloc[te], y[te])}
    if do_learning_curve:
        result["learning_curve"] = learning_curve(
            X, y, groups, is_syn, tr, te, use_smote=use_smote, n_jobs=n_jobs)

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "classifier.joblib")
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    log.info("Saved %s and metrics.json", out_dir / "classifier.joblib")
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description="Stage 5 — train the transit classifier")
    ap.add_argument("--features", type=Path, default=CATALOGS_DIR / "feature_matrix.csv")
    ap.add_argument("--out", type=Path, default=BASE_DIR / "models")
    ap.add_argument("--smote", action="store_true", help="SMOTE instead of class weights")
    ap.add_argument("--learning-curve", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=2)
    a = ap.parse_args()
    res = train(a.features, a.out, a.smote, a.learning_curve, a.n_jobs)
    t = res["test"]
    print(json.dumps({k: t.get(k) for k in ("n", "class_counts", "positive_class", "roc_auc", "pr_auc", "brier")}, indent=2))
    print("confusion (rows=true, cols=pred):", t["confusion_matrix"])
