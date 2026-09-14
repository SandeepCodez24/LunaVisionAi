# LunaVisionAI — Industry-Grade SaaS Product & Architecture Plan

> **Status**: Proposal / North-star architecture
> **Supersedes in scope (not in content)**: `Implementation_Plan.md` — that document remains the authoritative spec for the *scientific pipeline* (Stages 1–6). This document wraps that pipeline into a real commercial product and defines everything the hackathon scope never had to consider: multi-tenancy, billing, reliability, security, and scale.
> **Author context**: Written after auditing the current `exoplanet_pipeline/` codebase against `College_Project.pdf` (PRD v1.0) and the existing `Implementation_Plan.md`.
> **No code was changed to produce this document.**

---

## 0. How to read this document

`Implementation_Plan.md` answers: *"How do we detect and classify exoplanet transits correctly?"* — it is still correct and should stay the spec for the science.

This document answers a different question: *"How do we turn that pipeline into LunaVisionAI, a commercial SaaS product that research groups, universities, and citizen-science platforms can sign up for, trust, and pay for?"* That requires productizing six things the original PRD never scoped: tenancy, identity, billing, durability, observability, and a real web product — on top of a science core that is currently ~35% built.

---

## 1. Executive Summary

LunaVisionAI today is a **single-user research CLI in progress**: strong data-acquisition and preprocessing code, a partially-built feature pipeline with a genuinely sophisticated GPU/CPU feature-extraction layer, and everything downstream of that (detection formalization, classification, parameter fitting, visualization, reporting, orchestration, and the frontend) still unwritten. It was scoped and written as a hackathon deliverable (see `College_Project.pdf`, "Hackathon Technical Specification," June 2026).

To become an **industry-standard, first-class application**, LunaVisionAI needs two parallel tracks of work that this plan sequences deliberately so neither blocks the other:

1. **Finish the science core** — the remaining 65% of the pipeline (Stages 3–6) exactly as scoped in `Implementation_Plan.md`, so the product has something correct to sell.
2. **Build the product shell around it** — multi-tenant API, auth, billing, a real frontend, workflow orchestration, MLOps, and cloud infrastructure — so the science core is deliverable as a reliable, secure, paid service rather than a script someone runs on their laptop.

The recommended path is **not** a rewrite. The acquisition/preprocessing/features code is well-engineered (RAM guards, timeout wrappers, GPU↔CPU fallbacks, parallel batching) and becomes the execution engine inside a **Detection Worker Service**. The empty `src/agents/*.py` stub files are replaced outright — a real multi-tenant SaaS needs durable workflow orchestration (Temporal), not the `multiprocessing.Queue` message-passing sketch in `Implementation_Plan.md` §5, which has no retry semantics, no persistence, and no per-tenant isolation.

---

## 2. Current State Assessment (codebase audit)

| Module | Path | Lines | Status | Notes |
|---|---|---:|---|---|
| Acquisition | `src/acquisition.py` | 789 | ✅ Substantially built | MAST querying, TOI/ExoFOP/TCE parsing, TIC crossmatch, synthetic transit injection, full acquisition pipeline runner |
| Preprocessing | `src/preprocessing.py` | 404 | ✅ Substantially built | Normalize, sigma-clip, gap-fill, `wotan` detrend, quality gating, batch runner |
| Feature engineering | `src/features.py` | 665 | 🟡 Partially built | TLS run + geometry/stellar features exist here (not yet split into its own detection stage per the plan); has RAM guards, timeouts, parallel batch extraction |
| GPU features | `src/gpu_features.py` | 622 | ✅ Built, **not in original PRD** | CuPy/PyTorch-accelerated phase-fold, time-series stats, FFT features, morphological bins, odd/even depth, with CPU fallback for every GPU path — this is a genuine enhancement beyond the hackathon spec |
| Detection (standalone) | `src/detection.py` | 0 | ❌ Empty stub | TLS/BLS is currently *inline* inside `features.py`; needs extraction into its own module per `Implementation_Plan.md` §4 Stage 3 |
| Classification | `src/classifier.py` | 0 | ❌ Empty stub | No XGBoost/LightGBM ensemble, no SMOTE, no calibration, no model artifact |
| Parameter fitting | `src/fitting.py` | 0 | ❌ Empty stub | No `batman`/`emcee` MCMC fitting |
| Visualization | `src/visualization.py` | 0 | ❌ Empty stub | None of the 7 required plots exist |
| Reporting | `src/report.py` | 0 | ❌ Empty stub | No PDF/HTML/catalog generation |
| Orchestrator + 6 agents | `src/agents/*.py` | 0 each | ❌ Empty stubs | Entire multi-agent layer is unwritten |
| Frontend | `frontend/app.py` | 0 | ❌ Empty stub | No Streamlit dashboard yet |
| CLI entrypoint | `run_pipeline.py` | 0 | ❌ Empty stub | No end-to-end runnable pipeline yet |

**What already exists as real assets** (not just code): downloaded TESS FITS files for Sector 1 (`data/raw/sector_01_dvt/...`), ~28 detrended real light curves and ~50 synthetic-injection light curves (`data/processed/lc_*`), reference catalogs (ExoFOP TOI, NASA confirmed planets, SPOC TCE, TIC stellar params, unified labels, synthetic injections) under `data/catalogs/`, and a `feature_matrix.csv`. This is real, usable training/validation data — a head start most "SaaS from an idea" projects don't have.

**Bottom line**: the hard, novel astrophysics engineering (data acquisition semantics, quality gating, GPU-accelerated feature extraction) is the part that's done. The parts every ML-in-production system needs (classification, uncertainty quantification, reporting, orchestration, UI) are the part that's *not* done. That is actually the favorable order to be in.

---

## 3. The Core Pivot: research pipeline → SaaS product

| Dimension | Today (hackathon scope) | Target (industry SaaS) |
|---|---|---|
| Users | One person, one machine | Many tenants (labs, universities, individual researchers), isolated from each other |
| Invocation | `run_pipeline.py --sector 1 --camera 1 --ccd 1` on a laptop | Self-serve web app + REST/GraphQL API + API keys; jobs submitted, queued, tracked |
| Compute | Local CPU/GPU, `joblib.Parallel` | Autoscaled worker fleet (CPU + GPU node pools) in Kubernetes, queued via durable workflow engine |
| Storage | Local filesystem (`data/raw`, `data/processed`) | Object storage (S3-compatible) for FITS/light curves, managed Postgres for metadata, feature store for ML features |
| Orchestration | Sketch of `multiprocessing.Queue` agents (unbuilt) | Temporal (or equivalent) workflow engine: retries, timeouts, versioned workflows, per-tenant concurrency limits |
| ML lifecycle | `joblib.dump('models/ensemble_v1.0.pkl')` | Model registry (MLflow), versioned deployments, drift monitoring, canary rollout, scheduled retraining triggered by active-learning batches |
| Frontend | Streamlit prototype (planned) | Production web app (Next.js/React), Streamlit demoted to an internal admin/eval tool only |
| Access control | None | OIDC auth, RBAC, per-tenant data isolation, API-key scoping |
| Monetization | None | Usage-based billing (Stripe), plan tiers, quota enforcement |
| Reliability | "Works on my machine" | SLOs, health checks, alerting, incident response, backups/DR |
| Compliance | None | Audit logging, encryption at rest/in transit, data retention policy, (later) SOC 2 readiness |

---

## 4. Target Users & Jobs-to-be-Done

| Persona | Job to be done | Why they'd pay |
|---|---|---|
| Grad student / postdoc (astro) | "Screen a TESS sector for candidates without writing a pipeline from scratch" | Saves weeks of engineering; gets calibrated confidence + uncertainty for free |
| University research group / PI | "Give my group a shared, reproducible detection workflow with an audit trail" | Team workspaces, reproducibility, citable methodology |
| Citizen-science platform (e.g. Planet Hunters–style orgs) | "Pre-filter candidates before human volunteers review them" | Reduces volunteer review load via active-learning triage |
| Small observatory / planetarium / EdTech | "Show live, explainable exoplanet detection to visitors/students" | Turnkey dashboard, no ML expertise required |
| Independent researcher / hobbyist astronomer | "Run detection on my own light curve data" | Pay-as-you-go, no infra to manage |

---

## 5. v1 SaaS Feature Set

- Self-serve signup/login (email + OAuth), organization/workspace model
- Data ingestion: (a) point at a TESS sector/camera/CCD and let the platform pull from MAST, or (b) upload your own light curve (FITS/CSV)
- Job submission + status tracking (queued → preprocessing → detecting → classifying → fitting → reporting → done)
- Interactive results dashboard: candidate table (sortable/filterable, matches `Implementation_Plan.md` §6.3), per-target detail view with all 7 required plots, model metrics page
- Downloadable outputs: CSV/FITS catalog, 3-page PDF report, per-target HTML summary
- Active-learning review queue for low-confidence candidates (feeds back into retraining — this is a real product moat, not just a feature)
- REST API + API keys for programmatic access (so power users/labs can integrate into their own notebooks/pipelines)
- Usage dashboard + billing management
- Team roles: Owner / Admin / Member / Viewer per workspace

---

## 6. Pricing & Packaging (proposal)

| Tier | Target | Limits | Price shape |
|---|---|---|---|
| **Free** | Individual trial | 1 sector-slice or ≤500 targets/month, 7-day result retention | $0 — top-of-funnel |
| **Pro** | Lab / small team | Metered by targets processed + GPU-minutes, includes team workspace | Base seat fee + usage overage |
| **Enterprise** | University dept. / institution | Custom volume, SSO, VPC/private deployment option, SLA, dedicated support | Annual contract |

Metering unit: **targets processed** (light curves through the full pipeline) and **GPU-minutes** for feature extraction/CNN inference — both map directly onto real infra cost, which keeps unit economics honest from day one.

---

## 7. Target Architecture

### 7.1 High-level diagram

```
                                   ┌─────────────────────┐
                                   │      Web App         │
                                   │  (Next.js / React)   │
                                   └──────────┬───────────┘
                                              │ HTTPS
                                   ┌──────────▼───────────┐
                                   │   API Gateway / BFF   │  ← auth, rate limiting, tenant resolution
                                   └──────────┬───────────┘
                     ┌────────────────────────┼─────────────────────────┐
                     │                        │                         │
            ┌────────▼────────┐    ┌──────────▼──────────┐   ┌──────────▼──────────┐
            │  Tenant/Auth Svc │    │   Job Orchestrator   │   │   Billing/Usage Svc  │
            │  (OIDC, RBAC)    │    │  (Temporal workflows) │   │   (Stripe metering)  │
            └──────────────────┘    └──────────┬───────────┘   └──────────────────────┘
                                                │ dispatches durable workflow steps
                     ┌──────────────────────────┼──────────────────────────┐
                     │                          │                          │
           ┌─────────▼────────┐      ┌──────────▼─────────┐     ┌──────────▼─────────┐
           │ Acquisition Worker│      │ Preprocessing Worker│     │  Detection Worker    │
           │ (wraps            │      │ (wraps               │     │  (wraps              │
           │  acquisition.py)  │      │  preprocessing.py)   │     │  detection.py — new) │
           └─────────┬────────┘      └──────────┬─────────┘     └──────────┬─────────┘
                     │                          │                          │
           ┌─────────▼──────────────────────────▼──────────────────────────▼─────────┐
           │        Feature Worker (CPU/GPU node pool — wraps features.py +           │
           │        gpu_features.py, unchanged execution engine)                      │
           └─────────┬────────────────────────────────────────────────────────────────┘
                     │
           ┌─────────▼────────┐      ┌─────────────────────┐     ┌──────────────────────┐
           │ Classification    │      │  Fitting Worker       │     │  Reporting Worker      │
           │ Worker (loads     │      │  (batman+emcee,       │     │  (plots, PDF, CSV/     │
           │ model from        │      │  timeout+fallback,    │     │  FITS catalog)         │
           │ registry)         │      │  same pattern as      │     │                        │
           │                   │      │  features.py)         │     │                        │
           └─────────┬────────┘      └─────────────────────┘     └──────────────────────┘
                     │
           ┌─────────▼────────┐
           │ Active Learning   │──► low-confidence candidates surfaced in web app for expert
           │ Worker            │    labeling; approved labels feed scheduled retraining job
           └───────────────────┘

  Data plane (all workers read/write through these, never local disk):
  ┌───────────────┐  ┌────────────────┐  ┌────────────────┐  ┌──────────────────┐
  │ Object Storage │  │ Postgres (RLS   │  │ Feature Store   │  │ Model Registry    │
  │ (S3) — FITS,   │  │ multi-tenant)   │  │ (Feast) — 35+   │  │ (MLflow) —        │
  │ light curves,  │  │ — job/tenant/   │  │ feature vectors │  │ versioned ensemble│
  │ reports        │  │ candidate       │  │ per candidate   │  │ + calibration     │
  │                │  │ metadata        │  │                 │  │ artifacts         │
  └───────────────┘  └────────────────┘  └────────────────┘  └──────────────────┘

  Cross-cutting: Prometheus/Grafana metrics · OpenTelemetry tracing · structured logs (Loki)
                 · Terraform IaC · GitHub Actions CI/CD · Kubernetes (EKS/GKE) with
                 CPU + GPU node pools, HPA on queue depth
```

### 7.2 Frontend layer
- **Next.js/React**, TypeScript, Tailwind. Replaces the planned Streamlit dashboard as the *product* surface; Streamlit is kept only as an internal tool for the ML team to eyeball model metrics quickly (`Implementation_Plan.md` §6.6 content, repurposed).
- Plotly/Bokeh chart components from the original plan (light curve, periodogram, phase-fold, corner plot, odd/even, centroid) port over conceptually — same 7 visualizations, just served by a real charting layer in the web app instead of `st.plotly_chart`.

### 7.3 API / BFF layer
- REST (OpenAPI-documented) + webhooks for job-completion events. GraphQL only if a specific integration partner needs it — don't add it speculatively.
- All requests carry a tenant context resolved from the auth token; every downstream call is tenant-scoped.

### 7.4 Core domain services
The six empty `src/agents/*.py` stubs map directly onto real services — the *responsibilities* described in `Implementation_Plan.md` §5.3–5.10 are correct and reusable, only the transport changes (Temporal workflow activities instead of a raw multiprocessing queue):

| Old stub (unbuilt) | New service | Backed by |
|---|---|---|
| `acquisition_agent.py` | Acquisition Worker | `acquisition.py` (existing, reused as-is) |
| `preprocessing_agent.py` | Preprocessing Worker | `preprocessing.py` (existing, reused as-is) |
| `detection_agent.py` | Detection Worker | new `detection.py`, extracted from the TLS logic currently inline in `features.py` |
| `feature_agent.py` | Feature Worker | `features.py` + `gpu_features.py` (existing, reused as-is) |
| `classification_agent.py` | Classification Worker | new `classifier.py`, loads versioned model from MLflow registry |
| `fitting_agent.py` | Fitting Worker | new `fitting.py` |
| `reporting_agent.py` | Reporting Worker | new `report.py` + `visualization.py` |
| `active_learning_agent.py` | Active Learning Worker | new, persists labels to Postgres, triggers retraining workflow |
| `orchestrator.py` | Temporal workflow definitions | replaces this file entirely — don't hand-roll retry/timeout/health-check logic |

### 7.5 Orchestration layer — why Temporal over the original sketch
`Implementation_Plan.md` §5.2 proposes `multiprocessing.Queue` with manual retry counters and a 30-minute stall timeout. That's reasonable for a single-machine hackathon run; it is not adequate for a paid, multi-tenant service because it has no durability (a process crash loses state), no cross-machine scaling, and no per-tenant fairness. **Temporal** (or Prefect/Dagster as lighter-weight alternatives if the team wants to avoid running Temporal's own infra) gives you: workflow state survives worker crashes, built-in retry/backoff policies per activity, human-in-the-loop signals (perfect fit for the active-learning approval step), and per-tenant rate limiting on workflow execution.

### 7.6 Data layer
- **Object storage (S3-compatible)**: raw FITS, processed `.npz`, generated plots/PDFs — same file *shapes* as today's `data/raw|processed|outputs`, just tenant-prefixed in a bucket instead of local disk.
- **Postgres with row-level security**: tenants, users, jobs, candidates, labels, billing usage. RLS enforces tenant isolation at the database layer, not just in application code — this is the single highest-leverage security control for a multi-tenant SaaS and should be non-negotiable.
- **Feature store (Feast, or just well-indexed Postgres/Parquet tables at MVP scale)**: the 35+ features per candidate, so classification and active-learning retraining always read consistent, versioned feature definitions.
- **Model registry (MLflow)**: replaces `joblib.dump('models/ensemble_v1.0.pkl')` with versioned, queryable model artifacts, staged rollout (staging → production), and lineage back to the training run/dataset version.

### 7.7 ML platform & MLOps
- Training remains a **scheduled/triggered batch job**, not a live service — same 5-fold CV + Optuna + SMOTE + Platt calibration approach from `Implementation_Plan.md` §5.5 (a) is sound and doesn't need to change.
- Add: **drift monitoring** (feature distribution + prediction distribution vs. training baseline), **automatic retraining trigger** when the active-learning label batch crosses a threshold (the PRD already specifies "≥50 new labels" — wire this to actually fire a Temporal workflow instead of being a manual step).
- Add: **shadow evaluation** — new model versions score in parallel against production traffic before promotion, with the SHAP-based sanity check from `Implementation_Plan.md` §9 (Model Validation) gating promotion automatically.

### 7.8 Infrastructure & DevOps
- **Kubernetes** (EKS or GKE) with two node pools: CPU (acquisition/preprocessing/API) and GPU (feature extraction, CNN inference) — `gpu_features.py`'s existing CPU-fallback design means the same code path runs on either pool, which materially simplifies capacity planning.
- **Terraform** for all infra (VPC, K8s cluster, RDS/CloudSQL Postgres, S3 buckets, IAM) — no manually-clicked cloud resources.
- **CI/CD** (GitHub Actions): lint/type-check/unit-test on PR, integration test against a staging sector on merge to main, canary deploy to prod.
- **Secrets**: a managed secrets store (AWS Secrets Manager / GCP Secret Manager), never `.env` files in the repo or containers.

### 7.9 Multi-tenancy & auth
- **OIDC** via a managed provider (Auth0, Clerk, or WorkOS for enterprise SSO later) — don't build custom auth.
- Every table with tenant-owned data carries `tenant_id`; Postgres RLS policies enforce that a query can never cross tenants even if application code has a bug.
- API keys are tenant- and scope-bound (e.g., read-only vs. job-submission) and independently revocable.

### 7.10 Billing & metering
- **Stripe** for subscription + usage-based billing. Each completed job emits a usage event (targets processed, GPU-minutes consumed) to a metering pipeline that rolls up into Stripe usage records.
- Hard quota enforcement at the API layer (reject job submission once a tenant's plan limit is hit) *and* soft alerts at 80% usage.

### 7.11 Security, compliance & data governance
- Encrypt at rest (S3/Postgres default encryption) and in transit (TLS everywhere, including internal service mesh traffic).
- Audit log of all data-access and job-submission events per tenant (needed for enterprise sales conversations even before formal SOC 2).
- Data retention policy per tier (e.g., Free tier auto-deletes raw FITS after 7 days to control storage cost).
- Treat **NASA/MAST data licensing and attribution requirements** as a compliance item, not an afterthought — TESS/TIC/ExoFOP data has citation requirements that should be surfaced in every generated report.

### 7.12 Observability & reliability
- Metrics (Prometheus/Grafana): queue depth, per-stage job duration, GPU utilization, model inference latency, error rate per workflow step.
- Tracing (OpenTelemetry) across API → workflow → worker, so a slow job is debuggable end-to-end.
- Define **SLOs** early even if informal: e.g., "95% of ≤1000-target jobs complete in <15 min," "API p99 latency <500ms." These become the actual acceptance criteria for the product, layered on top of the science-accuracy criteria already in `Implementation_Plan.md` §11.

---

## 8. Completing the Scientific Core (the missing 65%)

This work is unavoidable regardless of SaaS ambitions — no product exists until candidates can be classified and reported. Sequence it to unblock the platform work in parallel:

1. **Extract `detection.py`** from the TLS logic already living in `features.py` (`run_tls`, lines ~210–242) into its own module matching `Implementation_Plan.md` §4 Stage 3 — this is refactor-not-invent work since the TLS call already exists and works.
2. **`classifier.py`** — XGBoost + LightGBM stacked ensemble, SMOTE balancing, Optuna tuning, Platt calibration, SHAP validation, exactly as specified in `Implementation_Plan.md` §4 Stage 5. The label data (`unified_labels.csv`, `feature_matrix.csv`) already exists to train against.
3. **`fitting.py`** — `batman` + `emcee` MCMC parameter estimation, following the same defensive pattern already established in `features.py` (timeout wrapper, RAM guard, fallback to `scipy.optimize` least-squares on non-convergence).
4. **`visualization.py` + `report.py`** — the 7 required plots and the 3-page PDF/HTML report. Build these to render from the feature/candidate schema, not ad hoc, so the same functions serve both the CLI report and the web app's chart components later.
5. **`run_pipeline.py`** — a real CLI entrypoint wiring stages 1–6 together locally, which doubles as the reference implementation each Temporal workflow activity wraps.
6. **Multi-mission generalization** (post-MVP, expands TAM): the acquisition layer's MAST/TIC assumptions are TESS-specific; Kepler/K2 and eventually PLATO support is a natural v1.1+ expansion that reuses almost the entire downstream pipeline unchanged (detection, features, classification, fitting all operate on a generic time/flux array regardless of mission).

---

## 9. Phased Roadmap

| Phase | Goal | Key deliverables | Exit criteria |
|---|---|---|---|
| **Phase 0 — Finish the science core** | A correct, locally-runnable pipeline | `detection.py`, `classifier.py`, `fitting.py`, `visualization.py`, `report.py`, `run_pipeline.py` | All acceptance criteria in `Implementation_Plan.md` §11 pass on Sector 1 |
| **Phase 1 — MVP SaaS (single-tenant-shaped, multi-tenant-ready)** | Prove people will submit jobs through a hosted product | REST API wrapping the pipeline as an async job; basic web app (submit job, view results); Postgres with `tenant_id` from day one; single-region deploy; manual/no billing | 5–10 pilot users (labs/individuals) successfully run a sector end-to-end via the web app |
| **Phase 2 — Multi-tenant hardening** | Make it safe to onboard strangers | Real auth (OIDC), Postgres RLS, API keys, Temporal orchestration replacing any direct pipeline calls, object storage instead of local disk, Stripe billing + quotas | Two unrelated tenants' data is provably isolated (pen-test / access-control audit); billing charges correctly |
| **Phase 3 — Production platform** | Reliability & scale | Model registry + drift monitoring + auto-retrain trigger, autoscaling GPU workers, full observability stack, defined SLOs, CI/CD with staging environment | Meets self-defined SLOs for 30 consecutive days under real pilot traffic |
| **Phase 4 — Enterprise readiness** | Land institutional customers | SSO, audit logs, VPC/private-deploy option, data retention controls, SLA contracts, multi-mission (Kepler/K2) support | First signed enterprise/institutional contract |
| **Phase 5 — Moat & scale** | Defensibility | Active-learning flywheel materially improving F1 over time on proprietary label volume, published validation against TOI catalog, partnerships with universities/observatories, PLATO readiness ahead of that mission's data release | Model performance and label volume become a competitive advantage, not just feature parity |

---

## 10. Team & Org Plan (by phase)

| Phase | Roles needed | Notes |
|---|---|---|
| 0–1 | You (ML/astro + backend), 1 frontend engineer (part-time ok) | Science core + thin API + basic UI |
| 2 | + 1 backend/platform engineer (auth, multi-tenancy, billing) | This is where most "SaaS-ness" gets built |
| 3 | + 1 DevOps/SRE (part-time ok initially), + 1 ML engineer (retraining/monitoring) | Reliability and MLOps become real jobs, not side tasks |
| 4+ | + sales/customer success (even 1 person), + security/compliance advisor | Enterprise deals need a human relationship and a compliance story |

---

## 11. Rough Cost Model

| Scale | Monthly infra estimate | Drivers |
|---|---|---|
| Pilot (Phase 1, <10 tenants, occasional jobs) | ~$200–500 | Small K8s cluster or even managed container service, small Postgres, minimal S3, no GPU pool running 24/7 (spot/on-demand GPU only during jobs) |
| Early growth (Phase 2–3, dozens of tenants, regular jobs) | ~$1.5k–4k | GPU node pool with autoscaling, larger managed Postgres, observability stack, Stripe fees |
| Institutional scale (Phase 4+) | Highly variable, budget $8k+ | Multi-region option, VPC-per-enterprise-customer possibility, dedicated support tooling |

Keep GPU nodes **scale-to-zero** outside active jobs at every stage before Phase 4 — the existing CPU fallback in `gpu_features.py` means correctness never depends on a GPU being available, only speed does, which is exactly the property that makes scale-to-zero cheap and safe.

---

## 12. Risks & Mitigations

| Risk | Impact | Severity | Mitigation |
|---|---|---|---|
| Market is small/niche (research astronomy) | Slow growth, hard to reach venture-scale | High | Validate willingness-to-pay in Phase 1 pilots before over-investing in Phase 2+ infra; consider citizen-science/EdTech as a volume complement to research labs |
| Free competition (NASA/MAST tools, open pipelines like `lightkurve`/`transitleastsquares` themselves) | Hard to justify price | High | Differentiate on *time saved* (hosted, no-ops, calibrated ensemble + active learning) and on *collaboration features* (teams, reproducibility, audit trail) academia's free tools don't offer |
| Multi-tenant data isolation bug | Catastrophic trust/legal failure | High | RLS at the DB layer (not just app-layer checks) + isolation pen-test before Phase 2 exits |
| Class imbalance / model accuracy shortfall (inherited from original PRD risk) | Product doesn't deliver its core promise | High | Same mitigation as `Implementation_Plan.md` §10: SMOTE + injection-recovery augmentation; additionally, be transparent in-product about confidence scores rather than hiding uncertainty |
| GPU cost runaway | Margin erosion | Medium | Scale-to-zero GPU pool, per-tenant quota hard caps, metering tied directly to billing |
| MAST/NASA data licensing/attribution missteps | Reputational/legal risk with the scientific community | Medium | Explicit attribution in every report; review NASA data use policies before GA launch |
| Orchestration complexity (Temporal) is new to the team | Slower Phase 2 delivery | Low–Medium | Time-box a spike; fall back to Prefect (lighter operational footprint) if Temporal's ops burden is too high for a small team |

---

## 13. Success Metrics (product KPIs, in addition to the science KPIs already in `Implementation_Plan.md` §11)

- Activation: % of signups that submit a job within 7 days
- Time-to-first-result (job submit → report ready)
- Monthly active tenants, targets processed/month, GPU-minutes/month
- Model macro-F1 trend over time (should trend up as active-learning labels accumulate — this is the flywheel metric)
- Net revenue retention (Pro/Enterprise), churn
- SLO adherence (job success rate, p99 API latency)

---

## 14. Immediate Next Steps (30/60/90)

- **Next 30 days**: finish `detection.py`, `classifier.py` — get to a locally-runnable pipeline on Sector 1 that hits the accuracy criteria in `Implementation_Plan.md` §11. This is the single highest-leverage thing to do next; nothing above is sellable without it.
- **Next 60 days**: `fitting.py`, `visualization.py`, `report.py`, `run_pipeline.py` complete; stand up a thin REST API (even without auth yet) that wraps the CLI as an async job with Postgres job tracking; start Phase 1 web app.
- **Next 90 days**: recruit 5–10 pilot users from personal/academic network, run them through the hosted MVP, and use their feedback to decide whether Phase 2 (real multi-tenancy + billing investment) is justified before building it.

---

*This document is a living plan — revisit the phase gates above as real usage data comes in rather than executing the whole roadmap on faith.*
