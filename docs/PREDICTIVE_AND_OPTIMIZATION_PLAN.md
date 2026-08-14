# Predictive & Optimization Roadmap

This document plans three additions discussed for Batch Insight Analyzer and how
they fit the existing staged pipeline. It deliberately separates **three
different jobs** that are easy to conflate:

| Job | Question | Tools | Nature |
|---|---|---|---|
| 1. Driver ranking | "Which parameters associate with the outcome?" | PLS, RF, **XGBoost+SHAP** | Associative |
| 2. Forecast / drift | "Given the batch sequence, what comes next / is it drifting?" | **Chronos-2**, EWMA, control charts | Temporal |
| 3. Propose test batches | "What should we try next?" | **Bayesian optimization**, **DoE** | Prescriptive (valid only under intervention) |

Tools do not substitute across jobs. In particular, **Chronos-2 does not replace
Random Forest** — they answer different questions. Random Forest is kept; it also
doubles as the uncertainty-bearing surrogate for the Bayesian-optimization
proposer.

## Non-negotiable design constraints (from CLAUDE.md / docs/DEVELOPMENT.md)

- **The base app must keep working with zero optional dependencies installed.**
  XGBoost, Chronos-2, and the optimizer are gated like `advanced_methods.py`
  (CatBoost): `find_spec(...)` check, an `available`/`reason` result shape, and a
  tab that shows install instructions instead of crashing.
- **Python computes facts; the LLM narrates them.** Any proposed parameter,
  band, or forecast must be a deterministic module output before the interpreter
  sees it.
- **Never present proposals as optimized setpoints.** BO/DoE outputs are
  *candidate experiments to run*, not production targets. `report_validator.py`
  gets new guards so the narrative cannot upgrade them to setpoints.
- New methods get a `method_registry.py` entry and a `docs/THEORY.md` note.

---

## 1. XGBoost + SHAP (driver ranking, Job 1)

**Why:** stronger non-linear/tabular learner than RF on small/medium data;
native TreeSHAP gives global *and* per-batch attribution. A third independent
vote strengthens the method-agreement logic that `confidence.py` already uses.

**Placement:** `analysis/advanced_methods.py` (alongside CatBoost), **not** the
base `methods.py`. Rationale: it stays optional, so the base ranked-driver score
(`combine_ranked_drivers`, PCA+PLS+RF) remains computable without `xgboost`
installed — identical posture to the existing CatBoost note ("does not currently
change the main ranked-driver score").

**SHAP without the heavy `shap` package:** use XGBoost's built-in
`Booster.predict(..., pred_contribs=True)` (TreeSHAP). This mirrors how the
CatBoost path uses *native* SHAP and avoids adding the large `shap` dependency.

**New code:**
- `run_xgboost_shap(dataframe, process_columns, outcome_column, validation_order_column=...)`
  in `advanced_methods.py`, mirroring `run_catboost_shap`: same
  `available`/`reason` shape, same `make_time_ordered_split` reuse, same
  `normalize_scores` → `xgb_score` per process variable, plus a long SHAP table.
- `get_xgboost_dependency_status()` like `get_catboost_shap_dependency_status()`.
- One-hot encode categoricals via the existing `preprocess_features` (XGBoost has
  no native categorical handling as clean as CatBoost's; reuse the encoder so
  `feature_groups` collapse back to process variables consistently).

**Wiring:**
- `app.py`: extend the existing optional-models tab area (currently CatBoost) with
  an XGBoost+SHAP sub-tab; reuse `render_catboost_shap_tab` structure.
- `confidence.py`: when an XGBoost result is present, add it as a fourth
  agreement vote and surface it in the confidence breakdown. Base label math is
  unchanged when it is absent.
- `method_registry.py`: add `xgboost_shap` (status `optional`).

**Tests:** `tests/test_advanced_methods.py` gains XGBoost cases guarded by
`importlib.util.find_spec("xgboost")` (skip when absent), plus a synthetic-truth
check that XGBoost recovers the planted `feed_rate`/`temperature` signal.

---

## 2. Chronos-2 (forecast / drift, Job 2)

**Why:** time-series early-warning on a campaign — "given the trajectory of
recent batches, is the outcome drifting?" Complements the existing control charts
in `utils/plots.py`. It is **not** a driver method and does **not** join the
ranked-driver table.

**Gate it twice:**
1. On dependency (`torch` + Chronos available), like CatBoost.
2. On **data**: only offer it when `audit.py`'s drift check indicates real
   temporal structure and there is a usable order column with enough points.
   If batches are effectively independent, there is nothing to forecast.

**Honest fallback:** ship a dependency-free **EWMA / rolling-trend** forecaster in
the same module first. For dozens-to-low-hundreds of batches it is often more
interpretable than a pretrained foundation model, and it has no torch/download
cost. Chronos-2 becomes the "upgrade" path.

**New code:**
- `analysis/forecast.py`:
  - `run_trend_forecast(...)` — EWMA/linear drift, always available (numpy/pandas).
  - `run_chronos_forecast(...)` — optional, `available`/`reason` shape; loads the
    model lazily; supports covariates if using the Bolt/-2 variant.
  - `get_chronos_dependency_status()`.

**Wiring:** `app.py` new "Stability & forecast" panel next to control charts;
`method_registry.py` entry `chronos_forecast` (status `optional`).

**⚠ Heavy/awkward dependency — see Dependencies below.** torch is large, and
Chronos weights download from HuggingFace Hub (network egress), which conflicts
with the local/sensitive-data posture. Keep it strictly opt-in and documented.

---

## 3. Bayesian optimization + DoE (propose test batches, Job 3)

**Why:** the only honest path to "suggest new parameters." Output is **candidate
experiments to run**, not setpoints — the validity comes from physically running
the proposal, which breaks confounding by intervention. This is the feature that
most needs the `report_validator.py` guard rails.

**Two-step, conservative-first:**

**3a. DoE candidates around historical response bands (ship first).**
Reuse the response bands already computed in `evidence.py` (quartile means +
best-observed bin). Generate a small space-filling / factorial design centered on
those bands using `scipy.stats.qmc` (Latin Hypercube / Sobol — **scipy is already
a dependency, zero new packages**). Optional classical CCD/factorial via
`pyDOE3`. This is defensible, reuses an existing fact, and matches the app's
conservative posture.

**3b. RF-surrogate Bayesian optimization (graduate to this).**
Use the **existing sklearn `RandomForestRegressor` as the surrogate** (tree
variance → uncertainty, SMAC-style), with an acquisition function (Expected
Improvement / UCB) computed in numpy over the DoE candidate pool from 3a. This:
- handles the mixed continuous + categorical parameter space (supplier,
  bioreactor, shift) better than a GP,
- adds **no new heavy dependency** (RF + numpy + scipy already present),
- keeps RF in the stack with a second, concrete job.
Optional `scikit-optimize` (GP-EI) can be offered for smooth all-continuous
spaces, but is not required.

**New code:**
- `analysis/doe.py`: `propose_doe_candidates(response_bands, bounds, tunable_vars, ...)`
  → candidate parameter table + coverage notes (flags extrapolation outside the
  historical cloud).
- `analysis/optimize.py`:
  `propose_experiments(model_or_dataframe, candidates, acquisition="ei"|"ucb", ...)`
  → ranked candidate experiments with predicted mean, uncertainty, and acquisition
  score. Every row is labeled "candidate experiment," never "setpoint."

**Wiring:**
- `app.py` new "Experiment proposals" tab; lock copy to investigation framing.
- `evidence.py`: include proposals in the pack **only** under a clearly-labeled
  "candidate experiments (hypotheses)" section.
- `report_validator.py`: add guards flagging any narrative that turns a proposal
  into a setpoint ("set X to", "optimal value is", "recommended setpoint").
- `method_registry.py`: entries `doe_candidates` and `bayes_opt`.
- `docs/THEORY.md`: notes; consider a short `docs/EXPERIMENT_DESIGN.md` spelling
  out the "candidates, not setpoints" contract for regulatory readers.

**Tests:** `tests/test_doe.py` (candidate coverage, extrapolation flagging),
`tests/test_optimize.py` (acquisition ranks a known optimum region higher on the
synthetic U-shaped `ph_setpoint` relationship).

---

## Dependencies (new)

Tiered to preserve the "base app needs nothing optional" rule. Pins are starting
points — **verify against the installed `numpy`/`pandas`/`scipy`/`scikit-learn`
and adjust before committing a lockfile.**

| Package | Job | Tier / file | Notes |
|---|---|---|---|
| `xgboost` | 1 | `requirements-optional.txt` | Use native `pred_contribs` TreeSHAP; **no `shap` package needed**. Lightweight, wheels for all platforms. |
| `scipy.stats.qmc` | 3a | already installed | Latin Hypercube / Sobol DoE. **No new dependency.** |
| `pyDOE3` | 3a | `requirements-optional.txt` (optional) | Only if classical factorial / central-composite designs are wanted. Pure-Python, light. |
| sklearn RF + numpy | 3b | already installed | RF-surrogate BO + numpy acquisition. **No new dependency.** |
| `scikit-optimize` | 3b | `requirements-optional.txt` (optional) | Only for GP-EI on smooth continuous spaces. Optional alternative to the RF surrogate. |
| `torch` | 2 | `requirements-forecast.txt` | **Heavy.** Choose CPU vs CUDA wheel deliberately. Hundreds of MB. |
| `chronos-forecasting` (or `autogluon-timeseries`) | 2 | `requirements-forecast.txt` | Pulls `transformers`/`accelerate`. **Downloads model weights from HuggingFace Hub** — network egress; conflicts with offline/sensitive posture. Pre-download + pin the model. Confirm the exact distribution that ships "Chronos-2" at implementation time. |

Install commands:

```powershell
# Job 1 driver explainability (CatBoost already here; XGBoost added)
.\.venv\Scripts\python.exe -m pip install -r requirements-optional.txt

# Job 2 forecasting (heavy, network download for weights)
.\.venv\Scripts\python.exe -m pip install -r requirements-forecast.txt
```

Net effect: **Jobs 1 and 3 add at most one light optional package each** (and 3a/3b
core paths add none). **Only Job 2 (Chronos-2) introduces a heavy dependency**,
and it is isolated in its own requirements file with an EWMA fallback so the
feature is usable without it.

## Suggested sequencing

1. **XGBoost+SHAP** — lowest risk, reuses the CatBoost pattern exactly.
2. **DoE-around-response-bands (3a)** — zero new heavy deps, reuses `evidence.py`.
3. **RF-surrogate BO (3b)** — builds on 3a, still no heavy deps; add the
   `report_validator.py` setpoint guards here.
4. **EWMA forecast then Chronos-2 (2)** — last, gated on the drift check; the only
   step that pulls torch.
