# AeroReady AI — Predictive Maintenance & Fleet Availability

**SIH 2026 · PS 26249 · Ministry of Defence / Defence Services Staff College · Transportation & Logistics**

AeroReady AI is a working prototype of an integrated maintenance-intelligence platform. It connects the whole chain in one application and one database:

**aircraft health data → predictive risk → work orders → spare-parts coordination → fleet availability**

> **Demonstration prototype.** All aircraft, components, part numbers, teams and data are fictional and synthetic. The risk model is trained and evaluated on synthetic data only and is **not** validated for real-world airworthiness or maintenance decisions.

---

## Quick start

```bash
# 1. (optional) create a virtual environment
python -m venv .venv
# Windows:  .venv\Scripts\activate        macOS/Linux:  source .venv/bin/activate

# 2. install dependencies
python -m pip install -r requirements.txt

# 3. run the app (opens http://localhost:8501)
python -m streamlit run app.py

# 4. run the tests
python -m pytest -q
```

**Sign in** with a demo account:

| Username | Password | Role |
|---|---|---|
| `admin` | `admin@123` | Administrator (full access, can reset demo data) |
| `planner` | `planner@123` | Maintenance Planner (full operational access, cannot reset) |

Passwords are stored as salted PBKDF2-SHA256 hashes in the local SQLite database. After 5 failed attempts, sign-in pauses for 30 seconds. Accounts survive **Reset demo data**. This is a demonstration login, not production identity management.

On first launch the app trains the risk model (about 2 seconds) and generates the fictional dataset in `data/`. Later launches reuse both. To start again from scratch, use **Reset demo data** in the sidebar, or delete the `data/` folder. Setting the `AEROREADY_DATA_DIR` environment variable points the app at a different data folder.

No internet access, API keys or external datasets are needed.

---

## 5-minute demo script (end-to-end acceptance flow)

| # | Step | Where | What you see |
|---|------|-------|--------------|
| 0 | Sign in as `admin` | Login page | The sidebar shows the user and role, with a sign-out button. |
| 1 | Inspect fleet readiness | **Fleet Command Centre** | Availability 87.5% (21/24). Two aircraft are under maintenance and one is grounded (AC-402, hydraulic pump with the part out of stock). Also shown: alerts, trends and the readiness board. |
| 2 | Select an aircraft | Pick **KM-306** on the dashboard, then **Open digital twin** | Every component of KM-306 with its risk badge. The fuel pump is Critical. |
| 3 | Change sensor values | Digital Twin, simulation controls | Move vibration, temperature or pressure. The score, category and decision-support text recompute instantly. *Commit simulated reading* stores a new timestamped reading (source `digital-twin`). The simulated state (Serviceable / Degraded / Under inspection / In maintenance / Returned to service) follows the work-order lifecycle. |
| 4 | Explanation | Same page, plus **Predictive Maintenance**, then *What-if* / *Model card* | Change-versus-reference chart (an association, not proven cause), the largest differences in plain language, and the validation/test comparison of ML and rule-based scoring. |
| 5 | Create a work order | *Create maintenance work order* expander | A live parts check runs before you submit. |
| 6 | Spare-part availability | Same form, then **Spare-Parts Intelligence** | PN-PRP-1020 is out of stock, so the work order becomes **Awaiting Parts** and KM-306 becomes *Unavailable*. Place a purchase order, then *Mark as received*. The stock is auto-allocated and the work order becomes **Open**. |
| 7 | Schedule / prioritise | **Maintenance Planner**, then *Suggested schedule*, *Generate*, *Apply* | Priority queue, Gantt chart by team, a "why this date" note for each job, and a no-double-booking check. Under *Manage*: start the work, then complete it. |
| 8 | Fleet impact | **Availability Simulator** and the dashboard | Two separate comparisons: **proactive vs reactive** on identical inputs, and **existing work orders vs resource changes** (parts expedited, extra team, FIFO). After completion the aircraft returns to *Serviceable*. |
| 9 | Persistence | Restart `streamlit run app.py` | All work orders, stock, readings and logs are still there (SQLite). |

---

## What each module does

| Module | Page | Highlights |
|---|---|---|
| A. Fleet Command Centre | `views/dashboard.py` | KPIs computed from the database: availability, serviceable/under-maintenance/unavailable, components needing attention, high-priority and overdue work orders, critical spares shortages. Also a readiness board, live alerts, 30-day risk trend, weekly maintenance activity, risk by component type and an activity log. |
| B. Maintenance Data Hub | `views/data_hub.py` | Linked registry of aircraft, components, sensor readings, maintenance records, faults, work orders, purchase orders and the maintenance programme (flight-hour based). Search and filter on every table, an aircraft detail view, CSV export of any table, and **validated, all-or-nothing CSV import** (spare parts, sensor readings, maintenance records) with row-level errors. |
| C. Predictive Maintenance | `views/predictive.py` | Fleet risk ranking, per-component explanation, recommendation and work-order creation (single or bulk). An interactive what-if over all 7 features, and a full model card. |
| D. Component Digital Twin | `views/digital_twin.py` | Identity and usage, simulation controls, optional live-style noisy telemetry, health history, per-sensor trends, a linear-trend degradation projection ("days until High risk"), commit-reading-to-record, maintenance/fault history and work-order creation. |
| E. Maintenance Planner | `views/planner.py` | Priority queue with due dates and parts checks, a suggested schedule (heuristic, Gantt), manual assignment with overlap and skill validation, and the start / complete / unschedule / cancel lifecycle. |
| F. Spare-Parts Intelligence | `views/inventory.py` | Stock status, reserved/free stock, tasks waiting for parts, predicted demand from risky components, suggested reorder quantities, purchase orders, simulated deliveries with automatic allocation, stock-take adjustment, and shortage impact in aircraft-days. |
| G. Availability Simulator | `views/simulator.py` | Availability now, aircraft awaiting maintenance, planned downtime, before/after plan completion, up to 4 scenario curves, a per-aircraft downtime timeline, stated assumptions, and an automatically derived "binding constraint" reading. |

---

## How it works

### Architecture

```
app.py                     Streamlit entry point (login gate, navigation, sidebar, reset)
aeroready/
  config.py                paths, fictional catalogue, risk bands, statuses
  db.py                    SQLite schema and helpers (new connection per operation)
  seed.py                  reproducible synthetic fleet (24 aircraft, 150 components, 30-day telemetry)
  model.py                 synthetic training data, monotonic GBM, metrics, explanations, rule fallback
  risk.py                  feature derivation, fleet scoring, recommendations
  workorders.py            work-order lifecycle with consistent parts reservation
  inventory.py             stock status, reorder heuristic, purchase orders
  planner.py               greedy scheduling heuristic (no team overlap)
  availability.py          aircraft status rule, day-by-day simulator, scenarios
  csv_io.py                validated CSV import/export
  auth.py                  demo login: hashed passwords, roles
  ui.py                    theme, KPI cards, badges, Plotly template, shared forms
views/                     one file per module (A–G) + login.py
assets/login_bg.jpg        login-page background photo (user-supplied)
tests/                     pytest suite (model, workflow, planner, availability, CSV, UI smoke)
```

### Risk model (experimental)

* **Features:** operating hours as % of rated life, hours since last maintenance, component age, temperature deviation, vibration, pressure deviation, prior faults.
* **Synthetic data:** 8,000 records (seed 42). A hidden degradation variable drives noisy sensors and the probability of a synthetic "unscheduled removal within 50 FH". The model never sees the hidden variable.
* **Model:** `HistGradientBoostingClassifier` with **monotonic constraints**, so a worse input can never lower the score.
* **Evaluation (60/20/20):** stratified train / validation / test split made *before* fitting. The candidate ML model is fitted on train; **ML vs the rule-based score is chosen on validation ROC-AUC only**; the ML model is then refitted on train+validation and the frozen choice is evaluated once on the test split. Test metrics are reported for **both** methods. For real sequential sensor data, split by aircraft/component and time instead.
* **Selection rule** (`config.py`): ML is used only if its validation ROC-AUC is ≥ `MIN_ACCEPTABLE_AUC` and beats the rule score's validation ROC-AUC by at least `ML_MIN_AUC_ADVANTAGE` (default 0). On the current synthetic data the two are essentially tied (validation ≈ 0.82 for both), so the transparent rule-based score is selected; the model card shows both. `MODEL_VERSION` is bumped whenever the evaluation design changes so a stale pickled model is never silently reused.
* **What the score means:** a *prototype maintenance-risk score* from 0 to 100. The rule-based score is a weighted index, and the ML score is uncalibrated, so neither is a validated failure probability: a 70 does not mean a 70% chance of failure.
* **Explanations:** permutation importance (global). For a single prediction, each feature is reset to the healthy-component median and the change in the score is shown. This is a change relative to a reference profile, not proof that a sensor caused the risk.
* **Bands:** Low < 0.20 ≤ Medium < 0.45 ≤ High < 0.70 ≤ Critical (applied to the 0–1 score).
* **Recommendations are decision support only.** Text reads e.g. "High simulated maintenance risk. Prioritize review by authorized maintenance personnel under approved procedures. This model does not determine aircraft serviceability." The model never presets grounding: the work-order checkbox is a simulated decision the user must set.

### Data ingestion and telemetry

* **Idempotent:** every sensor reading carries a `source` and a stable `source_record_id`. Re-importing a file (or re-running a connector) skips readings already stored and reports how many were skipped. Maintenance records are de-duplicated on (component, date, type, description).
* **Traceable:** every accepted or rejected import is written to `ingestion_log`; the Data Hub shows each source's record count, latest reading and last validation status.
* **Connector interface:** `aeroready/ingestion.py` defines `Connector` (`fetch` + `normalize`) and ships a **simulated** JSON/API source that goes through the same validation and persistence as CSV imports. It is not a real aircraft-system or IoT integration.
* **Stale telemetry:** a component whose newest reading is older than `STALE_READING_DAYS` (7), or that has none, is flagged *Stale* / *No data* in the dashboard, Data Hub, digital twin and recommendations. Missing data is never read as healthy.
* **Existing databases** are migrated automatically (new `source_record_id` column and `ingestion_log` table).

### Digital twin (simulation-based)

* Component state is derived from the work-order workflow, so there is no second state mechanism: *Serviceable*, *Degraded*, *Under inspection* (work order raised), *In maintenance* (in progress), *Returned to service* (completed within 7 days).
* Completing a work order restores the simulated condition, writes a `post-maintenance` reading and a maintenance record together.
* The trend line needs at least `MIN_TREND_SAMPLES` (5) valid readings since the last maintenance reset; otherwise the page says **insufficient data**. The projection is an illustrative linear extrapolation, not a remaining-useful-life estimate.

### Business rules

* **Aircraft status:** *Under Maintenance* if any work order is In Progress. Otherwise *Unavailable* if any active work order grounds the aircraft. Otherwise *Serviceable*. **Availability = serviceable / total**, and this one rule is used everywhere.
* **Parts:** creating a work order reserves stock if enough is free; otherwise the order becomes *Awaiting Parts*. Deliveries and stock corrections auto-allocate to waiting orders by priority. Completion consumes the stock. Cancellation releases it. The invariant *reserved = Σ reserved active orders* is tested.
* **Completion:** writes a maintenance record, closes the component's open faults and restores its simulated condition (a replacement resets hours, age and faults; servicing removes most of the sensor deviation).
* **Scheduling heuristic:** jobs are ordered by grounding, then severity, then risk, then age. A job cannot start before its parts arrive (PO date, or the lead time if nothing is ordered). Each job goes to the qualified team that can **finish earliest**, and a team is never double-booked. This is a priority-based suggested schedule from a greedy rule of thumb and **not** a globally optimal scheduler. (A future version could use Google OR-Tools CP-SAT with an explicit objective and constraints.)
* **Simulator:** an aircraft is unavailable during its maintenance windows, and also from the grounding date until the work ends. Scenario assumptions are shown in the app. Results are scenario comparisons, **not forecasts** of actual availability.
  * *Proactive vs reactive* uses the same fleet, horizon, existing work orders and unaddressed High/Critical components; positive `improvement = reactive − proactive` means fewer simulated unavailable aircraft-days with proactive maintenance, negative means proactive did worse under the assumptions. The reactive failure timings are user-set, not model predictions.
  * *Existing work orders only* (formerly "Current plan") excludes unaddressed predicted failures, so it is a baseline for **resource** questions (parts, teams, ordering) and must not be compared with proactive maintenance as evidence of predictive-maintenance value.

---

## Testing

```bash
python -m pip install -r requirements.txt     # or requirements-lock.txt for the exact tested versions
python -m pytest -q
```

GitHub Actions (`.github/workflows/tests.yml`) runs the suite on every push and pull request.

* **Model:** reproducibility, 60/20/20 disjoint stratified splits, validation-based method selection (including a test that scrambling the *test* labels changes neither the choice nor the validation scores), both methods reported on test, monotonicity, risk bands, rule fallback, missing values, version bump.
* **Scenario fairness:** proactive and reactive share the same fleet, horizon, work orders and at-risk set; the existing-work-orders baseline excludes predicted failures; comparison direction is stated explicitly.
* **Ingestion:** repeated CSV import creates no duplicates (readings and maintenance records), connector uses the same validation/persistence, malformed payloads rejected and logged, source summary, legacy-database migration, stale/missing telemetry flagged.
* **Digital twin:** state rules, the full lifecycle (new reading, work order, completion, state, history, maintenance record, lower subsequent risk), minimum-sample and post-maintenance trend guards.
* **Safety wording:** no authoritative grounding/airworthiness language, no model-preset grounding, bulk work orders do not ground.
* **Workflow, planner, availability, CSV, auth, UI smoke:** as before, plus parts-ready / qualified-team / no-double-booking scheduling checks.

---

## Limitations (honest summary)

* All data is synthetic. Model metrics measure recovery of the synthetic generator's rules, **not** real-world predictive power.
* The digital twin is a software simulation (statistical sensor model plus an illustrative linear trend), not a physics or engineering model.
* Scheduling is a transparent greedy heuristic with whole-day granularity. It does not model shifts, hangar slots, tooling, technician certification or multi-part kits.
* Each work order uses a single part number. Purchase-order lead times are deterministic.
* Demo login only: two fixed roles, with no account management UI, password reset or concurrent-edit handling. Run it locally, not on a public network.

---

## Deployment boundary: synthetic vs simulated vs required for real use

| Area | In this prototype | What real deployment would need |
|------|-------------------|---------------------------------|
| Failure prediction | Model trained and evaluated on synthetic data; prototype risk score | Appropriately labelled historical health and maintenance data, calibration and independent validation |
| Aircraft-system integration | CSV import and one **simulated** JSON connector | Approved interfaces, source-system documentation (units, timestamps, error rules), access permissions and credentials |
| Maintenance decisions | Decision-support text; grounding is a simulated work-order choice | Approved maintenance procedures and authorized personnel; the model never determines serviceability |
| Agency coordination | Fictional teams, parts and capacities | Authoritative capacities, qualifications, facility constraints and work rules |
| Digital twin | Simulation with an illustrative trend | Physical or empirical degradation models and engineering validation |
| Availability benefit | Scenario comparison under stated assumptions | Fair evaluation on representative operational data with consistent baselines |
| Security | Demo login with two fixed roles | Strong authentication, authorization, audit and integration with operational systems |

Until those inputs exist, treat AeroReady AI as a demonstration prototype on fictional data.
