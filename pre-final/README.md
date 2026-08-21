# ImmunoVision — Pre-Final Defense Prototype

Predictive analytics dashboard for child immunization coverage
**San Jacinto Rural Health Unit, Pangasinan**

---

## What this build is

A complete, running system: three role-based portals, child records, coverage
analytics, vaccine inventory and requests, report generation with export, and
user administration.

**The machine-learning model is deliberately not part of this build.** Risk
flags are produced by a transparent rule-based scorer (`data_processor.py` →
`SCORING_RULES`). The Risk Prediction and Continuation Predictor pages say so
on screen and report **no** accuracy, precision, or recall figures, because no
model has been trained here. The training code remains in the repository for
the final defense.

---

## Setup

Requires Python 3.10 or newer. From this folder:

```bash
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux

# 2. Install dependencies (5 packages, no ML libraries)
pip install -r requirements.txt

# 3. Run
python app.py
```

**There is no database step.** The demo data ships pre-built in
`data/immunovision.db`, so the app starts in under a second. If that file is
ever missing or deleted, `app.py` rebuilds it automatically on startup (about a
minute, once) — so `python app.py` always works.

To rebuild the data deliberately: `python seed.py --reset`

Open <http://127.0.0.1:5000>

---

## Demo accounts

All use the password `password123`.

| Role          | Username    | Lands on               |
| ------------- | ----------- | ---------------------- |
| Administrator | `jdelacruz` | Admin dashboard        |
| RHU Personnel | `msantos`   | RHU dashboard          |
| BHW           | `areyes`    | BHW dashboard (Guibal) |

Sign-in is by credentials alone — the account's own role decides which portal
it opens.

---

## Suggested demo path

1. **Log in as `msantos`** (RHU) — dashboard shows 563 children across 19 barangays.
2. **Child Records** — 563 rows by 23 columns; the ID column stays pinned while
   you scroll sideways through the 15 vaccine doses.
3. **Risk Prediction** — note the prototype-scoring banner.
4. **Continuation Predictor** — shows the actual weighted rules in use.
5. **Coverage Analytics / Municipality Map** — barangay-level breakdown.
6. **Reports** — choose a barangay and a period, Generate, then Excel or Print.
7. **Log in as `areyes`** (BHW) — the same system scoped to one barangay.
8. **Log in as `jdelacruz`** (Admin) — user management and activity logs.

---

## Data

- `data/raw/` — the two real barangay registry exports (Guibal, Macayug),
  387 children. Names are anonymised on load.
- The other 17 barangays are a simulated demo population (176 children),
  labelled as prototype data in the UI.
- `python seed.py --reset` rebuilds the database from scratch.

---

## For the final defense

To train and enable the ML model:

```bash
pip install -r requirements-ml.txt
python data_processor.py          # trains and saves data/risk_model.joblib
```

The application detects the saved model automatically: the prototype banner
disappears, the Scoring Rules table is replaced by the Model Comparison table,
and real metrics are shown. **No code changes are needed** —
`predict_for_child()` returns the same shape either way.

---

## Project layout

```
app.py                 Flask app: routes, analytics, all three portals
models.py              SQLAlchemy models (9 tables)
data_processor.py      OSEMN pipeline, feature engineering, rule-based scorer,
                       and the (currently unused) model training code
seed.py                Builds the demo database
templates/             30 Jinja templates
  _icons.html          Central SVG icon set (42 icons)
  _brand.html          ImmunoVision logo mark
static/css/style.css   Design system
static/js/vendor/      Chart.js (vendored — no internet needed at runtime)
data/raw/              Real barangay registry CSVs
```
