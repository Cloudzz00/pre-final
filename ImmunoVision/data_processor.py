"""
data_processor.py
==================
Everything data-related for ImmunoVision in one place: reference
constants, real registry CSV ingestion/cleaning, synthetic data
simulation, feature engineering, ML model training/inference, and the
live analytics used by the dashboards.

This module implements the OSEMN pipeline described in the capstone
proposal:
  Obtain    -> load_registry_csv() / load_and_merge_real_registries()
  Scrub     -> _clean_date(), _clean_sex(), _clean_barangay(), dedup/impute
  Explore   -> compute_features() (feature engineering)
  Model     -> train_and_save_model()
  iNterpret -> RiskPredictor.predict_for_child() + explain_prediction()
"""
import csv
import glob
import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta

# NOTE (pre-final defense build): joblib / numpy / scikit-learn / xgboost are NOT
# imported at module load. They are imported lazily inside train_and_save_model()
# and _load_artifact() only. The running prototype therefore needs none of them
# installed, while the training code stays in the repository for the final defense.

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
MODEL_PATH = os.path.join(BASE_DIR, "data", "risk_model.joblib")

# ---------------------------------------------------------------------------
# Reference data: San Jacinto barangays and the DOH EPI schedule.
# ---------------------------------------------------------------------------

BARANGAYS = [
    "Awai", "Bolo", "Capaoay", "Casibong", "Imelda",
    "Guibal", "Labney", "Magsaysay", "Lobong", "Macayug",
    "Bagong Pag-asa", "San Guillermo", "San Jose", "San Juan",
    "San Roque", "San Vicente", "Santa Cruz", "Santa Maria", "Santo Tomas",
]

# The 7 vaccine antigens tracked in Vaccine Inventory / Vaccine Requests.
VACCINE_ANTIGENS = [
    ("BCG", "BCG"),
    ("HEPB", "Hepatitis B"),
    ("DPT", "DPT-Hib-HepB"),
    ("OPV", "OPV"),
    ("PCV", "PCV"),
    ("IPV", "IPV"),
    ("MMR", "MMR"),
]

# What each antigen protects against. Used to show a health worker the
# consequence of a missed dose rather than only the dose code. Wording follows
# the DOH EPI programme description.
ANTIGEN_PROTECTS = {
    "BCG":  "Tuberculosis",
    "HEPB": "Hepatitis B",
    "DPT":  "Diphtheria, pertussis, tetanus, Hib, hepatitis B",
    "OPV":  "Polio",
    "PCV":  "Pneumococcal disease (pneumonia, meningitis)",
    "IPV":  "Polio",
    "MMR":  "Measles, mumps, rubella",
}


def vaccines_for_missed(missed_codes):
    """Human-readable vaccine names for the missed dose codes, de-duplicated.

    Groups multiple doses of the same vaccine: a child missing DPT1, DPT2, DPT3
    shows 'DPT-Hib-HepB' once, not three times."""
    antigen_of = {code: ant for code, _n, ant, _d, _r in VACCINE_SCHEDULE}
    name_of = {code: name for code, name, _a, _d, _r in VACCINE_SCHEDULE}
    seen, out = set(), []
    for code in missed_codes:
        ant = antigen_of.get(code, "")
        # Group by antigen so DPT1/DPT2/DPT3 → one entry
        if ant and ant not in seen:
            seen.add(ant)
            # Use the first dose name as the label, strip the dose number
            label = name_of.get(code, code).rsplit(" ", 2)[0]
            out.append(label)
        elif not ant and code not in seen:
            seen.add(code)
            out.append(name_of.get(code, code))
    return out


def diseases_for_missed(missed_codes):
    """Diseases the missed doses would have protected against, de-duplicated.

    A child missing four DPT-series doses is exposed to one set of diseases, not
    four, so the antigens are collapsed before the names are listed."""
    antigen_of = {code: ant for code, _n, ant, _d, _r in VACCINE_SCHEDULE}
    seen, out = set(), []
    for code in missed_codes:
        ant = antigen_of.get(code)
        if not ant:
            continue
        # De-duplicate on the disease, not the antigen: OPV and IPV both cover
        # polio, so a child missing both is exposed to polio once.
        label = ANTIGEN_PROTECTS.get(ant, ant)
        if label not in seen:
            seen.add(label)
            out.append(label)
    return out


# Per-dose schedule (code, name, antigen_code, dose_number, recommended_age_days).
# Matches the RHU's own registry column layout exactly (see data/raw/*.csv):
# BCG, Hepa B-BD, DPT-Hib-HepB x3, OPV x3, PCV x3, IPV x2, MMR x2.
VACCINE_SCHEDULE = [
    ("BCG", "BCG", "BCG", 1, 0),
    ("HEPA-BD", "Hepa B-BD", "HEPB", 1, 0),
    ("DPT1", "DPT-Hib-HepB 1st Dose", "DPT", 1, 42),
    ("OPV1", "OPV 1st Dose", "OPV", 1, 42),
    ("PCV1", "PCV 1st Dose", "PCV", 1, 42),
    ("DPT2", "DPT-Hib-HepB 2nd Dose", "DPT", 2, 70),
    ("OPV2", "OPV 2nd Dose", "OPV", 2, 70),
    ("PCV2", "PCV 2nd Dose", "PCV", 2, 70),
    ("DPT3", "DPT-Hib-HepB 3rd Dose", "DPT", 3, 98),
    ("OPV3", "OPV 3rd Dose", "OPV", 3, 98),
    ("PCV3", "PCV 3rd Dose", "PCV", 3, 98),
    ("IPV1", "IPV 1st Dose", "IPV", 1, 98),
    ("IPV2", "IPV 2nd Dose", "IPV", 2, 270),
    ("MMR1", "MMR 1st Dose", "MMR", 1, 270),
    ("MMR2", "MMR 2nd Dose", "MMR", 2, 365),
]
VACCINE_CODES = [row[0] for row in VACCINE_SCHEDULE]

GRACE_PERIOD_DAYS = 30

# Score at or above which a child is flagged At-Risk by the prototype scorer.
RISK_THRESHOLD = 0.50


ROLE_ADMIN, ROLE_RHU, ROLE_BHW = "admin", "rhu", "bhw"
REQUEST_PENDING, REQUEST_APPROVED, REQUEST_REJECTED, REQUEST_FULFILLED = (
    "pending", "approved", "rejected", "fulfilled",
)

# ---------------------------------------------------------------------------
# Obtain & Scrub: real RHU registry CSV ingestion and cleaning.
# ---------------------------------------------------------------------------
# The RHU's raw exports use a two-row header (grouped dose columns), mixed
# date formats ("2024-01-08" and "5/8/24"), sentinel non-dates ("NULL",
# "MOVE OUT" for relocated children, blank), missing child names, missing
# sequential IDs, and inconsistent barangay spelling/sub-locality suffixes
# (e.g. "Guibal (Centro)", "Centro, Guibal", "Guibal Pandes"). This section
# normalizes all of that into clean Python records.

_DATE_FORMATS = ["%Y-%m-%d", "%m/%d/%y", "%m/%d/%Y"]
_MISSING_SENTINELS = {"", "null", "none", "move out", "n/a", "na"}
_FIRST_DATA_COL = 6  # ID, RegDate, DOB, Name, Sex, Barangay come before the dose columns
_DOSE_COL_ORDER = [
    "BCG", "HEPA-BD", "DPT1", "DPT2", "DPT3", "OPV1", "OPV2", "OPV3",
    "PCV1", "PCV2", "PCV3", "IPV1", "IPV2", "VITAMIN_A", "MNP", "MMR1", "MMR2",
]


def _clean_date(value):
    if value is None:
        return None
    value = str(value).strip()
    if value.lower() in _MISSING_SENTINELS:
        return None
    for fmt in _DATE_FORMATS:
        try:
            d = datetime.strptime(value, fmt).date()
            # 2-digit years: assume 2000s
            if fmt == "%m/%d/%y" and d.year < 2000:
                d = d.replace(year=d.year + 100)
            return d
        except ValueError:
            continue
    return None


def _clean_sex(value):
    v = (value or "").strip().lower()
    if v.startswith("m"):
        return "M"
    if v.startswith("f"):
        return "F"
    return "M"  # default; real data never leaves this blank in practice


def _clean_barangay(raw_value, file_scope_barangay):
    """The RHU's own per-barangay ledgers occasionally record a
    purok/sitio suffix or a neighboring locality name in the Barangay
    column (e.g. "Guibal (Centro)", "Nambalangan"). Since each raw file
    IS that barangay's official registry, every row is canonicalized to
    the file's own barangay -- the raw value is preserved in the
    address field for traceability rather than silently discarded.
    """
    return file_scope_barangay


def _clean_name(raw_value, fallback_id, barangay):
    v = (raw_value or "").strip()
    if not v or v.upper() == "NULL":
        return f"Unnamed Child #{fallback_id} ({barangay})"
    # Registry format is "Last, First Middle" -> reformat to "First Middle Last"
    if "," in v:
        last, _, first = v.partition(",")
        v = f"{first.strip()} {last.strip()}".strip()
    return v.title()


def load_registry_csv(path, barangay_name):
    """Parses one RHU registry export (2-row header, 23 columns) into a
    list of cleaned child dicts: {full_name, sex, date_of_birth,
    barangay, address, date_registered, doses: {code: date|None},
    number_of_visits, moved_out}.
    """
    skipped_bad_dates = 0
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))

    records = []
    next_seq = 1
    for row in rows[2:]:  # skip the 2 header rows
        if not row or not any(c.strip() for c in row):
            continue  # blank trailer row
        row = row + [""] * (max(0, _FIRST_DATA_COL + len(_DOSE_COL_ORDER) + 1 - len(row)))

        raw_id = row[0].strip()
        reg_date = _clean_date(row[1])
        dob = _clean_date(row[2])
        if dob is None:
            continue  # date of birth is the one field we cannot proceed without

        # Data-quality rule: the registry contains at least one record whose date
        # of birth falls after its registration date (and after today), which is
        # a data-entry error at source. A child cannot be registered before being
        # born, so the implausible date of birth is discarded rather than carried
        # into the analytics, where it would skew age and coverage calculations.
        if dob > date.today() or (reg_date is not None and dob > reg_date):
            skipped_bad_dates += 1
            continue
        name = _clean_name(row[3], raw_id or next_seq, barangay_name)
        sex = _clean_sex(row[4])
        raw_barangay = row[5].strip()
        barangay = _clean_barangay(raw_barangay, barangay_name)

        doses = {}
        moved_out = False
        for i, code in enumerate(_DOSE_COL_ORDER):
            raw_val = row[_FIRST_DATA_COL + i] if _FIRST_DATA_COL + i < len(row) else ""
            if str(raw_val).strip().upper() == "MOVE OUT":
                moved_out = True
            doses[code] = _clean_date(raw_val)

        visits_raw = row[_FIRST_DATA_COL + len(_DOSE_COL_ORDER)] if len(row) > _FIRST_DATA_COL + len(_DOSE_COL_ORDER) else ""
        try:
            number_of_visits = int(str(visits_raw).strip())
        except (ValueError, TypeError):
            number_of_visits = sum(1 for v in doses.values() if v is not None)

        if reg_date is None:
            reg_date = min([d for d in [dob] + list(doses.values()) if d], default=dob)

        seq = int(raw_id) if raw_id.isdigit() else next_seq
        next_seq = max(next_seq, seq) + 1

        records.append({
            "source_id": seq,
            "full_name": name,
            "sex": sex,
            "date_of_birth": dob,
            "barangay": barangay,
            "raw_barangay_value": raw_barangay,
            "date_registered": reg_date,
            "doses": {c: doses[c] for c in VACCINE_CODES},
            "vitamin_a_date": doses.get("VITAMIN_A"),
            "mnp_given": doses.get("MNP") is not None,
            "number_of_visits": number_of_visits,
            "moved_out": moved_out,
        })
    if skipped_bad_dates:
        print(f"  {path.name if hasattr(path, 'name') else path}: skipped "
              f"{skipped_bad_dates} record(s) with an implausible date of birth")
    return records


def load_and_merge_real_registries(raw_dir=RAW_DATA_DIR):
    """Loads every *.csv under data/raw/, inferring each file's barangay
    from its filename (Data_SJ_<Barangay>...), and returns one merged list
    of cleaned child records across all real, RHU-provided registries.
    """
    merged = []
    for path in sorted(glob.glob(os.path.join(raw_dir, "*.csv"))):
        base = os.path.splitext(os.path.basename(path))[0]
        # Underscores stand in for spaces, so the whole tail is the barangay:
        # "Data_SJ_San_Guillermo" -> "San Guillermo". Matching on letters alone
        # would stop at the first underscore and resolve "San_Guillermo" to the
        # wrong barangay.
        m = re.search(r"Data_SJ_+(.+)$", base)
        barangay_key = (m.group(1) if m else base).replace("_", " ").strip()

        def _norm(v):
            return re.sub(r"[^a-z]", "", v.lower())

        barangay_name = next(
            (b for b in BARANGAYS if _norm(b) == _norm(barangay_key)),
            next((b for b in BARANGAYS if _norm(b).startswith(_norm(barangay_key))), barangay_key),
        )
        merged.extend(load_registry_csv(path, barangay_name))
    return merged


# ---------------------------------------------------------------------------
# Feature engineering (Explore) -- shared by training and inference.
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = [
    "sex_male", "age_days", "registration_delay_days", "barangay_risk_rate",
    "doses_due", "doses_completed", "doses_delayed", "doses_missed",
    "completion_rate", "avg_delay_days", "max_delay_days", "days_since_last_dose",
]

# How each engineered feature is expected to behave once a classifier is
# trained. Direction says which way the feature pushes the prediction; the note
# explains why. Shown in the interface so the feature set can be read without
# the source.
FEATURE_MODEL_ROLE = {
    "weighted_doses_missed": ("Raises risk",
        "Direct evidence of drop-off. Expected to carry the most weight in any fitted model."),
    "completion_rate": ("Lowers risk",
        "Share of due doses received. A high rate is the clearest signal a child is on schedule."),
    "days_since_last_dose": ("Raises risk",
        "Counted only while doses remain outstanding, so completing the schedule is not penalised."),
    "avg_delay_days": ("Raises risk",
        "Habitual lateness often precedes stopping altogether."),
    "doses_delayed": ("Raises risk",
        "Counts how often a dose ran past the grace period, separately from how late."),
    "registration_delay_days": ("Raises risk",
        "A late first contact with the RHU is an early indicator of weak engagement."),
    "doses_due": ("Context",
        "Denominator for completion. Also separates a young child from an older one with the same counts."),
    "doses_completed": ("Lowers risk",
        "Absolute count behind the completion rate."),
    "max_delay_days": ("Raises risk",
        "Distinguishes one long lapse from consistent small delays."),
    "age_days": ("Context",
        "Older children have more doses due, so age conditions every other count."),
    "barangay_risk_rate": ("Raises risk",
        "Historical non-completion in the child's barangay. Captures access and distance effects "
        "the child-level features cannot."),
    "sex_male": ("Neutral",
        "Collected for completeness. No association is assumed; the model decides whether it matters."),
}

# How much a missed dose of each vaccine matters.
#
# Not every missed dose carries the same consequence. The weight is based on the
# herd immunity threshold for the disease the vaccine prevents, because that is
# the published figure that directly expresses how much population protection a
# missed dose costs. Diseases with a higher threshold tolerate fewer unprotected
# children.
#
#   measles      R0 12-18, threshold about 95%      (the DOH target is set by this)
#   pertussis    R0 12-17, threshold about 92-94%
#   polio        R0 5-7,   threshold about 80-86%
#   diphtheria   R0 6-7,   threshold about 85%
#
# BCG, the hepatitis B birth dose and PCV have no herd immunity threshold in the
# same sense: they mainly protect the individual child rather than blocking
# community transmission. They are weighted on the severity of the illness in an
# infant instead, which is why they sit in a single lower band rather than being
# given invented decimals.
#
# Three bands rather than seven precise numbers: the evidence supports an
# ordering, not two-decimal precision.
ANTIGEN_SEVERITY = {
    "MMR":  1.00,   # measles, threshold ~95%
    "DPT":  1.00,   # pentavalent, carries pertussis at ~92-94%
    "OPV":  0.85,   # polio, threshold ~80-86%
    "IPV":  0.85,   # polio
    "BCG":  0.70,   # severe childhood TB; individual protection
    "HEPB": 0.70,   # perinatal transmission; individual protection
    "PCV":  0.70,   # pneumococcal disease; individual protection
}

SEVERITY_BANDS = [
    (1.00, "Highest", "Measles and pertussis need about 95% coverage to stop transmission"),
    (0.85, "High", "Polio needs about 80 to 86% coverage"),
    (0.70, "Standard", "Protects the individual child rather than blocking transmission"),
]


def severity_of(antigen_code):
    return ANTIGEN_SEVERITY.get(antigen_code, 0.70)


def weighted_missed(missed_codes):
    """Missed doses counted by consequence rather than as a plain tally.

    Three missed PCV doses and three missed MMR doses are not the same thing,
    so each missed dose contributes its antigen's severity weight."""
    antigen_of = {code: ant for code, _n, ant, _d, _r in VACCINE_SCHEDULE}
    return sum(severity_of(antigen_of.get(c, "")) for c in missed_codes)


FEATURE_LABELS = {
    "sex_male": "Sex (Male)",
    "age_days": "Child's Age",
    "registration_delay_days": "Delay in RHU Registration After Birth",
    "barangay_risk_rate": "Barangay Historical Risk Rate",
    "doses_due": "Number of Doses Due So Far",
    "doses_completed": "Doses Completed On Time",
    "doses_delayed": "Doses Administered Late",
    "doses_missed": "Doses Missed (Past Grace Period)",
    "weighted_doses_missed": "Doses Missed, Weighted by Vaccine",
    "completion_rate": "Overall Completion Rate",
    "avg_delay_days": "Average Delay Per Dose",
    "max_delay_days": "Longest Single Dose Delay",
    "days_since_last_dose": "Days Since Last Vaccination Visit",
}


def compute_features(date_of_birth, date_registered, sex, dose_records, assessment_date, barangay_risk_rate):
    """dose_records: list of {recommended_age_days, date_administered}."""
    age_days = (assessment_date - date_of_birth).days
    registration_delay = max(0, (date_registered - date_of_birth).days)

    due = [r for r in dose_records if r["recommended_age_days"] <= age_days]
    completed, delayed, missed, delays = [], [], [], []
    last_dose_day = None

    for r in due:
        administered = r.get("date_administered")
        if administered is None:
            if age_days - r["recommended_age_days"] > GRACE_PERIOD_DAYS:
                missed.append(r)
            continue
        admin_offset = (administered - date_of_birth).days
        if administered > assessment_date:
            continue
        delay = admin_offset - r["recommended_age_days"]
        delays.append(max(0, delay))
        # A late dose is still received: it counts toward completion and is
        # separately recorded as delayed. Excluding it from `completed` made
        # completion_rate punish children who finished the schedule late.
        completed.append(r)
        if delay > GRACE_PERIOD_DAYS:
            delayed.append(r)
        if last_dose_day is None or admin_offset > last_dose_day:
            last_dose_day = admin_offset

    doses_due, doses_completed, doses_delayed, doses_missed = len(due), len(completed), len(delayed), len(missed)
    # Missed doses counted by consequence: a missed MMR weighs more than a
    # missed PCV, because measles needs far higher coverage to stay contained.
    weighted_missed_doses = weighted_missed([r["code"] for r in missed if r.get("code")])
    completion_rate = (doses_completed / doses_due) if doses_due else 1.0
    avg_delay = (sum(delays) / len(delays)) if delays else 0.0
    max_delay = max(delays) if delays else 0.0
    # Only meaningful while the child still has doses outstanding. A child who
    # completed the schedule has no reason to return, so a long gap is expected
    # rather than a warning sign - counting it flagged finished children.
    outstanding = doses_due - len(completed)
    if outstanding <= 0:
        days_since_last = 0.0
    else:
        days_since_last = (age_days - last_dose_day) if last_dose_day is not None else age_days

    return {
        "sex_male": 1 if sex == "M" else 0,
        "age_days": age_days,
        "registration_delay_days": registration_delay,
        "barangay_risk_rate": barangay_risk_rate,
        "doses_due": doses_due,
        "doses_completed": doses_completed,
        "doses_delayed": doses_delayed,
        "doses_missed": doses_missed,
        "weighted_doses_missed": round(weighted_missed_doses, 2),
        "completion_rate": completion_rate,
        "avg_delay_days": avg_delay,
        "max_delay_days": max_delay,
        "days_since_last_dose": days_since_last,
    }


def final_outcome_label(date_of_birth, dose_records, window_days=365):
    due = [r for r in dose_records if r["recommended_age_days"] <= window_days]
    if not due:
        return "Not At-Risk"
    completed = 0
    for r in due:
        admin = r.get("date_administered")
        if admin is None:
            continue
        if (admin - date_of_birth).days - r["recommended_age_days"] <= GRACE_PERIOD_DAYS:
            completed += 1
    completion_rate = completed / len(due)
    missed = sum(1 for r in due if r.get("date_administered") is None)
    return "At-Risk" if (completion_rate < 0.8 or missed >= 2) else "Not At-Risk"


def _dose_records_for(doses_dict):
    return [
        {"code": code, "recommended_age_days": rec_days, "date_administered": doses_dict.get(code)}
        for code, name, antigen, dose_no, rec_days in VACCINE_SCHEDULE
    ]


# ---------------------------------------------------------------------------
# Synthetic data simulator -- used to (a) fill the 17 barangays without a
# real registry export, and (b) augment the ML training corpus alongside
# the real registry data.
# ---------------------------------------------------------------------------

FIRST_NAMES_M = ["Juan", "Miguel", "Jose", "Antonio", "Carlos", "Rafael", "Gabriel", "Diego",
                 "Marco", "Angelo", "Vincent", "Emmanuel", "Joshua", "Nathaniel", "Elijah"]
FIRST_NAMES_F = ["Maria", "Sofia", "Isabella", "Andrea", "Camille", "Angela", "Bianca", "Kyla",
                  "Daniella", "Trisha", "Reign", "Althea", "Janelle", "Precious", "Faith"]
LAST_NAMES = ["Reyes", "Santos", "Cruz", "Bautista", "Garcia", "Mendoza", "Torres", "Ramos",
              "Flores", "Villanueva", "Castillo", "Aquino", "De Guzman", "Del Rosario",
              "Fernandez", "Gonzales", "Pascual", "Rivera", "Salazar", "Navarro"]

_rng_seed = random.Random(42)
BARANGAY_REMOTENESS = {b: round(_rng_seed.uniform(0.05, 0.35), 2) for b in BARANGAYS}

# Baseline at-risk rate per barangay, derived from BARANGAY_REMOTENESS. Used as
# a feature input; the trained model will instead learn this from history.
BARANGAY_BASE_RISK = {b: round(0.10 + BARANGAY_REMOTENESS.get(b, 0.2) * 0.45, 3)
                      for b in BARANGAYS}


def _simulate_dose_history(dob, propensity, rng):
    records = []
    for code, name, antigen, dose_no, rec_days in VACCINE_SCHEDULE:
        roll = rng.random()
        if roll < propensity * 0.22:
            administered = None
        else:
            delay = max(0, int(rng.gauss(mu=propensity * 16, sigma=7)))
            administered = dob + timedelta(days=rec_days + delay)
            if administered > date.today():
                administered = None
        records.append({"code": code, "recommended_age_days": rec_days, "date_administered": administered})
    return records


def _propensity_for(barangay, rng):
    remoteness = BARANGAY_REMOTENESS.get(barangay, 0.2)
    household = rng.uniform(0.0, 0.35)
    return max(0.02, min(0.9, remoteness * 0.55 + household * 0.55 + rng.uniform(-0.04, 0.04)))


def generate_synthetic_children(barangays, n_per_barangay=(6, 14), seed=99):
    rng = random.Random(seed)
    children = []
    for barangay in barangays:
        count = rng.randint(*n_per_barangay)
        for _ in range(count):
            sex = rng.choice(["M", "F"])
            first = rng.choice(FIRST_NAMES_M if sex == "M" else FIRST_NAMES_F)
            last = rng.choice(LAST_NAMES)
            dob = date.today() - timedelta(days=rng.randint(10, 364))
            propensity = _propensity_for(barangay, rng)
            reg_delay = int(max(0, rng.gauss(mu=propensity * 20, sigma=5)))
            date_registered = min(date.today(), dob + timedelta(days=reg_delay))
            history = _simulate_dose_history(dob, propensity, rng)
            children.append({
                "full_name": f"{first} {last}", "sex": sex, "date_of_birth": dob,
                "barangay": barangay, "date_registered": date_registered,
                "guardian_name": f"{rng.choice(FIRST_NAMES_F)} {last}",
                "guardian_contact": f"09{rng.randint(100000000, 999999999)}",
                "address": f"Purok {rng.randint(1, 7)}, {barangay}",
                "doses": {r["code"]: r["date_administered"] for r in history},
                "number_of_visits": sum(1 for r in history if r["date_administered"]),
                "moved_out": False,
            })
    return children


def generate_training_dataset(n_children=3000, seed=7, real_children=None):
    """Builds the labeled feature dataset for model training: early-warning
    features (snapshot at a random in-progress age) -> final 12-month outcome.
    Combines synthetic children with any real registry children supplied
    (data augmentation - the real sample alone is too small to train on
    reliably, per literature on small clinical datasets).
    """
    rng = random.Random(seed)
    rows = []

    def _add_row(dob, date_registered, sex, doses_dict, barangay):
        dose_records = _dose_records_for(doses_dict)
        label = final_outcome_label(dob, dose_records, window_days=365)
        assessment_age = rng.randint(60, 300)
        assessment_date = min(date.today(), dob + timedelta(days=assessment_age))
        feats = compute_features(dob, date_registered, sex, dose_records, assessment_date, barangay_risk_rate=None)
        feats["barangay"] = barangay
        feats["label"] = label
        rows.append(feats)

    for _ in range(n_children):
        barangay = rng.choice(BARANGAYS)
        sex = rng.choice(["M", "F"])
        dob = date(2025, 1, 1) - timedelta(days=rng.randint(0, 365 * 2))
        propensity = _propensity_for(barangay, rng)
        reg_delay = int(max(0, rng.gauss(mu=propensity * 20, sigma=5)))
        date_registered = dob + timedelta(days=reg_delay)
        history = _simulate_dose_history(dob, propensity, rng)
        _add_row(dob, date_registered, sex, {r["code"]: r["date_administered"] for r in history}, barangay)

    for c in (real_children or []):
        _add_row(c["date_of_birth"], c["date_registered"], c["sex"], c["doses"], c["barangay"])

    barangay_counts = defaultdict(lambda: [0, 0])
    for r in rows:
        barangay_counts[r["barangay"]][1] += 1
        if r["label"] == "At-Risk":
            barangay_counts[r["barangay"]][0] += 1
    barangay_risk_rate = {b: (c[0] / c[1] if c[1] else 0.15) for b, c in barangay_counts.items()}
    for b in BARANGAYS:
        barangay_risk_rate.setdefault(b, 0.15)
    for r in rows:
        r["barangay_risk_rate"] = barangay_risk_rate[r["barangay"]]

    return rows, barangay_risk_rate


# ---------------------------------------------------------------------------
# Model (train) and iNterpret (predict + explain).
# ---------------------------------------------------------------------------

def train_and_save_model(n_synthetic=3000, seed=7, use_real_data=True):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier

    real_children = load_and_merge_real_registries() if use_real_data else []
    if real_children:
        print(f"Loaded {len(real_children)} real children from data/raw/*.csv for training augmentation.")

    rows, barangay_risk_rate = generate_training_dataset(n_synthetic, seed, real_children)
    import numpy as np  # lazy: training only
    X = np.array([[r[c] for c in FEATURE_COLUMNS] for r in rows], dtype=float)
    y = np.array([1 if r["label"] == "At-Risk" else 0 for r in rows], dtype=int)
    print(f"Training dataset: {len(y)} rows ({len(real_children)} real + {n_synthetic} synthetic), "
          f"{y.sum()} At-Risk ({100 * y.mean():.1f}%)")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

    def _eval(name, model, Xte):
        proba = model.predict_proba(Xte)[:, 1]
        pred = (proba >= 0.5).astype(int)
        m = {"accuracy": accuracy_score(y_test, pred), "precision": precision_score(y_test, pred, zero_division=0),
             "recall": recall_score(y_test, pred, zero_division=0), "f1": f1_score(y_test, pred, zero_division=0),
             "roc_auc": roc_auc_score(y_test, proba)}
        print(f"  {name:22s} " + " ".join(f"{k}={v:.3f}" for k, v in m.items()))
        return m

    candidates = {}
    lr = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X_train_s, y_train)
    candidates["LogisticRegression"] = (lr, _eval("LogisticRegression", lr, X_test_s))

    rf = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5,
                                 class_weight="balanced", random_state=seed).fit(X_train, y_train)
    candidates["RandomForest"] = (rf, _eval("RandomForest", rf, X_test))

    pos_weight = (len(y_train) - y_train.sum()) / max(1, y_train.sum())
    xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.08, subsample=0.9,
                         colsample_bytree=0.9, scale_pos_weight=pos_weight, eval_metric="logloss",
                         random_state=seed).fit(X_train, y_train)
    candidates["XGBoost"] = (xgb, _eval("XGBoost", xgb, X_test))

    best_name = max(candidates, key=lambda k: (candidates[k][1]["recall"], candidates[k][1]["roc_auc"]))
    best_model, best_metrics = candidates[best_name]
    all_metrics = {name: metrics for name, (model, metrics) in candidates.items()}
    print(f"Selected best model -> {best_name}")

    needs_scaler = best_name == "LogisticRegression"
    cv_scores = cross_val_score(best_model, X_train_s if needs_scaler else X_train, y_train, cv=5, scoring="recall")
    print(f"  5-fold CV recall: mean={cv_scores.mean():.3f} std={cv_scores.std():.3f}")

    feature_importance = (
        dict(zip(FEATURE_COLUMNS, best_model.feature_importances_.tolist()))
        if hasattr(best_model, "feature_importances_") else None
    )

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    artifact = {
        "model": best_model, "model_name": best_name, "model_version": "v2-real+synthetic",
        "scaler": scaler if needs_scaler else None, "needs_scaler": needs_scaler,
        "feature_columns": FEATURE_COLUMNS, "barangay_risk_rate": barangay_risk_rate,
        "metrics": best_metrics, "all_metrics": all_metrics, "feature_importance": feature_importance,
        "real_data_children": len(real_children),
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"Saved model -> {MODEL_PATH}")
    return artifact


_artifact_cache = None


def model_is_trained():
    """True once a model artifact has been trained and saved to disk."""
    return os.path.exists(MODEL_PATH)


def model_info():
    """What the UI should say about the prediction engine.

    In the pre-final prototype no model has been trained, so this reports the
    rule-based scorer honestly instead of inventing accuracy figures."""
    if model_is_trained():
        art = _load_artifact()
        return {
            "trained": True,
            "engine": art.get("model_name", "ML Classifier"),
            "version": art.get("model_version", "unknown"),
            "metrics": art.get("metrics", {}),
            "all_metrics": art.get("all_metrics", {}),
        }
    return {
        "trained": False,
        "engine": SCORER_NAME,
        "version": SCORER_VERSION,
        "metrics": {},
        "all_metrics": {},
    }


def _load_artifact():
    """Only reachable once a model has been trained (final defense)."""
    global _artifact_cache
    if _artifact_cache is None:
        import joblib  # lazy: not needed by the prototype
        _artifact_cache = joblib.load(MODEL_PATH)
    return _artifact_cache


_REFERENCE_NOT_AT_RISK = {
    "doses_missed": 0.2, "doses_delayed": 0.5, "avg_delay_days": 8.0, "max_delay_days": 15.0,
    "completion_rate": 0.95, "registration_delay_days": 5.0, "days_since_last_dose": 30.0,
    "barangay_risk_rate": 0.15,
}


def _describe(key, feats):
    if key == "doses_missed":
        return f"{int(feats['doses_missed'])} dose(s) missed beyond the grace period"
    if key == "doses_delayed":
        return f"{int(feats['doses_delayed'])} dose(s) administered late"
    if key == "avg_delay_days":
        return f"Average delay of {feats['avg_delay_days']:.0f} days per dose"
    if key == "max_delay_days":
        return f"Longest single delay: {feats['max_delay_days']:.0f} days"
    if key == "completion_rate":
        return f"Only {feats['completion_rate']*100:.0f}% of due doses completed on time"
    if key == "registration_delay_days":
        return f"Registered {feats['registration_delay_days']:.0f} days after birth"
    if key == "days_since_last_dose":
        return f"{feats['days_since_last_dose']:.0f} days since last vaccination visit"
    if key == "barangay_risk_rate":
        return f"Barangay historical at-risk rate: {feats['barangay_risk_rate']*100:.0f}%"
    return ""


def _explain(feats, artifact, top_n=3):
    importance = artifact.get("feature_importance") or {}
    scored = []
    for key, ref in _REFERENCE_NOT_AT_RISK.items():
        val = feats.get(key, 0)
        deviation = max(0.0, (ref - val) if key == "completion_rate" else (val - ref))
        scored.append((key, deviation * (importance.get(key, 0.1) + 0.05)))
    scored.sort(key=lambda x: x[1], reverse=True)
    factors = [{"factor": FEATURE_LABELS.get(k, k), "detail": _describe(k, feats)} for k, s in scored[:top_n] if s > 0]
    return factors or [{"factor": "On-schedule vaccination history", "detail": "No significant risk indicators detected."}]


# ---------------------------------------------------------------------------
# Rule-based risk scorer (pre-final defense prototype)
#
# The trained classifier is deliberately not part of this build. Risk flags are
# produced by the transparent weighted rules below, drawn from the same feature
# set the model will eventually consume, so the whole system is demonstrable
# end-to-end without any ML dependency.
#
# Every weight is visible and explainable, which is the point: nothing here
# claims to be a trained model, and no accuracy figures are reported for it.
# ---------------------------------------------------------------------------

SCORER_NAME = "Rule-Based Scorer"
SCORER_VERSION = "rules-v3-weighted"

# Each rule contributes to a 0..1 risk score. Weights sum to 1.0.
SCORING_RULES = {
    "weighted_doses_missed":             0.34,  # strongest single signal of drop-off
    "completion_rate":          0.24,
    "days_since_last_dose":     0.16,
    "avg_delay_days":           0.12,
    "doses_delayed":            0.08,
    "registration_delay_days":  0.06,
}

# Value at which a rule is considered fully "triggered" (contributes its whole
# weight). Below this the contribution scales linearly.
RULE_SATURATION = {
    "weighted_doses_missed": 3.0,
    "days_since_last_dose": 180.0,
    "avg_delay_days": 45.0,
    "doses_delayed": 4.0,
    "registration_delay_days": 60.0,
}


def _rule_contributions(feats):
    """{rule: (contribution, share_of_score)} for one child."""
    out = {}
    for key, weight in SCORING_RULES.items():
        if key == "completion_rate":
            # Inverted: a LOW completion rate raises risk.
            intensity = 1.0 - float(feats.get("completion_rate", 1.0))
        else:
            cap = RULE_SATURATION[key]
            intensity = min(1.0, float(feats.get(key, 0)) / cap) if cap else 0.0
        out[key] = max(0.0, min(1.0, intensity)) * weight
    return out


def score_child(feats):
    """Risk score in 0..1 from the weighted rules above."""
    return round(min(1.0, sum(_rule_contributions(feats).values())), 4)


def _explain_rules(feats, top_n=3):
    """The rules that contributed most to this child's score."""
    contribs = sorted(_rule_contributions(feats).items(), key=lambda kv: kv[1], reverse=True)
    factors = [
        {"factor": FEATURE_LABELS.get(k, k), "detail": _describe(k, feats)}
        for k, c in contribs[:top_n] if c > 0.01
    ]
    return factors or [{"factor": "On-schedule vaccination history",
                        "detail": "No significant risk indicators detected."}]


def predict_for_child(date_of_birth, date_registered, sex, doses_dict, barangay_name):
    """Public scoring entrypoint used by app.py and seed.py.

    doses_dict maps VACCINE_CODES -> date|None for one child. Returns the same
    shape the trained model will return, so swapping the ML model back in for
    the final defense requires no changes in app.py."""
    dose_records = _dose_records_for(doses_dict)
    feats = compute_features(
        date_of_birth, date_registered, sex, dose_records, date.today(),
        barangay_risk_rate=BARANGAY_BASE_RISK.get(barangay_name, 0.15),
    )
    score = score_child(feats)
    return {
        "label": "At-Risk" if score >= RISK_THRESHOLD else "Not At-Risk",
        "probability": score,
        "model_version": SCORER_VERSION,
        "top_factors": _explain_rules(feats),
        "features": feats,
    }


# ---------------------------------------------------------------------------
# Vaccine allocation when stock is short
# ---------------------------------------------------------------------------
# When pending requests exceed what the RHU holds, someone has to decide who
# gets what. That is a clinical and political decision, so the system does not
# impose a single rule: it computes three and shows the effect of each, leaving
# the choice with RHU personnel.
#
#   proportional      every barangay receives the same share of what it asked
#                     for. Fair between requesters, blind to need.
#   priority          urgency set by the requesting health worker is served
#                     first, then the remainder is split proportionally.
#   coverage          barangays furthest below the coverage target are served
#                     first. Closes gaps fastest.
#
# No allocation exceeds what a barangay requested: a barangay should not receive
# more than it asked for and can store.

PRIORITY_RANK = {"Urgent": 0, "High": 1, "Normal": 2}

ALLOCATION_METHODS = {
    "priority": "Priority first, then proportional",
    "proportional": "Proportional to request",
    "coverage": "Lowest coverage first",
}


def _largest_remainder(weights, available):
    """Split `available` whole vials across weights without losing a vial.

    Plain rounding of proportional shares either overshoots or leaves vials
    unassigned, so the fractional parts decide who gets the remainder."""
    total_w = sum(weights.values())
    if total_w <= 0 or available <= 0:
        return {k: 0 for k in weights}
    exact = {k: available * w / total_w for k, w in weights.items()}
    out = {k: int(v) for k, v in exact.items()}
    left = available - sum(out.values())
    for k, _ in sorted(exact.items(), key=lambda kv: -(kv[1] - int(kv[1]))):
        if left <= 0:
            break
        out[k] += 1
        left -= 1
    return out


def allocate_stock(requests, available, method="priority", coverage_by_bgy=None):
    """Suggested allocation of `available` vials across pending requests.

    requests: [{"id", "barangay", "requested", "priority"}]
    Returns a row per request with the suggested quantity and what it leaves
    short, plus whether the requests can be met in full."""
    total_requested = sum(r["requested"] for r in requests)
    alloc = {r["id"]: 0 for r in requests}

    if total_requested <= available:
        # No shortfall: everyone receives what they asked for.
        alloc = {r["id"]: r["requested"] for r in requests}
    elif method == "proportional":
        weights = {r["id"]: r["requested"] for r in requests}
        alloc = _largest_remainder(weights, available)
    elif method == "coverage":
        cov = coverage_by_bgy or {}
        # Lowest coverage served first, in full where stock allows.
        left = available
        for r in sorted(requests, key=lambda r: cov.get(r["barangay"], 100)):
            take = min(r["requested"], left)
            alloc[r["id"]] = take
            left -= take
    else:  # priority
        left = available
        for r in sorted(requests, key=lambda r: (PRIORITY_RANK.get(r["priority"], 9),
                                                 -r["requested"])):
            take = min(r["requested"], left)
            alloc[r["id"]] = take
            left -= take

    rows = []
    for r in requests:
        given = min(alloc.get(r["id"], 0), r["requested"])
        rows.append({**r, "allocated": given, "short": r["requested"] - given,
                     "share": round(given / r["requested"] * 100) if r["requested"] else 0})
    return {
        "rows": rows,
        "available": available,
        "requested": total_requested,
        "allocated": sum(r["allocated"] for r in rows),
        "shortfall": max(0, total_requested - available),
        "can_meet_all": total_requested <= available,
    }


# ---------------------------------------------------------------------------
# Vaccine demand forecasting
# ---------------------------------------------------------------------------
# Demand for a future year has two parts, and only one of them is forecast:
#
#   Known    doses children already in the registry will become due for. A child
#            born in 2025 is due MMR 2nd in 2026. Computed exactly from dates of
#            birth against the schedule, no estimation involved.
#
#   Projected  doses for children not yet born or registered. This needs a birth
#              cohort estimate, which is where the uncertainty sits.
#
# The cohort is estimated from the mean of the last complete years rather than a
# fitted trend. A linear fit on this registry gives +40 births/year because the
# earliest year holds only a handful of records - a registry starting up, not a
# demographic fact - and projects numbers that are not credible.

FORECAST_YEARS = 3          # beyond this the cohort assumption dominates
COHORT_BASE_YEARS = 3       # years averaged for the cohort estimate
COHORT_MIN_ROWS = 20        # a year below this is registry start-up, not a cohort
COHORT_HIGH_MARGIN = 0.15   # upper bound of the range, as a share


def cohort_estimate(births_by_year, today_year):
    """Low and high estimate of annual births, with the years it is based on.

    Only complete years with a credible number of records are used. The current
    year is excluded because it is partial."""
    usable = {y: n for y, n in births_by_year.items()
              if y < today_year and n >= COHORT_MIN_ROWS}
    if not usable:
        return None
    years = sorted(usable)[-COHORT_BASE_YEARS:]
    mean = sum(usable[y] for y in years) / len(years)
    return {
        "low": round(mean),
        "high": round(mean * (1 + COHORT_HIGH_MARGIN)),
        "years": years,
        "counts": {y: usable[y] for y in years},
    }


def doses_due_in_year(date_of_birth, year):
    """Which dose codes fall due for this child during the given calendar year."""
    out = []
    for code, _name, _ant, _dose, rec_days in VACCINE_SCHEDULE:
        due = date_of_birth + timedelta(days=rec_days)
        if due.year == year:
            out.append(code)
    return out


def forecast_vaccine_demand(children_dobs, births_by_year, today=None,
                            years=FORECAST_YEARS):
    """Projected doses needed per antigen, per year.

    children_dobs is the dates of birth already in the registry. Returns one
    entry per forecast year with the known and projected components separated,
    so it is clear which part is measured and which is estimated."""
    today = today or date.today()
    antigen_of = {code: ant for code, _n, ant, _d, _r in VACCINE_SCHEDULE}
    doses_per_antigen = Counter(ant for _c, _n, ant, _d, _r in VACCINE_SCHEDULE)
    cohort = cohort_estimate(births_by_year, today.year)

    out = []
    for offset in range(1, years + 1):
        y = today.year + offset

        # Known: doses the existing children become due for in year y.
        known = Counter()
        for dob in children_dobs:
            for code in doses_due_in_year(dob, y):
                known[antigen_of[code]] += 1

        # Projected: a new cohort born in year y needs the doses whose
        # recommended age falls inside that same year. A child born in December
        # will not reach its 12-month dose until the following year, so roughly
        # the first-year doses land in the birth year. Held simple and stated.
        rows = []
        for antigen_code, antigen_doses in doses_per_antigen.items():
            k = known.get(antigen_code, 0)
            p_low = round(cohort["low"] * antigen_doses) if cohort else 0
            p_high = round(cohort["high"] * antigen_doses) if cohort else 0
            rows.append({
                "antigen": antigen_code,
                "doses_per_child": antigen_doses,
                "known": k,
                "projected_low": p_low,
                "projected_high": p_high,
                "total_low": k + p_low,
                "total_high": k + p_high,
            })
        rows.sort(key=lambda r: -r["total_high"])
        out.append({
            "year": y,
            "rows": rows,
            "total_low": sum(r["total_low"] for r in rows),
            "total_high": sum(r["total_high"] for r in rows),
            "known_total": sum(r["known"] for r in rows),
        })
    return {"cohort": cohort, "years": out}


def risk_tier(probability):
    if probability >= 0.70:
        return "High Risk"
    if probability >= 0.50:
        return "Medium Risk"
    return "Low Risk"


def risk_tier_color(tier):
    return {"High Risk": "red", "Medium Risk": "amber", "Low Risk": "green"}.get(tier, "green")


def serialize_factors(factors):
    return json.dumps(factors)


def deserialize_factors(text):
    return json.loads(text) if text else []