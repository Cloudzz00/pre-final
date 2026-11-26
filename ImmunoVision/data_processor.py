"""
data_processor.py
=================
All of ImmunoVision's data work, in the order of the OSEMN framework:

  Obtain     load_and_merge_real_registries(), load_registry_csv()
  Scrub      _clean_date(), _clean_sex(), _clean_name()
  Explore    compute_features()  - the 10 model inputs
             final_outcome_label() - the 12-month At-Risk label
  Model      generate_training_dataset(), train_and_save_model()
  iNterpret  predict_for_child() - risk score and the reasons behind it

Also here: the rule-based backup scorer, stock allocation and the vaccine
demand forecast used by the dashboards.

The machine-learning libraries (numpy, scikit-learn, xgboost, joblib) are only
imported inside the functions that need them.
"""
import csv
import glob
import hashlib
import json
import os
import random
import re
from collections import Counter
from datetime import date, datetime, timedelta

BASE_DIR = os.path.abspath(os.path.dirname(__file__))  # app folder
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")  # registry CSVs
MODEL_PATH = os.path.join(BASE_DIR, "data", "risk_model.joblib")  # saved model

# ===========================================================================
# 1. Reference data: barangays and the DOH immunization schedule
# ===========================================================================

BARANGAYS = [  # 19 barangays
    "Awai", "Bolo", "Capaoay", "Casibong", "Imelda",
    "Guibel", "Labney", "Magsaysay", "Lobong", "Macayug",
    "Bagong Pag-asa", "San Guillermo", "San Jose", "San Juan",
    "San Roque", "San Vicente", "Santa Cruz", "Santa Maria", "Santo Tomas",
]

VACCINE_ANTIGENS = [  # 7 routine vaccines (Inventory / Requests)
    ("BCG", "BCG"),
    ("HEPB", "Hepatitis B"),
    ("PENTA", "Pentavalent"),
    ("OPV", "OPV"),
    ("PCV", "PCV"),
    ("IPV", "IPV"),
    ("MMR", "MMR"),
]

# Usual doses per vial (WHO/DOH EPI). Only pre-fills the Add Stock form; each
# batch records its own value from the label, and the admin can change these.
DEFAULT_DOSES_PER_VIAL = {
    "BCG": 20, "HEPB": 10, "PENTA": 1, "OPV": 20, "PCV": 4, "IPV": 5, "MMR": 10, "ROTA": 1,
}

# Vaccines whose opened vials are discarded at the end of the session or after
# 6 hours (WHO multi-dose vial policy, 2014; Philippine NIP manual: BCG and MMR
# once reconstituted). Other opened vials may be kept up to 28 days if the
# policy's conditions are met. The admin can change this per vaccine.
DISCARD_AFTER_SESSION = {"BCG", "MMR"}

ANTIGEN_PROTECTS = {  # disease each vaccine prevents
    "BCG":   "Tuberculosis",
    "HEPB":  "Hepatitis B",
    "PENTA": "Diphtheria, pertussis, tetanus, Hib, hepatitis B",
    "OPV":   "Polio",
    "PCV":   "Pneumococcal disease (pneumonia, meningitis)",
    "IPV":   "Polio",
    "MMR":   "Measles, mumps, rubella",
}

# The 15 scheduled doses: (code, name, vaccine, dose number, due age in days).
VACCINE_SCHEDULE = [  # DOH schedule
    ("BCG", "BCG", "BCG", 1, 0),  # birth
    ("HEPA-BD", "Hepa B-BD", "HEPB", 1, 0),
    ("PENTA1", "Pentavalent 1st Dose", "PENTA", 1, 42),  # 6 weeks
    ("OPV1", "OPV 1st Dose", "OPV", 1, 42),
    ("PCV1", "PCV 1st Dose", "PCV", 1, 42),
    ("PENTA2", "Pentavalent 2nd Dose", "PENTA", 2, 70),  # 10 weeks
    ("OPV2", "OPV 2nd Dose", "OPV", 2, 70),
    ("PCV2", "PCV 2nd Dose", "PCV", 2, 70),
    ("PENTA3", "Pentavalent 3rd Dose", "PENTA", 3, 98),  # 14 weeks
    ("OPV3", "OPV 3rd Dose", "OPV", 3, 98),
    ("PCV3", "PCV 3rd Dose", "PCV", 3, 98),
    ("IPV1", "IPV 1st Dose", "IPV", 1, 98),
    ("IPV2", "IPV 2nd Dose", "IPV", 2, 270),  # 9 months
    ("MMR1", "MMR 1st Dose", "MMR", 1, 270),
    ("MMR2", "MMR 2nd Dose", "MMR", 2, 365),  # 12 months
]
VACCINE_CODES = [row[0] for row in VACCINE_SCHEDULE]  # dose codes

# Earliest a dose may be given, so recorded dates follow the DOH schedule.
# A dose's own minimum age (first doses of each series, and the 9-month
# doses) plus a minimum gap between doses of the same vaccine.
DOSE_MIN_AGE_DAYS = {"PENTA1": 42, "OPV1": 42, "PCV1": 42, "IPV1": 98,  # 6 / 14 weeks
                     "IPV2": 270, "MMR1": 270}  # 9 months
MIN_DOSE_GAP_DAYS = 28  # 4 weeks between doses of the same vaccine


def schedule_violations(dob, dates_by_code, check_codes):
    """Messages for dose dates that break the schedule, or [].

    dates_by_code holds every core dose date as it will be saved; only the
    doses in check_codes (newly entered or changed) are judged, so records
    imported with older problems never stop a child from being saved.
    A dose is refused if it is before its minimum age, before the previous
    dose of the same vaccine, or less than MIN_DOSE_GAP_DAYS after it (and
    likewise if a changed earlier dose would land too close to a later one)."""
    fmt = lambda d: d.strftime("%m/%d/%y")
    name = {code: n for code, n, *_ in VACCINE_SCHEDULE}
    out = []
    series = {}
    for code, _n, antigen, dose_no, _days in VACCINE_SCHEDULE:
        series.setdefault(antigen, []).append(code)
    for code in VACCINE_CODES:
        given = dates_by_code.get(code)
        if given is None or code not in check_codes or dob is None:
            continue
        min_age = DOSE_MIN_AGE_DAYS.get(code)
        if min_age and (given - dob).days < min_age:
            earliest = dob + timedelta(days=min_age)
            out.append(f"{name[code]} can't be recorded before {fmt(earliest)} "
                       f"(the child is too young for it on {fmt(given)}).")
    for codes in series.values():
        dated = [(c, dates_by_code.get(c)) for c in codes]
        for (prev, d1), (cur, d2) in zip(dated, dated[1:]):
            if d1 is None or d2 is None or not ({prev, cur} & set(check_codes)):
                continue
            if d2 < d1:
                out.append(f"{name[cur]} ({fmt(d2)}) can't be dated before {name[prev]} ({fmt(d1)}).")
            elif (d2 - d1).days < MIN_DOSE_GAP_DAYS:
                earliest = d1 + timedelta(days=MIN_DOSE_GAP_DAYS)
                out.append(f"{name[cur]} can't be recorded before {fmt(earliest)}, "
                           f"4 weeks after {name[prev]} ({fmt(d1)}).")
    return out
ANTIGEN_OF = {code: vaccine for code, _name, vaccine, _n, _age in VACCINE_SCHEDULE}  # dose -> vaccine

# BCG is no longer given after the first year; every other dose can be caught up.
CORE_CATCH_UP_DEADLINES = {"BCG": 365}  # BCG cutoff

# Optional vaccine: recorded only if given, never used by the risk model.
# The last value is the age after which it may no longer be given.
OPTIONAL_VACCINE_ANTIGENS = [("ROTA", "Rotavirus")]
OPTIONAL_VACCINE_SCHEDULE = [
    ("ROTA1", "Rotavirus 1st Dose", "ROTA", 1, 42, 104),  # must start by ~15 weeks
    ("ROTA2", "Rotavirus 2nd Dose", "ROTA", 2, 70, 240),  # series closes ~8 months
]

GRACE_PERIOD_DAYS = 30            # a dose is "on time" within 30 days of its due age
RISK_THRESHOLD = 0.50             # score 0.50 or more = At-Risk
CONTINUATION_WINDOW_DAYS = 365    # the model scores children up to 12 months old ...
ML_ASSESSMENT_FLOOR_DAYS = GRACE_PERIOD_DAYS  # ... and from 30 days old
TRAINING_AT_RISK_SHARE = 0.5      # training data balanced to 50% At-Risk / 50% Not At-Risk


def diseases_for_missed(missed_codes):  # missed diseases
    """Diseases the missed doses would have prevented, each listed once
    (e.g. missing OPV and IPV lists polio only once)."""
    out = []
    for code in missed_codes:
        vaccine = ANTIGEN_OF.get(code)
        if vaccine and ANTIGEN_PROTECTS.get(vaccine, vaccine) not in out:
            out.append(ANTIGEN_PROTECTS.get(vaccine, vaccine))
    return out


# ===========================================================================
# 2. Obtain & Scrub: read and clean the RHU registry CSV files
# ===========================================================================
# The raw files have a 2-row header, mixed date formats ("2024-01-08",
# "5/8/24"), text such as "NULL" or "MOVE OUT" in date columns, and blank names.

_DATE_FORMATS = ["%Y-%m-%d", "%m/%d/%y", "%m/%d/%Y"]  # date formats
_MISSING_SENTINELS = {"", "null", "none", "move out", "n/a", "na"}  # = no date
_FIRST_DATA_COL = 6  # ID, RegDate, DOB, Name, Sex, Barangay come first
_DOSE_COL_ORDER = [  # dose columns, in the CSV's order
    "BCG", "HEPA-BD", "PENTA1", "PENTA2", "PENTA3", "OPV1", "OPV2", "OPV3",
    "PCV1", "PCV2", "PCV3", "IPV1", "IPV2", "VITAMIN_A", "MNP", "MMR1", "MMR2",
]
_VISITS_COL = _FIRST_DATA_COL + len(_DOSE_COL_ORDER)  # "No. of visits" column


def _clean_date(value):  # clean date
    """A registry cell as a date, or None if it is blank or unreadable."""
    if value is None:
        return None
    value = str(value).strip()
    if value.lower() in _MISSING_SENTINELS:  # NULL, MOVE OUT, blank
        return None
    for fmt in _DATE_FORMATS:  # try each format
        try:
            d = datetime.strptime(value, fmt).date()
        except ValueError:
            continue
        if fmt == "%m/%d/%y" and d.year < 2000:  # "5/8/24" means 2024
            d = d.replace(year=d.year + 100)
        return d
    return None  # unreadable


def _clean_sex(value):  # clean sex
    v = (value or "").strip().lower()
    if v.startswith("f"):
        return "F"
    return "M"  # "m..." or blank (never blank in the real data)


def _clean_name(raw_value, fallback_id, barangay):  # clean name
    """"Last, First Middle" -> "First Middle Last"; blank -> a placeholder."""
    v = (raw_value or "").strip()
    if not v or v.upper() == "NULL":
        return f"{PLACEHOLDER_NAME_PREFIX}{fallback_id} ({barangay})"
    if "," in v:
        last, _, first = v.partition(",")
        v = f"{first.strip()} {last.strip()}".strip()
    return v.title()


def load_registry_csv(path, barangay_name):  # read one CSV
    """Read one barangay's registry file into a list of cleaned children."""
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))

    records, next_seq, skipped = [], 1, 0
    for row in rows[2:]:  # skip the 2 header rows
        if not any(c.strip() for c in row):  # blank row
            continue
        row = row + [""] * (_VISITS_COL + 1 - len(row))  # pad short rows

        raw_id = row[0].strip()
        reg_date = _clean_date(row[1])
        dob = _clean_date(row[2])
        if dob is None:  # cannot use a child without a birth date
            continue
        # A child cannot be born after registration or in the future: data-entry error.
        if dob > date.today() or (reg_date is not None and dob > reg_date):
            skipped += 1
            continue

        doses, moved_out = {}, False
        for i, code in enumerate(_DOSE_COL_ORDER):  # each dose column
            cell = row[_FIRST_DATA_COL + i]
            if cell.strip().upper() == "MOVE OUT":
                moved_out = True
            dose_date = _clean_date(cell)
            # A future date is a planned visit pencilled in, not a dose given.
            doses[code] = None if dose_date and dose_date > date.today() else dose_date

        try:
            number_of_visits = int(row[_VISITS_COL].strip())
        except ValueError:  # blank: count the doses instead
            number_of_visits = sum(1 for d in doses.values() if d)

        if reg_date is None:  # missing: earliest known date
            reg_date = min([d for d in [dob] + list(doses.values()) if d], default=dob)

        seq = int(raw_id) if raw_id.isdigit() else next_seq  # child number
        next_seq = max(next_seq, seq) + 1

        records.append({  # cleaned child
            "source_id": seq,
            "full_name": _clean_name(row[3], raw_id or seq, barangay_name),
            "sex": _clean_sex(row[4]),
            "date_of_birth": dob,
            # Each file is one barangay's registry, so every row belongs to it;
            # the raw Barangay cell (e.g. "Guibel (Centro)") is kept for reference.
            "barangay": barangay_name,
            "raw_barangay_value": row[5].strip(),
            "date_registered": reg_date,
            "doses": {c: doses[c] for c in VACCINE_CODES},
            "vitamin_a_date": doses.get("VITAMIN_A"),
            "mnp_given": doses.get("MNP") is not None,
            "number_of_visits": number_of_visits,
            "moved_out": moved_out,
        })
    if skipped:
        print(f"  {os.path.basename(path)}: skipped {skipped} record(s) with an implausible date of birth")
    return records


def _letters_only(text):
    return re.sub(r"[^a-z]", "", text.lower())


def load_and_merge_real_registries(raw_dir=RAW_DATA_DIR):  # read all CSVs
    """Read every data/raw/*.csv into one list of cleaned children (1,689).
    The barangay comes from the file name: "Data_SJ_San_Guillermo" -> "San Guillermo"."""
    merged = []
    for path in sorted(glob.glob(os.path.join(raw_dir, "*.csv"))):  # each file
        base = os.path.splitext(os.path.basename(path))[0]
        m = re.search(r"Data_SJ_+(.+)$", base)
        key = _letters_only((m.group(1) if m else base).replace("_", " ").strip())
        barangay = next((b for b in BARANGAYS if _letters_only(b) == key),  # exact match
                        next((b for b in BARANGAYS if _letters_only(b).startswith(key)), key))  # or prefix
        merged.extend(load_registry_csv(path, barangay))

    # Give children with no registry name a generated name (same as the database does).
    names = generated_child_names(
        [(c["full_name"], c["sex"]) for c in merged if is_placeholder_name(c["full_name"])],
        [c["full_name"] for c in merged if not is_placeholder_name(c["full_name"])],
    )
    for c in merged:
        c["full_name"] = names.get(c["full_name"], c["full_name"])
    return merged


PLACEHOLDER_NAME_PREFIX = "Unnamed Child #"


def is_placeholder_name(name):  # placeholder?
    return (name or "").startswith(PLACEHOLDER_NAME_PREFIX)


def generated_child_names(placeholders, taken_names):  # generate names
    """{placeholder: generated name} for (placeholder, sex) pairs. The same
    placeholder always gets the same name, and no name repeats one in use."""
    used = {(n or "").strip().lower() for n in taken_names}
    out = {}
    for key, sex in sorted(placeholders):  # each unnamed child
        firsts = FIRST_NAMES_F if sex == "F" else FIRST_NAMES_M
        attempt = 0
        while True:
            h = hashlib.sha256(f"{key}|{attempt}".encode()).digest()  # stable hash
            name = f"{firsts[h[0] % len(firsts)]} {LAST_NAMES[h[1] % len(LAST_NAMES)]}"
            if name.lower() not in used:
                break
            attempt += 1
        used.add(name.lower())
        out[key] = name
    return out


# ===========================================================================
# 3. Explore: the 10 model features and the 12-month label
# ===========================================================================
# Sex and barangay are recorded for each child but are not model inputs: sex
# had almost no weight, and neither should decide who is flagged.

FEATURE_COLUMNS = [  # 10 features
    "age_days", "registration_delay_days",
    "doses_due", "doses_completed", "doses_delayed", "doses_missed",
    "completion_rate", "avg_delay_days", "max_delay_days", "days_since_last_dose",
]

FEATURE_LABELS = {  # names shown to users
    "age_days": "Child's Age",
    "registration_delay_days": "Delay in RHU Registration After Birth",
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

FEATURE_MODEL_ROLE = {  # (effect on risk, why) - shown in the interface
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
}

# Weight of a missed dose, by how much coverage the disease needs to stop
# spreading (herd immunity threshold): measles ~95%, pertussis ~92-94%,
# polio ~80-86%. BCG, Hepa B and PCV mainly protect the child, so they share
# one lower band. Used by the rule-based scorer and the Vaccine Gaps page.
ANTIGEN_SEVERITY = {  # vaccine weights
    "MMR":   1.00,  # measles
    "PENTA": 1.00,  # includes pertussis
    "OPV":   0.85,  # polio
    "IPV":   0.85,  # polio
    "BCG":   0.70,  # individual protection
    "HEPB":  0.70,
    "PCV":   0.70,
}

SEVERITY_BANDS = [  # weight bands, shown in the interface
    (1.00, "Highest", "Measles and pertussis need about 95% coverage to stop transmission"),
    (0.85, "High", "Polio needs about 80 to 86% coverage"),
    (0.70, "Standard", "Protects the individual child rather than blocking transmission"),
]


def severity_of(antigen_code):  # vaccine weight
    return ANTIGEN_SEVERITY.get(antigen_code, 0.70)


def weighted_missed(missed_codes):  # weighted misses
    """Missed doses added up by weight: a missed MMR counts more than a missed PCV."""
    return sum(severity_of(ANTIGEN_OF.get(c, "")) for c in missed_codes)


def compute_features(date_of_birth, date_registered, dose_records, assessment_date):  # make features
    """The child's features on assessment_date, using only the doses due by then.
    dose_records: [{code, recommended_age_days, date_administered, catch_up_deadline_days}]."""
    age_days = (assessment_date - date_of_birth).days  # age
    registration_delay = max(0, (date_registered - date_of_birth).days)

    due = [r for r in dose_records if r["recommended_age_days"] <= age_days  # due by now
           and (r.get("catch_up_deadline_days") is None or age_days <= r["catch_up_deadline_days"])]
    completed, delayed, missed, delays = [], [], [], []
    last_dose_day = None

    for r in due:  # each due dose
        given = r.get("date_administered")
        if given is None:  # not given
            if age_days - r["recommended_age_days"] > GRACE_PERIOD_DAYS:
                missed.append(r)  # past the grace period = missed
            continue
        if given > assessment_date:  # given after the assessment date
            continue
        given_at_age = (given - date_of_birth).days
        delay = given_at_age - r["recommended_age_days"]  # days late
        delays.append(max(0, delay))
        completed.append(r)  # a late dose still counts as received ...
        if delay > GRACE_PERIOD_DAYS:
            delayed.append(r)  # ... and is also counted as delayed
        if last_dose_day is None or given_at_age > last_dose_day:
            last_dose_day = given_at_age  # latest visit

    doses_due = len(due)
    if doses_due - len(completed) <= 0:  # nothing outstanding: a long gap is normal
        days_since_last = 0.0
    else:
        days_since_last = age_days - last_dose_day if last_dose_day is not None else age_days

    return {  # feature values
        "age_days": age_days,
        "registration_delay_days": registration_delay,
        "doses_due": doses_due,
        "doses_completed": len(completed),
        "doses_delayed": len(delayed),
        "doses_missed": len(missed),
        "weighted_doses_missed": round(weighted_missed([r["code"] for r in missed if r.get("code")]), 2),
        "completion_rate": len(completed) / doses_due if doses_due else 1.0,
        "avg_delay_days": sum(delays) / len(delays) if delays else 0.0,
        "max_delay_days": max(delays) if delays else 0.0,
        "days_since_last_dose": days_since_last,
    }


def final_outcome_label(date_of_birth, dose_records, window_days=365):  # training label
    """The 12-month outcome the model learns: At-Risk if fewer than 80% of the
    first-year doses were on time (within 30 days), or 2 or more were never given."""
    due = [r for r in dose_records if r["recommended_age_days"] <= window_days]
    if not due:
        return "Not At-Risk"
    on_time = sum(1 for r in due if r.get("date_administered") is not None and
                  (r["date_administered"] - date_of_birth).days - r["recommended_age_days"] <= GRACE_PERIOD_DAYS)
    never_given = sum(1 for r in due if r.get("date_administered") is None)
    return "At-Risk" if (on_time / len(due) < 0.8 or never_given >= 2) else "Not At-Risk"


def _dose_records_for(doses_dict):  # doses vs schedule
    """{dose code: date or None} -> the dose list compute_features() expects."""
    return [
        {"code": code, "recommended_age_days": age, "date_administered": doses_dict.get(code),
         "catch_up_deadline_days": CORE_CATCH_UP_DEADLINES.get(code)}
        for code, _name, _vaccine, _n, age in VACCINE_SCHEDULE
    ]


# ===========================================================================
# 4. Synthetic children (training data and demo barangays)
# ===========================================================================

FIRST_NAMES_M = ["Juan", "Miguel", "Jose", "Antonio", "Carlos", "Rafael", "Gabriel", "Diego",
                 "Marco", "Angelo", "Vincent", "Emmanuel", "Joshua", "Nathaniel", "Elijah",
                 "Paolo", "Enrico", "Ramon", "Francisco", "Benedict", "Xavier", "Patrick",
                 "Daniel", "Mark", "Christian", "Kurt", "Ivan", "Luis", "Renzo", "Julian"]
FIRST_NAMES_F = ["Maria", "Sofia", "Isabella", "Andrea", "Camille", "Angela", "Bianca", "Kyla",
                 "Daniella", "Trisha", "Reign", "Althea", "Janelle", "Precious", "Faith",
                 "Patricia", "Erika", "Michelle", "Nicole", "Samantha", "Hannah", "Alexa",
                 "Gabrielle", "Kristine", "Charisse", "Jasmine", "Rica", "Shaira", "Mikaela"]
LAST_NAMES = ["Reyes", "Santos", "Cruz", "Bautista", "Garcia", "Mendoza", "Torres", "Ramos",
              "Flores", "Villanueva", "Castillo", "Aquino", "De Guzman", "Del Rosario",
              "Fernandez", "Gonzales", "Pascual", "Rivera", "Salazar", "Navarro",
              "Ocampo", "Domingo", "Pacheco", "Manalo", "Ferrer", "Macaraeg", "Esguerra",
              "Tolentino", "Agustin", "Marquez", "Soriano", "Lazaro", "Alvarado", "Jimenez",
              "Cabrera", "Valdez", "Panganiban", "Villareal", "Dizon", "Umali"]

# Simulated remoteness of each barangay (fixed seed, so always the same values).
_remoteness_rng = random.Random(42)
BARANGAY_REMOTENESS = {b: round(_remoteness_rng.uniform(0.05, 0.35), 2) for b in BARANGAYS}


def _propensity_for(barangay, rng):  # dropout tendency
    """How likely a simulated child is to skip or delay doses (0.02 to 0.9):
    barangay remoteness plus a random household factor."""
    remoteness = BARANGAY_REMOTENESS.get(barangay, 0.2)
    household = rng.uniform(0.0, 0.35)
    return max(0.02, min(0.9, remoteness * 0.55 + household * 0.55 + rng.uniform(-0.04, 0.04)))


def _simulate_dose_history(dob, propensity, rng):  # simulate doses
    """Higher propensity = more skipped doses and longer delays."""
    records = []
    for code, _name, _vaccine, _n, age in VACCINE_SCHEDULE:  # each dose
        if rng.random() < propensity * 0.22:  # skipped
            given = None
        else:
            delay = max(0, int(rng.gauss(mu=propensity * 16, sigma=7)))  # days late
            given = dob + timedelta(days=age + delay)
            if given > date.today():  # not due yet
                given = None
        records.append({"code": code, "recommended_age_days": age, "date_administered": given})
    return records


def generate_synthetic_children(barangays, n_per_barangay=(6, 14), seed=99,  # demo children
                                dob_start=None, dob_end=None):
    """Simulated children for demo barangays. By default every child is under a
    year old; pass dob_start/dob_end to spread births over several years."""
    rng = random.Random(seed)
    children = []
    for barangay in barangays:
        for _ in range(rng.randint(*n_per_barangay)):
            sex = rng.choice(["M", "F"])
            first = rng.choice(FIRST_NAMES_M if sex == "M" else FIRST_NAMES_F)
            last = rng.choice(LAST_NAMES)
            if dob_start and dob_end:
                dob = dob_start + timedelta(days=rng.randint(0, max(0, (dob_end - dob_start).days)))
            else:
                dob = date.today() - timedelta(days=rng.randint(10, 364))
            propensity = _propensity_for(barangay, rng)
            reg_delay = int(max(0, rng.gauss(mu=propensity * 20, sigma=5)))
            history = _simulate_dose_history(dob, propensity, rng)
            children.append({
                "full_name": f"{first} {last}", "sex": sex, "date_of_birth": dob,
                "barangay": barangay, "date_registered": min(date.today(), dob + timedelta(days=reg_delay)),
                "guardian_name": f"{rng.choice(FIRST_NAMES_F)} {last}",
                "guardian_contact": f"09{rng.randint(100000000, 999999999)}",
                "address": f"Purok {rng.randint(1, 7)}, {barangay}",
                "doses": {r["code"]: r["date_administered"] for r in history},
                "number_of_visits": sum(1 for r in history if r["date_administered"]),
                "moved_out": False,
            })
    return children


# ===========================================================================
# 5. Model: build the training data, train and save
# ===========================================================================

def generate_training_dataset(n_children=3000, seed=7, real_children=None,  # training rows
                              target_share=TRAINING_AT_RISK_SHARE):
    """One training row per child: features at a random age between 30 and
    365 days, labelled with the child's 12-month outcome.

    Uses 3,000 synthetic children plus the real children whose first year is
    complete in the registry (the 221 younger children are left out: their
    outcome is not known yet). Synthetic Not At-Risk rows are then randomly
    removed until the data is 50% At-Risk (target_share=None keeps them all).
    """
    rng = random.Random(seed)
    rows = []

    def add_row(dob, date_registered, doses_dict, barangay, source):  # one row
        dose_records = _dose_records_for(doses_dict)
        assessment_age = rng.randint(ML_ASSESSMENT_FLOOR_DAYS, CONTINUATION_WINDOW_DAYS)  # random age
        assessment_date = min(date(2026, 11, 8), dob + timedelta(days=assessment_age))  # fixed for reproducibility
        row = compute_features(dob, date_registered, dose_records, assessment_date)
        row.update(barangay=barangay, source=source,
                   label=final_outcome_label(dob, dose_records, window_days=365))
        rows.append(row)

    for _ in range(n_children):  # 3,000 synthetic children
        barangay = rng.choice(BARANGAYS)
        rng.choice(["M", "F"])  # sex: drawn to keep the random sequence; not a model input
        dob = date(2025, 1, 1) - timedelta(days=rng.randint(0, 365 * 2))
        propensity = _propensity_for(barangay, rng)
        reg_delay = int(max(0, rng.gauss(mu=propensity * 20, sigma=5)))
        history = _simulate_dose_history(dob, propensity, rng)
        add_row(dob, dob + timedelta(days=reg_delay),
                {r["code"]: r["date_administered"] for r in history}, barangay, "synthetic")

    real_children = real_children or []
    dose_dates = [d for c in real_children for d in c["doses"].values() if d]
    data_end = max(dose_dates) if dose_dates else date.today()  # last dose in the registry
    for c in real_children:
        if c["date_of_birth"] + timedelta(days=CONTINUATION_WINDOW_DAYS) <= data_end:  # first year complete
            add_row(c["date_of_birth"], c["date_registered"], c["doses"], c["barangay"], "real")

    if target_share:  # balance by randomly removing synthetic Not At-Risk rows
        at_risk = sum(1 for r in rows if r["label"] == "At-Risk")
        real_ok = sum(1 for r in rows if r["label"] != "At-Risk" and r["source"] == "real")
        syn_ok = [i for i, r in enumerate(rows) if r["label"] != "At-Risk" and r["source"] == "synthetic"]
        keep = max(0, min(len(syn_ok), round(at_risk * (1 - target_share) / target_share) - real_ok))
        drop = set(syn_ok) - set(random.Random(seed + 1).sample(syn_ok, keep))
        rows = [r for i, r in enumerate(rows) if i not in drop]
    return rows


def train_and_save_model(n_synthetic=3000, seed=7, use_real_data=True):  # train + save
    """Train Logistic Regression, Random Forest and XGBoost on the same 80/20
    split, keep the one with the highest test recall (ROC-AUC breaks ties),
    check it with 5-fold cross-validation and save it to MODEL_PATH."""
    import joblib
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier

    real_children = load_and_merge_real_registries() if use_real_data else []
    if real_children:
        print(f"Loaded {len(real_children)} real children from data/raw/*.csv for training augmentation.")

    rows = generate_training_dataset(n_synthetic, seed, real_children)
    X = np.array([[r[c] for c in FEATURE_COLUMNS] for r in rows], dtype=float)  # features
    y = np.array([1 if r["label"] == "At-Risk" else 0 for r in rows], dtype=int)  # labels
    n_real = sum(1 for r in rows if r["source"] == "real")
    print(f"Training dataset: {len(y)} rows ({n_real} real + {len(y) - n_real} synthetic), "
          f"{y.sum()} At-Risk ({100 * y.mean():.1f}%)")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)  # 80/20
    scaler = StandardScaler().fit(X_train)  # only Logistic Regression uses scaled inputs

    def evaluate(name, model, X_eval):  # test-set metrics
        proba = model.predict_proba(X_eval)[:, 1]
        pred = (proba >= 0.5).astype(int)
        m = {"accuracy": accuracy_score(y_test, pred),
             "precision": precision_score(y_test, pred, zero_division=0),
             "recall": recall_score(y_test, pred, zero_division=0),
             "f1": f1_score(y_test, pred, zero_division=0),
             "roc_auc": roc_auc_score(y_test, proba)}
        print(f"  {name:22s} " + " ".join(f"{k}={v:.3f}" for k, v in m.items()))
        return m

    pos_weight = (len(y_train) - y_train.sum()) / max(1, y_train.sum())  # class balance (1.0 when 50/50)
    lr = LogisticRegression(max_iter=1000, class_weight="balanced").fit(scaler.transform(X_train), y_train)
    rf = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5,
                                class_weight="balanced", random_state=seed).fit(X_train, y_train)
    xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.08, subsample=0.9,
                        colsample_bytree=0.9, scale_pos_weight=pos_weight, eval_metric="logloss",
                        random_state=seed).fit(X_train, y_train)
    candidates = {
        "LogisticRegression": (lr, evaluate("LogisticRegression", lr, scaler.transform(X_test))),
        "RandomForest": (rf, evaluate("RandomForest", rf, X_test)),
        "XGBoost": (xgb, evaluate("XGBoost", xgb, X_test)),
    }

    best_name = max(candidates, key=lambda k: (candidates[k][1]["recall"], candidates[k][1]["roc_auc"]))
    best_model, best_metrics = candidates[best_name]
    print(f"Selected best model -> {best_name}")

    needs_scaler = best_name == "LogisticRegression"
    cv = cross_val_score(best_model, scaler.transform(X_train) if needs_scaler else X_train, y_train,
                         cv=5, scoring="recall")  # 5-fold CV
    print(f"  5-fold CV recall: mean={cv.mean():.3f} std={cv.std():.3f}")

    artifact = {  # saved model file
        "model": best_model, "model_name": best_name, "model_version": "v2-real+synthetic",
        "scaler": scaler if needs_scaler else None, "needs_scaler": needs_scaler,
        "feature_columns": FEATURE_COLUMNS,
        "metrics": best_metrics, "all_metrics": {k: m for k, (_model, m) in candidates.items()},
        "feature_importance": (dict(zip(FEATURE_COLUMNS, best_model.feature_importances_.tolist()))
                               if hasattr(best_model, "feature_importances_") else None),
        "real_data_children": n_real, "synthetic_children": len(y) - n_real,
        "at_risk_share": float(y.mean()),
    }
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)
    print(f"Saved model -> {MODEL_PATH}")
    return artifact


# ===========================================================================
# 6. iNterpret: score a child and explain why
# ===========================================================================

_artifact_cache = None  # model file, loaded once


def model_is_trained():  # model exists?
    return os.path.exists(MODEL_PATH)


def _load_artifact():  # load model
    global _artifact_cache
    if _artifact_cache is None:
        import joblib
        _artifact_cache = joblib.load(MODEL_PATH)
    return _artifact_cache


def model_info():  # model details
    """Which engine is scoring children, for the interface."""
    if model_is_trained():
        art = _load_artifact()
        return {"trained": True, "engine": art.get("model_name", "ML Classifier"),
                "version": art.get("model_version", "unknown"),
                "metrics": art.get("metrics", {}), "all_metrics": art.get("all_metrics", {})}
    return {"trained": False, "engine": SCORER_NAME, "version": SCORER_VERSION,
            "metrics": {}, "all_metrics": {}}


def _in_model_window(age_days):  # 30-365 days?
    return ML_ASSESSMENT_FLOOR_DAYS <= age_days <= CONTINUATION_WINDOW_DAYS


def _label(score):
    return "At-Risk" if score >= RISK_THRESHOLD else "Not At-Risk"


def _model_scores(feature_rows):  # model probabilities
    """Random Forest probability (0 to 1) for each feature row. May raise."""
    import numpy as np
    artifact = _load_artifact()
    x = np.array([[f[c] for c in artifact["feature_columns"]] for f in feature_rows], dtype=float)
    if artifact.get("needs_scaler") and artifact.get("scaler") is not None:  # Logistic Regression only
        x = artifact["scaler"].transform(x)
    return [round(float(p), 4) for p in artifact["model"].predict_proba(x)[:, 1]], artifact


def _predict_with_model(feats):  # model score
    """(score, artifact) from the trained model, or None if it cannot be used."""
    try:
        scores, artifact = _model_scores([feats])
        return scores[0], artifact
    except Exception as exc:
        print(f"WARNING: ML scoring failed ({exc}); falling back to rule-based scorer.")
        return None


# Reasons shown with a model score: features ranked by how far the child is
# from a typical on-track child, weighted by the model's feature importance.
_REFERENCE_NOT_AT_RISK = {  # typical on-track values
    "doses_missed": 0.2, "doses_delayed": 0.5, "avg_delay_days": 8.0, "max_delay_days": 15.0,
    "completion_rate": 0.95, "registration_delay_days": 5.0, "days_since_last_dose": 30.0,
}

_NO_RISK_FACTOR = [{"factor": "On-schedule vaccination history",
                    "detail": "No significant risk indicators detected."}]


def _describe(key, feats):  # reason text
    templates = {
        "weighted_doses_missed": "Missed doses weighted by vaccine importance: {:.1f}",
        "doses_missed": "{:.0f} dose(s) missed beyond the grace period",
        "doses_delayed": "{:.0f} dose(s) administered late",
        "avg_delay_days": "Average delay of {:.0f} days per dose",
        "max_delay_days": "Longest single delay: {:.0f} days",
        "registration_delay_days": "Registered {:.0f} days after birth",
        "days_since_last_dose": "{:.0f} days since last vaccination visit",
    }
    if key == "completion_rate":
        return f"Only {feats['completion_rate'] * 100:.0f}% of due doses completed on time"
    if key in ("doses_missed", "doses_delayed"):  # whole numbers
        return templates[key].format(int(feats[key]))
    return templates[key].format(feats[key]) if key in templates else ""


def _explain(feats, artifact, top_n=3):  # model reasons
    importance = artifact.get("feature_importance") or {}
    scored = []
    for key, typical in _REFERENCE_NOT_AT_RISK.items():  # each feature
        value = feats.get(key, 0)
        how_far = max(0.0, (typical - value) if key == "completion_rate" else (value - typical))
        scored.append((key, how_far * (importance.get(key, 0.1) + 0.05)))
    scored.sort(key=lambda x: x[1], reverse=True)  # biggest first
    factors = [{"factor": FEATURE_LABELS.get(k, k), "detail": _describe(k, feats)}
               for k, s in scored[:top_n] if s > 0]
    return factors or _NO_RISK_FACTOR


# ---------------------------------------------------------------------------
# Rule-based scorer: used for children outside 30-365 days (or if no model
# file exists). Six signals, each scaled 0 to 1, times a fixed weight.
# ---------------------------------------------------------------------------

SCORER_NAME = "Rule-Based Scorer"
SCORER_VERSION = "rules-v3-weighted"

SCORING_RULES = {  # weights (sum to 1.0)
    "weighted_doses_missed":    0.34,
    "completion_rate":          0.24,  # inverted: low completion raises risk
    "days_since_last_dose":     0.16,
    "avg_delay_days":           0.12,
    "doses_delayed":            0.08,
    "registration_delay_days":  0.06,
}

RULE_SATURATION = {  # value at which a rule gives its full weight
    "weighted_doses_missed": 3.0,
    "days_since_last_dose": 180.0,
    "avg_delay_days": 45.0,
    "doses_delayed": 4.0,
    "registration_delay_days": 60.0,
}


def _rule_contributions(feats):  # rule parts
    """{rule: points} for one child."""
    out = {}
    for key, weight in SCORING_RULES.items():  # each rule
        if key == "completion_rate":
            level = 1.0 - float(feats.get("completion_rate", 1.0))  # inverted
        else:
            level = min(1.0, float(feats.get(key, 0)) / RULE_SATURATION[key])  # scaled 0-1
        out[key] = max(0.0, min(1.0, level)) * weight
    return out


def score_child(feats):  # rule score
    return round(min(1.0, sum(_rule_contributions(feats).values())), 4)


def _explain_rules(feats, top_n=3):  # rule reasons
    contribs = sorted(_rule_contributions(feats).items(), key=lambda kv: kv[1], reverse=True)
    factors = [{"factor": FEATURE_LABELS.get(k, k), "detail": _describe(k, feats)}
               for k, points in contribs[:top_n] if points > 0.01]
    return factors or _NO_RISK_FACTOR


def predict_for_child(date_of_birth, date_registered, sex, doses_dict, barangay_name):  # score a child
    """Score one child today. Called whenever a child is saved.

    The trained model is used for children aged 30-365 days (the ages it was
    trained on); everyone else is scored by the rule-based scorer.
    sex and barangay_name are accepted for the callers but are not model inputs."""
    feats = compute_features(date_of_birth, date_registered, _dose_records_for(doses_dict), date.today())

    if model_is_trained() and _in_model_window(feats["age_days"]):
        result = _predict_with_model(feats)
        if result is not None:
            score, artifact = result
            return {"label": _label(score), "probability": score,
                    "model_version": f"{artifact.get('model_name', 'ml')}-{artifact.get('model_version', '')}",
                    "top_factors": _explain(feats, artifact)}

    score = score_child(feats)
    return {"label": _label(score), "probability": score,
            "model_version": SCORER_VERSION, "top_factors": _explain_rules(feats)}


def score_children_as_of(children, assessment_date):  # past-date scores
    """{key: risk score} for many children as they stood on assessment_date
    (the dashboards' date filter). Same rule as predict_for_child().
    children: (key, date_of_birth, date_registered, sex, doses_dict, barangay)."""
    out, model_keys, model_rows = {}, [], []
    use_model = model_is_trained()
    for key, dob, registered, _sex, doses_dict, _barangay in children:  # each child
        feats = compute_features(dob, registered, _dose_records_for(doses_dict), assessment_date)
        if use_model and _in_model_window(feats["age_days"]):
            model_keys.append(key)
            model_rows.append(feats)
        else:
            out[key] = score_child(feats)
    if model_rows:  # score model-age children in one batch
        try:
            scores, _artifact = _model_scores(model_rows)
            out.update(zip(model_keys, scores))
        except Exception as exc:
            print(f"WARNING: ML scoring failed ({exc}); falling back to rule-based scorer.")
            out.update((k, score_child(f)) for k, f in zip(model_keys, model_rows))
    return out


def risk_tier(probability):  # risk tier
    if probability >= 0.70:
        return "High Risk"
    if probability >= 0.50:
        return "Medium Risk"
    return "Low Risk"


def risk_tier_color(tier):  # tier color
    return {"High Risk": "red", "Medium Risk": "amber"}.get(tier, "green")


def serialize_factors(factors):  # reasons -> text for the database
    return json.dumps(factors)


# ===========================================================================
# 7. Stock allocation when requests exceed the stock on hand
# ===========================================================================
# Three suggestions; RHU personnel decide. No barangay gets more than it asked for.
#   priority      Urgent first, then High, then Normal (larger requests first within a level)
#   proportional  everyone gets the same share of what they asked for
#   coverage      lowest-coverage barangays first

PRIORITY_RANK = {"Urgent": 0, "High": 1, "Normal": 2}


def _largest_remainder(weights, available):  # fair rounding
    """Split `available` whole vials in proportion to weights without losing a
    vial: whole parts first, leftover vials to the largest fractions."""
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


def _fill_in_order(ordered_requests, available):
    """Give each request in turn as much as it asked for, until stock runs out."""
    alloc, left = {}, available
    for r in ordered_requests:
        alloc[r["id"]] = min(r["requested"], left)
        left -= alloc[r["id"]]
    return alloc


def allocate_stock(requests, available, method="priority", coverage_by_bgy=None):  # split stock
    """Suggested split of `available` vials.
    requests: [{"id", "barangay", "requested", "priority"}]."""
    total_requested = sum(r["requested"] for r in requests)

    if total_requested <= available:  # enough for everyone
        alloc = {r["id"]: r["requested"] for r in requests}
    elif method == "proportional":
        alloc = _largest_remainder({r["id"]: r["requested"] for r in requests}, available)
    elif method == "coverage":
        cov = coverage_by_bgy or {}
        alloc = _fill_in_order(sorted(requests, key=lambda r: cov.get(r["barangay"], 100)), available)
    else:  # priority
        alloc = _fill_in_order(sorted(requests, key=lambda r: (PRIORITY_RANK.get(r["priority"], 9),
                                                               -r["requested"])), available)

    rows = []
    for r in requests:
        given = min(alloc.get(r["id"], 0), r["requested"])
        rows.append({**r, "allocated": given, "short": r["requested"] - given,
                     "share": round(given / r["requested"] * 100) if r["requested"] else 0})
    return {"rows": rows, "available": available, "requested": total_requested,
            "allocated": sum(r["allocated"] for r in rows),
            "shortfall": max(0, total_requested - available),
            "can_meet_all": total_requested <= available}


# ===========================================================================
# 8. Vaccine demand forecast (rule-based, not machine learning)
# ===========================================================================
# Doses needed in a future year = known + projected:
#   known      doses that children already registered will become due for (exact)
#   projected  expected births x doses per child (an estimate, given as a range)
# Births are estimated from the average of the last 3 complete years, not a
# trend line (the registry's early years are too small to fit a trend).

FORECAST_YEARS = 3          # years ahead
COHORT_BASE_YEARS = 3       # complete years averaged
COHORT_MIN_ROWS = 20        # smaller years are registry start-up, not a real cohort
COHORT_HIGH_MARGIN = 0.15   # high estimate = average + 15%


def cohort_estimate(births_by_year, today_year):  # birth estimate
    """Low and high estimate of yearly births, from the last 3 complete years."""
    usable = {y: n for y, n in births_by_year.items() if y < today_year and n >= COHORT_MIN_ROWS}
    if not usable:
        return None
    years = sorted(usable)[-COHORT_BASE_YEARS:]
    mean = sum(usable[y] for y in years) / len(years)
    return {"low": round(mean), "high": round(mean * (1 + COHORT_HIGH_MARGIN)),
            "years": years, "counts": {y: usable[y] for y in years}}


def doses_due_in_year(date_of_birth, year):  # due that year
    return [code for code, _n, _v, _d, age in VACCINE_SCHEDULE
            if (date_of_birth + timedelta(days=age)).year == year]


def forecast_vaccine_demand(children_dobs, births_by_year, today=None,  # dose forecast
                            years=FORECAST_YEARS):
    """Doses needed per vaccine for each of the next `years` years, with the
    known and projected parts kept separate."""
    today = today or date.today()
    doses_per_child = Counter(ANTIGEN_OF.values())  # e.g. Pentavalent: 3
    cohort = cohort_estimate(births_by_year, today.year)

    out = []
    for y in range(today.year + 1, today.year + years + 1):  # each year
        known = Counter(ANTIGEN_OF[code] for dob in children_dobs for code in doses_due_in_year(dob, y))
        rows = []
        for vaccine, n_doses in doses_per_child.items():  # each vaccine
            k = known.get(vaccine, 0)
            p_low = round(cohort["low"] * n_doses) if cohort else 0
            p_high = round(cohort["high"] * n_doses) if cohort else 0
            rows.append({"antigen": vaccine, "doses_per_child": n_doses, "known": k,
                         "projected_low": p_low, "projected_high": p_high,
                         "total_low": k + p_low, "total_high": k + p_high})
        rows.sort(key=lambda r: -r["total_high"])
        out.append({"year": y, "rows": rows,
                    "total_low": sum(r["total_low"] for r in rows),
                    "total_high": sum(r["total_high"] for r in rows),
                    "known_total": sum(r["known"] for r in rows)})
    return {"cohort": cohort, "years": out}
