"""Reproduces Table 21 (Example Prediction Results) from the paper, scored
by whichever model is currently saved at data/risk_model.joblib — so you
can see what changes when the selected model is RandomForest instead of
XGBoost (or vice versa), with everything else held identical.

The five profiles (A-E) are built to match the paper's exact Doses
Due/Given/Missed counts and narrative descriptions, assessed on the same
reference date the paper used (10 Oct 2026), so this is an apples-to-apples
comparison — the only thing that changes between this and the existing
Table 21 is which trained model does the scoring.

Run: python example_profiles.py
"""
from datetime import date, timedelta

import data_processor as dp

ASSESSMENT_DATE = date(2026, 10, 10)
BARANGAY = "Awai"


def _profile(dob, given_offsets, date_registered=None):
    """given_offsets: {vaccine_code: days_after_dob the dose was given}.
    Any core vaccine code not present is left ungiven (None)."""
    doses = {code: None for code in dp.VACCINE_CODES}
    for code, offset in given_offsets.items():
        doses[code] = dob + timedelta(days=offset)
    return dob, (date_registered or dob), doses


def build_profiles():
    schedule = {code: due_days for code, _, _, _, due_days in dp.VACCINE_SCHEDULE}

    # A. 4 months old, all due doses given exactly on time.
    dob_a = date(2026, 6, 10)
    due_a = [c for c, d in schedule.items() if d <= (ASSESSMENT_DATE - dob_a).days]
    dob_a, reg_a, doses_a = _profile(dob_a, {c: schedule[c] for c in due_a})

    # B. 4 months old, only the two birth doses (BCG, HEPA-BD) given.
    dob_b = date(2026, 6, 10)
    dob_b, reg_b, doses_b = _profile(dob_b, {"BCG": 0, "HEPA-BD": 0})

    # C. 7 months old, every due dose given about 40 days late.
    dob_c = date(2026, 3, 10)
    due_c = [c for c, d in schedule.items() if d <= (ASSESSMENT_DATE - dob_c).days]
    dob_c, reg_c, doses_c = _profile(
        dob_c, {c: schedule[c] + 40 for c in due_c}, date_registered=dob_c + timedelta(days=20))

    # D. 11 months old, stopped after the 6-week visit (BCG, HEPA-BD, then
    # the day-42 batch: PENTA1/OPV1/PCV1), nothing since.
    dob_d = date(2025, 11, 10)
    dob_d, reg_d, doses_d = _profile(
        dob_d, {"BCG": 0, "HEPA-BD": 0, "PENTA1": 42, "OPV1": 42, "PCV1": 42})

    # E. 2 years old, both MMR doses missing; everything else on schedule.
    dob_e = date(2024, 10, 10)
    due_e = [c for c, d in schedule.items() if c not in ("MMR1", "MMR2")]
    dob_e, reg_e, doses_e = _profile(dob_e, {c: schedule[c] for c in due_e})

    return [
        ("A. 4 months old, all due doses on time", dob_a, reg_a, doses_a),
        ("B. 4 months old, birth doses only", dob_b, reg_b, doses_b),
        ("C. 7 months old, every dose about 40 days late", dob_c, reg_c, doses_c),
        ("D. 11 months old, stopped after the 6-week visit", dob_d, reg_d, doses_d),
        ("E. 2 years old, both MMR doses missing", dob_e, reg_e, doses_e),
    ]


def score_profile(dob, date_registered, doses_dict):
    """Same logic as predict_for_child(), but pinned to ASSESSMENT_DATE
    instead of date.today(), so this lines up with the paper's reference
    date instead of whatever day this script happens to be run."""
    dose_records = dp._dose_records_for(doses_dict)
    age_days = (ASSESSMENT_DATE - dob).days
    feats = dp.compute_features(
        dob, date_registered, "F", dose_records, ASSESSMENT_DATE,
        barangay_risk_rate=dp.BARANGAY_BASE_RISK.get(BARANGAY, 0.15),
    )
    due, given, missed = feats["doses_due"], feats["doses_completed"], feats["doses_missed"]

    if dp.model_is_trained() and dp.ML_ASSESSMENT_FLOOR_DAYS <= age_days <= dp.CONTINUATION_WINDOW_DAYS:
        ml_result = dp._predict_with_model(feats)
        if ml_result is not None:
            score, artifact = ml_result
            engine = artifact.get("model_name", "ml")
            factors = dp._explain(feats, artifact)
            label = "At-Risk" if score >= dp.RISK_THRESHOLD else "Not At-Risk"
            return due, given, missed, engine, label, score, factors

    score = dp.score_child(feats)
    label = "At-Risk" if score >= dp.RISK_THRESHOLD else "Not At-Risk"
    return due, given, missed, "Rule-based scorer", label, score, dp._explain_rules(feats)


def main():
    if not dp.model_is_trained():
        raise SystemExit("No trained model found at data/risk_model.joblib - run train_model.py first.")

    rows = []
    for name, dob, reg, doses in build_profiles():
        due, given, missed, engine, label, score, factors = score_profile(dob, reg, doses)
        top3 = "; ".join(f["factor"] for f in factors[:3])
        rows.append((name, f"{due} / {given} / {missed}", engine, f"{label} ({score:.3f})", top3))

    headers = ["Profile", "Doses Due/Given/Missed", "Engine Used", "Result (Probability)", "Top Factors Shown to the User"]
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*["-" * w for w in widths]))
    for r in rows:
        print(fmt.format(*r))

    print(f"\nAssessed as of {ASSESSMENT_DATE.isoformat()} (same reference date as the paper's Table 21).")
    print("Compare the 'Engine Used' column against your current paper to see what changed.")


if __name__ == "__main__":
    main()
