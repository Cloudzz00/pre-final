"""Reproduces Table 15 (Distribution of Immunization Completion Status)
from the paper, computed directly from data/raw/*.csv on the same
reference date the paper uses (10 Oct 2026):

(a) Fully immunized for age: completion_rate (doses given / doses due by
    the reference date, regardless of delay) >= 90%.
(b) Not At-Risk: the model's own training label (final_outcome_label,
    window_days=365) -- at least 80% of doses due by 12 months of age were
    given within the 30-day grace period, and fewer than two were never
    given at all.
(c) Strict completion: completion_rate == 100% (every dose due by the
    reference date is recorded, regardless of delay).

Run: python table_15.py
"""
from datetime import date

import data_processor as dp

ASSESSMENT_DATE = date(2026, 10, 10)


def main():
    children = dp.load_and_merge_real_registries()
    n = len(children)

    fully_immunized = 0
    strict_complete = 0
    not_at_risk = 0

    for c in children:
        dose_records = dp._dose_records_for(c["doses"])

        feats = dp.compute_features(
            c["date_of_birth"], c["date_registered"], c["sex"],
            dose_records, ASSESSMENT_DATE, barangay_risk_rate=None,
        )
        if feats["completion_rate"] >= 0.90:
            fully_immunized += 1
        if feats["completion_rate"] >= 1.0:
            strict_complete += 1

        label = dp.final_outcome_label(c["date_of_birth"], dose_records, window_days=365)
        if label == "Not At-Risk":
            not_at_risk += 1

    not_fully = n - fully_immunized
    at_risk = n - not_at_risk
    not_complete = n - strict_complete

    print(f"n = {n}\n")
    print("(a) Fully immunized for age")
    print(f"  Fully immunized   {fully_immunized:>5}  {fully_immunized/n*100:.2f}%")
    print(f"  Not fully         {not_fully:>5}  {not_fully/n*100:.2f}%")
    print()
    print("(b) 12-month outcome (model target)")
    print(f"  Not-At-Risk       {not_at_risk:>5}  {not_at_risk/n*100:.2f}%")
    print(f"  At-Risk           {at_risk:>5}  {at_risk/n*100:.2f}%")
    print()
    print("Strict completion")
    print(f"  Complete          {strict_complete:>5}  {strict_complete/n*100:.2f}%")
    print(f"  Not complete      {not_complete:>5}  {not_complete/n*100:.2f}%")


if __name__ == "__main__":
    main()
