"""Single entry point that regenerates every table and figure this session
produced for Chapter 4 of the capstone paper, all from the live
data/raw/*.csv registries plus a freshly trained model -- so the whole set
can be re-run in one command whenever the underlying data changes again.
Replaces table_15.py, table_17_18.py, example_profiles.py and
evaluate_model.py, which this now fully supersedes.

Produces, under --out (default paper_artifacts/):

  tables/table09_dataset_overview.txt
  tables/table11_data_quality.txt
  tables/table13_descriptive_stats.txt
  tables/table14_dose_administration.txt
  tables/table15_completion_status.txt
  tables/table17_train_test_split.txt
  tables/table18_model_comparison.txt
  tables/table19_classification_report.txt
  tables/table20_supplementary_evaluation.txt
  tables/table21_example_predictions.txt
  tables/extra_worked_example_predictions.txt   (not a numbered paper table)
  figures/figure03_records_per_barangay.png
  figures/figure04_birth_year_distribution.png
  figures/figure05_coverage_timeliness.png
  figures/figure06_fully_immunized_by_barangay.png
  figures/figure07_confusion_matrix.png
  figures/extra_feature_importance.png          (not a numbered paper figure)
  figures/extra_roc_precision_recall.png         (not a numbered paper figure)

Table/figure numbers match this session's paper draft, not any fixed
numbering scheme -- update the filenames if your paper's numbering moves.
The "extra_*" files aren't part of the paper's own numbering; they're the
groupmate's original feature-importance/supplementary-evaluation/
example-predictions ask, kept for convenience.

Run:
    python generate_paper_artifacts.py
    python generate_paper_artifacts.py --out my_artifacts --n-synthetic 3000 --seed 7
"""
import argparse
import os
import statistics
from collections import Counter, defaultdict
from datetime import date, timedelta

import numpy as np

import data_processor as dp

GRACE = dp.GRACE_PERIOD_DAYS
CORE_DOSE_CODES = ["BCG", "HEPA-BD", "PENTA1", "OPV1", "PCV1", "PENTA2", "OPV2", "PCV2",
                    "PENTA3", "OPV3", "PCV3", "IPV1", "IPV2", "MMR1", "MMR2"]
DOSE_DISPLAY = {
    "BCG": "BCG", "HEPA-BD": "Hepa B-BD", "PENTA1": "DPT-Hib-HepB 1", "OPV1": "OPV 1",
    "PCV1": "PCV 1", "PENTA2": "DPT-Hib-HepB 2", "OPV2": "OPV 2", "PCV2": "PCV 2",
    "PENTA3": "DPT-Hib-HepB 3", "OPV3": "OPV 3", "PCV3": "PCV 3", "IPV1": "IPV 1",
    "IPV2": "IPV 2", "MMR1": "MMR 1", "MMR2": "MMR 2",
}
DOSE_AGE_LABEL = {
    "BCG": "Birth", "HEPA-BD": "Birth", "PENTA1": "6 weeks", "OPV1": "6 weeks", "PCV1": "6 weeks",
    "PENTA2": "10 weeks", "OPV2": "10 weeks", "PCV2": "10 weeks",
    "PENTA3": "14 weeks", "OPV3": "14 weeks", "PCV3": "14 weeks", "IPV1": "14 weeks",
    "IPV2": "9 months", "MMR1": "9 months", "MMR2": "12 months",
}


def _write(path, text):
    with open(path, "w") as f:
        f.write(text)
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------
# Table 9. Dataset Overview
# ---------------------------------------------------------------------------
def table_09(children, assessment_date, out_dir):
    n = len(children)
    total_dose_records = n * len(CORE_DOSE_CODES)
    recorded = sum(1 for c in children for code in CORE_DOSE_CODES if c["doses"].get(code) is not None)
    dobs = [c["date_of_birth"] for c in children]
    regs = [c["date_registered"] for c in children]
    admins = [d for c in children for d in c["doses"].values() if d is not None]

    lines = [
        "Table 9. Dataset Overview", "",
        f"Child records: {n}",
        f"Barangays covered: {len(set(c['barangay'] for c in children))}",
        f"Vaccination records (15 core doses per child): {total_dose_records}",
        f"Doses with a recorded administration date: {recorded} ({recorded/total_dose_records*100:.2f}%)",
        f"Range of dates of birth: {min(dobs).strftime('%d %b %Y')} - {max(dobs).strftime('%d %b %Y')}",
        f"Range of dates of registration: {min(regs).strftime('%d %b %Y')} - {max(regs).strftime('%d %b %Y')}",
        f"Range of dose administration dates: {min(admins).strftime('%d %b %Y')} - {max(admins).strftime('%d %b %Y')}",
    ]
    _write(os.path.join(out_dir, "table09_dataset_overview.txt"), "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Table 11. Data Quality Findings
# ---------------------------------------------------------------------------
def table_11(children, assessment_date, out_dir):
    n = len(children)
    exact_dup = Counter((c["full_name"], c["date_of_birth"], c["sex"], c["barangay"]) for c in children)
    exact_dupes = sum(cnt for cnt in exact_dup.values() if cnt > 1)

    possible_dup = Counter((c["date_of_birth"], c["sex"], c["barangay"], c["date_registered"]) for c in children)
    possible_dup_records = sum(cnt for cnt in possible_dup.values() if cnt > 1)

    missing_dob = 0  # rows without a DOB are dropped on import, so always 0 among stored records
    missing_name = sum(1 for c in children if not c.get("full_name"))

    doses_total = n * len(CORE_DOSE_CODES)
    doses_without_date = sum(1 for c in children for code in CORE_DOSE_CODES if c["doses"].get(code) is None)

    impossible_dates = sum(1 for c in children for code, admin in c["doses"].items()
                            if admin is not None and admin < c["date_of_birth"])

    reg_before_dob = sum(1 for c in children if c["date_registered"] < c["date_of_birth"])
    reg_after_ref = sum(1 for c in children if c["date_registered"] > assessment_date)

    sparse_barangays = ["Awai", "Bolo", "Labney", "Santa Cruz"]
    sparse_children = sum(1 for c in children if c["barangay"] in sparse_barangays)

    ipv2_sparse_barangays = ["San Jose", "Santo Tomas", "Macayug"]
    ipv2_sparse_children = sum(1 for c in children if c["barangay"] in ipv2_sparse_barangays)

    moved_out = sum(1 for c in children if c.get("moved_out"))
    reg_delay_over_year = sum(1 for c in children if (c["date_registered"] - c["date_of_birth"]).days > 365)

    lines = [
        f"Table 11. Data Quality Findings (RHU Registry Records, n = {n})", "",
        f"Missing date of birth, sex or barangay: {missing_dob}",
        f"Missing child name: {missing_name}",
        f"Exact duplicate children (same name, DOB, sex, barangay): {exact_dupes}",
        f"Possible duplicates (same DOB, sex, barangay, registration date): {possible_dup_records} records",
        f"Doses without an administration date: {doses_without_date} of {doses_total} "
        f"({doses_without_date/doses_total*100:.2f}%)",
        f"Dose dated before the child's date of birth: {impossible_dates} doses "
        f"({impossible_dates/recorded_count(children)*100:.2f}% of recorded doses)",
        f"Registration date earlier than date of birth: {reg_before_dob} records",
        f"Registration date later than the reference date: {reg_after_ref} record(s)",
        f"Barangays with systematically sparse dose columns (Hepa B-BD/BCG): "
        f"{', '.join(sparse_barangays)}: {sparse_children} records ({sparse_children/n*100:.2f}%)",
        f"IPV 2nd dose not recorded: {', '.join(ipv2_sparse_barangays)}: {ipv2_sparse_children} records",
        f"Children marked \"MOVE OUT\": {moved_out}",
        f"Registration delay over one year: {reg_delay_over_year} records",
    ]
    _write(os.path.join(out_dir, "table11_data_quality.txt"), "\n".join(lines) + "\n")


def recorded_count(children):
    return sum(1 for c in children for code in CORE_DOSE_CODES if c["doses"].get(code) is not None)


# ---------------------------------------------------------------------------
# Table 13. Descriptive Statistics
# ---------------------------------------------------------------------------
def table_13(children, assessment_date, out_dir):
    n = len(children)
    male = sum(1 for c in children if c["sex"] == "M")
    female = n - male

    by_year = Counter(c["date_of_birth"].year for c in children)
    yr_2021_2024 = sum(cnt for yr, cnt in by_year.items() if 2021 <= yr <= 2024)
    yr_2025_2026 = sum(cnt for yr, cnt in by_year.items() if yr >= 2025)
    yr_before_2021 = n - yr_2021_2024 - yr_2025_2026

    ages = [(assessment_date - c["date_of_birth"]).days for c in children]
    aged_12mo = sum(1 for a in ages if a <= 365)

    delays_unclipped = [(c["date_registered"] - c["date_of_birth"]).days for c in children]
    q = statistics.quantiles(delays_unclipped, n=4)
    day0 = sum(1 for d in delays_unclipped if d == 0)
    over30 = sum(1 for d in delays_unclipped if d > 30)

    dose_counts = [sum(1 for code in CORE_DOSE_CODES if c["doses"].get(code) is not None) for c in children]
    q2 = statistics.quantiles(dose_counts, n=4)
    all15 = sum(1 for cnt in dose_counts if cnt == 15)
    no_dose = sum(1 for cnt in dose_counts if cnt == 0)

    vita = sum(1 for c in children if c.get("vitamin_a_date") is not None)

    by_barangay = Counter(c["barangay"] for c in children)
    largest = by_barangay.most_common(1)[0]
    smallest = min(by_barangay.items(), key=lambda kv: kv[1])

    lines = [
        f"Table 13. Descriptive Statistics of Child Immunization Records (RHU Registry, n = {n})", "",
        f"Sex: Male {male} ({male/n*100:.2f}%) / Female {female} ({female/n*100:.2f}%)",
        f"Year of birth: 2021-2024 {yr_2021_2024} ({yr_2021_2024/n*100:.2f}%) / "
        f"2025-2026 {yr_2025_2026} ({yr_2025_2026/n*100:.2f}%) / "
        f"2020 and earlier {yr_before_2021} ({yr_before_2021/n*100:.2f}%)",
        f"Age on {assessment_date.strftime('%d %b %Y')}: median {statistics.median(ages):.0f} days "
        f"(min {min(ages)} - max {max(ages)})",
        f"Aged 12 months or younger: {aged_12mo} ({aged_12mo/n*100:.2f}%)",
        f"Registration delay (days after birth): median {statistics.median(delays_unclipped):.0f} "
        f"(Q1 {q[0]:.0f} - Q3 {q[2]:.0f})",
        f"  Mean (SD): {statistics.mean(delays_unclipped):.2f} ({statistics.stdev(delays_unclipped):.2f})",
        f"  Registered on the day of birth: {day0} ({day0/n*100:.2f}%)",
        f"  Registered more than 30 days after birth: {over30} ({over30/n*100:.2f}%)",
        f"Doses recorded per child (of 15): median {statistics.median(dose_counts):.0f} "
        f"(Q1 {q2[0]:.0f} - Q3 {q2[2]:.0f})",
        f"  All 15 doses recorded: {all15} ({all15/n*100:.2f}%)",
        f"  No dose recorded: {no_dose} ({no_dose/n*100:.2f}%)",
        f"Vitamin A date recorded: {vita} ({vita/n*100:.2f}%)",
        f"Barangay: largest {largest[0]}, {largest[1]} / smallest {smallest[0]}, {smallest[1]}",
    ]
    _write(os.path.join(out_dir, "table13_descriptive_stats.txt"), "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Table 14. Vaccine Administration Records per Dose
# ---------------------------------------------------------------------------
def dose_stats(children, assessment_date):
    """Returns {code: (eligible, recorded, pct_recorded, median_delay, pct_grace)}."""
    out = {}
    for code, name, antigen, dose_no, rec_days in dp.VACCINE_SCHEDULE:
        eligible = 0
        recorded = 0
        within_grace = 0
        delays = []
        for c in children:
            age_days = (assessment_date - c["date_of_birth"]).days
            if rec_days > age_days:
                continue
            eligible += 1
            admin = c["doses"].get(code)
            if admin is None or admin > assessment_date:
                continue
            recorded += 1
            delay = (admin - c["date_of_birth"]).days - rec_days
            delays.append(delay)
            if delay <= GRACE:
                within_grace += 1
        pct_recorded = recorded / eligible * 100 if eligible else 0.0
        pct_grace = within_grace / recorded * 100 if recorded else 0.0
        median_delay = statistics.median(delays) if delays else 0
        out[code] = (eligible, recorded, pct_recorded, median_delay, pct_grace)
    return out


def table_14(children, assessment_date, out_dir):
    stats = dose_stats(children, assessment_date)
    n = len(children)
    lines = [f"Table 14. Vaccine Administration Records per Dose (RHU Registry, n = {n})", ""]
    header = f"{'Dose':<16}{'Recommended Age':<17}{'Recorded':>9}{'% Age-Eligible':>16}{'Median Delay':>14}{'% Within Grace':>16}"
    lines.append(header)
    lines.append("-" * len(header))
    for code, name, antigen, dose_no, rec_days in dp.VACCINE_SCHEDULE:
        eligible, recorded, pct_recorded, median_delay, pct_grace = stats[code]
        lines.append(f"{DOSE_DISPLAY[code]:<16}{DOSE_AGE_LABEL[code]:<17}{recorded:>9,}"
                     f"{pct_recorded:>16.2f}{median_delay:>14.0f}{pct_grace:>16.2f}")
    ipv2_mmr1_elig = stats["IPV2"][0]
    mmr2_elig = stats["MMR2"][0]
    lines.append("")
    lines.append(f"Age-eligible children: {n:,} for doses up to 14 weeks; "
                 f"{ipv2_mmr1_elig:,} for IPV 2 and MMR 1; {mmr2_elig:,} for MMR 2. "
                 "Delay statistics are computed over recorded doses only.")
    _write(os.path.join(out_dir, "table14_dose_administration.txt"), "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Table 15. Distribution of Immunization Completion Status
# ---------------------------------------------------------------------------
def table_15(children, assessment_date, out_dir):
    n = len(children)
    fully_immunized = 0
    strict_complete = 0
    not_at_risk = 0
    for c in children:
        dose_records = dp._dose_records_for(c["doses"])
        feats = dp.compute_features(c["date_of_birth"], c["date_registered"], c["sex"],
                                     dose_records, assessment_date, barangay_risk_rate=None)
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

    lines = [
        f"Table 15. Distribution of Immunization Completion Status (RHU Registry, n = {n})", "",
        "(a) Fully immunized for age (>=90% of doses due by reference date recorded):",
        f"  Fully immunized: {fully_immunized} ({fully_immunized/n*100:.2f}%)",
        f"  Not fully immunized: {not_fully} ({not_fully/n*100:.2f}%)", "",
        "(b) 12-month outcome (model target):",
        f"  Not-At-Risk: {not_at_risk} ({not_at_risk/n*100:.2f}%)",
        f"  At-Risk: {at_risk} ({at_risk/n*100:.2f}%)", "",
        "Strict completion (every dose due by reference date recorded):",
        f"  Complete: {strict_complete} ({strict_complete/n*100:.2f}%)",
        f"  Not complete: {not_complete} ({not_complete/n*100:.2f}%)",
    ]
    _write(os.path.join(out_dir, "table15_completion_status.txt"), "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Figure 3. Number of Child Records per Barangay
# ---------------------------------------------------------------------------
def figure_03(children, out_dir, plt):
    counts = Counter(c["barangay"] for c in children)
    data = counts.most_common()
    names = [d[0] for d in data]
    values = [d[1] for d in data]

    fig, ax = plt.subplots(figsize=(9, 7), dpi=200)
    y_pos = range(len(names))
    bars = ax.barh(y_pos, values, color="#1f77b4", height=0.6, label="RHU registry (real)")
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 4, bar.get_y() + bar.get_height() / 2, str(val),
                va="center", ha="left", fontsize=8.5, color="#333333")
    ax.set_xlabel("Number of child records", fontsize=10)
    ax.set_xlim(0, max(values) * 1.08)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#e4e3df", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    plt.tight_layout()
    path = os.path.join(out_dir, "figure03_records_per_barangay.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------
# Figure 4. Distribution of Registry Records by Year of Birth
# ---------------------------------------------------------------------------
def figure_04(children, out_dir, plt):
    counts = Counter(c["date_of_birth"].year for c in children)
    years = list(range(min(counts), max(counts) + 1))
    values = [counts.get(y, 0) for y in years]

    fig, ax = plt.subplots(figsize=(7.2, 3.0), dpi=200)
    bars = ax.bar(years, values, color="#636efa", width=0.6)
    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.015, str(val),
                    ha="center", va="bottom", fontsize=7, color="#333333")
    ax.set_xticks(years)
    ax.set_xticklabels(years, fontsize=7)
    ax.set_ylabel("Children", fontsize=8)
    ax.set_xlabel("Year of birth", fontsize=8)
    ax.set_ylim(0, max(values) * 1.12)
    ax.tick_params(axis="y", labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")
    ax.grid(axis="y", color="#eeeeee", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    plt.tight_layout()
    path = os.path.join(out_dir, "figure04_birth_year_distribution.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------
# Figure 5. Recorded Coverage and Timeliness per Vaccine Dose
# ---------------------------------------------------------------------------
def figure_05(children, assessment_date, out_dir, plt):
    stats = dose_stats(children, assessment_date)
    labels = [DOSE_DISPLAY[code].replace("DPT-Hib-HepB", "DPT").replace("Hepa B-BD", "HEPA-BD")
              for code, *_ in dp.VACCINE_SCHEDULE]
    pct_recorded = [stats[code][2] for code, *_ in dp.VACCINE_SCHEDULE]
    pct_grace = [stats[code][4] for code, *_ in dp.VACCINE_SCHEDULE]

    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8.5, 4.0), dpi=200)
    ax.bar(x - width / 2, pct_recorded, width, color="#4472c4", label="Dose recorded (% of age-eligible children)")
    ax.bar(x + width / 2, pct_grace, width, color="#2ca58d", label="Given within 30-day grace (% of recorded doses)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Percent", fontsize=9)
    ax.set_ylim(0, 100)
    ax.tick_params(axis="y", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e8e8e8", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=False, fontsize=8)
    plt.tight_layout()
    path = os.path.join(out_dir, "figure05_coverage_timeliness.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------
# Figure 6. Children Fully Immunized for Their Age, by Barangay
# ---------------------------------------------------------------------------
def figure_06(children, assessment_date, out_dir, plt):
    by_b = defaultdict(lambda: [0, 0])
    for c in children:
        dose_records = dp._dose_records_for(c["doses"])
        feats = dp.compute_features(c["date_of_birth"], c["date_registered"], c["sex"],
                                     dose_records, assessment_date, barangay_risk_rate=None)
        b = c["barangay"]
        by_b[b][1] += 1
        if feats["completion_rate"] >= 0.90:
            by_b[b][0] += 1

    rows = [(b, cnt[0], cnt[1], cnt[0] / cnt[1] * 100 if cnt[1] else 0) for b, cnt in by_b.items()]
    rows.sort(key=lambda r: -r[3])
    names = [r[0] for r in rows]
    pcts = [r[3] for r in rows]
    ns = [r[2] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 7.5), dpi=200)
    y_pos = range(len(names))
    bars = ax.barh(y_pos, pcts, color="#4472c4", height=0.62)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    for bar, pct, n in zip(bars, pcts, ns):
        ax.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}% (n={n})", va="center", ha="left", fontsize=7.8, color="#333333")
    ax.axvline(95, color="#888888", linestyle="--", linewidth=1)
    ax.text(95, -1.3, "DOH 95% target", ha="center", fontsize=8, color="#555555")
    ax.set_xlabel("Children fully immunized for their age (%)", fontsize=9.5)
    ax.set_xlim(0, 112)
    ax.set_ylim(len(names) - 0.3, -1.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#e8e8e8", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    plt.tight_layout()
    path = os.path.join(out_dir, "figure06_fully_immunized_by_barangay.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------
# Tables 17 & 18: train/test split + 3-candidate comparison (with specificity)
# ---------------------------------------------------------------------------
def tables_17_18(children, args, tables_dir):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    rows, _ = dp.generate_training_dataset(args.n_synthetic, args.seed, children)
    source = ["Registry" if r["source"] == "real" else "Synthetic" for r in rows]

    X = np.array([[r[c] for c in dp.FEATURE_COLUMNS] for r in rows], dtype=float)
    y = np.array([1 if r["label"] == "At-Risk" else 0 for r in rows], dtype=int)
    src = np.array(source)

    X_train, X_test, y_train, y_test, src_train, src_test = train_test_split(
        X, y, src, test_size=0.2, random_state=args.seed, stratify=y)

    def counts(y_arr, src_arr):
        from_registry = int((src_arr == "Registry").sum())
        from_synthetic = int((src_arr == "Synthetic").sum())
        at_risk = int(y_arr.sum())
        return len(y_arr), from_registry, from_synthetic, at_risk, len(y_arr) - at_risk

    lines17 = [f"Table 17. Training and Testing Dataset Distribution", ""]
    header = f"{'Set':<16}{'Rows':>8}{'From Registry':>16}{'From Synthetic':>16}{'At-Risk':>10}{'Not At-Risk':>14}"
    lines17.append(header)
    tr, te, tot = counts(y_train, src_train), counts(y_test, src_test), counts(y, src)
    lines17.append(f"{'Training (80%)':<16}{tr[0]:>8}{tr[1]:>16}{tr[2]:>16}{tr[3]:>10}{tr[4]:>14}")
    lines17.append(f"{'Testing (20%)':<16}{te[0]:>8}{te[1]:>16}{te[2]:>16}{te[3]:>10}{te[4]:>14}")
    lines17.append(f"{'Total':<16}{tot[0]:>8}{tot[1]:>16}{tot[2]:>16}{tot[3]:>10}{tot[4]:>14}")
    reg_risk_rate = sum(1 for r, s in zip(y, src) if r == 1 and s == "Registry") / max(1, (src == "Registry").sum())
    syn_risk_rate = sum(1 for r, s in zip(y, src) if r == 1 and s == "Synthetic") / max(1, (src == "Synthetic").sum())
    lines17.append("")
    lines17.append(f"Registry At-Risk rate: {reg_risk_rate*100:.2f}%  |  Synthetic At-Risk rate: {syn_risk_rate*100:.2f}%  "
                   f"|  Combined: {y.mean()*100:.2f}%")
    _write(os.path.join(tables_dir, "table17_train_test_split.txt"), "\n".join(lines17) + "\n")

    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

    def evaluate(name, model, Xte):
        proba = model.predict_proba(Xte)[:, 1]
        pred = (proba >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
        return {
            "name": name, "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
            "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
            "roc_auc": roc_auc_score(y_test, proba),
        }

    results = []
    lr = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X_train_s, y_train)
    results.append(evaluate("Logistic Regression", lr, X_test_s))
    rf = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5,
                                 class_weight="balanced", random_state=args.seed).fit(X_train, y_train)
    results.append(evaluate("Random Forest", rf, X_test))
    pos_weight = (len(y_train) - y_train.sum()) / max(1, y_train.sum())
    xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.08, subsample=0.9,
                         colsample_bytree=0.9, scale_pos_weight=pos_weight, eval_metric="logloss",
                         random_state=args.seed).fit(X_train, y_train)
    results.append(evaluate("XGBoost", xgb, X_test))
    selected = max(results, key=lambda r: (r["recall"], r["roc_auc"]))["name"]

    lines18 = [f"Table 18. Predictive Model Performance Comparison (Test Set, n = {len(y_test)})", ""]
    header = f"{'Model':<24}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}{'F1-Score':>10}{'Specificity':>13}{'ROC-AUC':>10}"
    lines18.append(header)
    for r in results:
        label = r["name"] + (" (selected)" if r["name"] == selected else "")
        lines18.append(f"{label:<24}{r['accuracy']:>10.3f}{r['precision']:>11.3f}{r['recall']:>9.3f}"
                       f"{r['f1']:>10.3f}{r['specificity']:>13.4f}{r['roc_auc']:>10.3f}")
    _write(os.path.join(tables_dir, "table18_model_comparison.txt"), "\n".join(lines18) + "\n")

    return {"X_test": X_test, "y_test": y_test, "rf": rf, "xgb": xgb, "selected": selected}


# ---------------------------------------------------------------------------
# Table 19, Table 20, Figure 7: built on the saved (production) model
# ---------------------------------------------------------------------------
def tables_19_20_figure_07(children, args, tables_dir, figures_dir, plt):
    from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    rows, _ = dp.generate_training_dataset(args.n_synthetic, args.seed, children)
    source = np.array(["Registry" if r["source"] == "real" else "Synthetic" for r in rows])
    age_days = np.array([r["age_days"] for r in rows])
    X = np.array([[r[c] for c in dp.FEATURE_COLUMNS] for r in rows], dtype=float)
    y = np.array([1 if r["label"] == "At-Risk" else 0 for r in rows], dtype=int)
    idx = np.arange(len(rows))

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, idx, test_size=0.2, random_state=args.seed, stratify=y)
    rf = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5,
                                 class_weight="balanced", random_state=args.seed).fit(X_train, y_train)
    proba_test = rf.predict_proba(X_test)[:, 1]
    pred_test = (proba_test >= 0.5).astype(int)

    # Table 19
    report_text = classification_report(y_test, pred_test, target_names=["Not At-Risk", "At-Risk"], digits=3)
    _write(os.path.join(tables_dir, "table19_classification_report.txt"),
          f"Table 19. Classification Report of the Selected Model (Test Set, n = {len(y_test)})\n\n{report_text}")

    # Figure 7: confusion matrix
    cm = confusion_matrix(y_test, pred_test)
    fig, ax = plt.subplots(figsize=(5, 5), dpi=200)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not At-Risk", "At-Risk"]).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("Confusion Matrix - Random Forest (selected)")
    fig.tight_layout()
    path = os.path.join(figures_dir, "figure07_confusion_matrix.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")

    # Extra: feature importance chart (not a numbered paper table/figure --
    # this was the groupmate's original "feature importance" ask, kept
    # alongside the numbered outputs for convenience)
    importance = dict(zip(dp.FEATURE_COLUMNS, rf.feature_importances_.tolist()))
    items = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)
    fig, ax = plt.subplots(figsize=(7, 5), dpi=200)
    ax.barh([k for k, v in items][::-1], [v for k, v in items][::-1], color="#2563eb")
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance - Random Forest (selected)")
    fig.tight_layout()
    path = os.path.join(figures_dir, "extra_feature_importance.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")

    # Extra: ROC curve + precision-recall curve (the groupmate's original
    # "supplementary evaluation" ask -- distinct from Table 20 above, which
    # the paper itself calls "Supplementary Evaluation of the Predictive
    # Model" for a different, 7-row breakdown)
    from sklearn.metrics import roc_curve, auc as sk_auc, precision_recall_curve
    fpr, tpr, _ = roc_curve(y_test, proba_test)
    roc_auc_val = sk_auc(fpr, tpr)
    prec_curve, rec_curve, _ = precision_recall_curve(y_test, proba_test)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=200)
    axes[0].plot(fpr, tpr, color="#2563eb", label=f"AUC = {roc_auc_val:.3f}")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="#9ca3af")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend(loc="lower right")
    axes[1].plot(rec_curve, prec_curve, color="#16a34a")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    fig.suptitle("ROC and Precision-Recall Curves - Random Forest (selected)")
    fig.tight_layout()
    path = os.path.join(figures_dir, "extra_roc_precision_recall.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")

    # Extra: worked example predictions -- a handful of actual test-set
    # children (not the constructed A-E profiles in Table 21), with the
    # features that drove each prediction
    n_examples = min(8, len(y_test))
    sample_idx = np.random.RandomState(args.seed).choice(len(y_test), size=n_examples, replace=False)
    labels = ["Not At-Risk", "At-Risk"]
    ex_lines = ["Worked Example Predictions - Random Forest (selected)", "=" * 60, ""]
    for i in sample_idx:
        row = rows[idx_test[i]]
        actual = labels[y_test[i]]
        predicted = labels[pred_test[i]]
        mark = "CORRECT" if actual == predicted else "WRONG"
        ex_lines.append(f"[{mark}] Barangay: {row['barangay']}")
        ex_lines.append(f"  Predicted: {predicted}  ({proba_test[i] * 100:.1f}% risk)   Actual: {actual}")
        ex_lines.append(
            f"  doses_missed={row['doses_missed']:.0f}  completion_rate={row['completion_rate']:.2f}  "
            f"avg_delay_days={row['avg_delay_days']:.1f}  days_since_last_dose={row['days_since_last_dose']:.0f}")
        ex_lines.append("")
    _write(os.path.join(tables_dir, "extra_worked_example_predictions.txt"), "\n".join(ex_lines))

    # Table 20: supplementary evaluation
    def metrics(mask, label):
        y_sub, pred_sub, proba_sub = y_test[mask], pred_test[mask], proba_test[mask]
        n_at_risk = int(y_sub.sum())
        tn, fp, fn, tp = confusion_matrix(y_sub, pred_sub, labels=[0, 1]).ravel()
        spec = tn / (tn + fp) if (tn + fp) else float("nan")
        try:
            auc = roc_auc_score(y_sub, proba_sub)
        except ValueError:
            auc = float("nan")
        acc = (pred_sub == y_sub).mean()
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        return (label, len(y_sub), n_at_risk, acc, prec, rec, spec, auc)

    src_test = source[idx_test]
    age_test = age_days[idx_test]
    rows20 = [
        metrics(src_test == "Registry", "Selected model, registry children in test set only"),
        metrics(src_test == "Synthetic", "Selected model, synthetic children in test set only"),
        metrics((age_test >= 30) & (age_test <= 120), "Selected model, assessed at 30-120 days"),
        metrics((age_test >= 121) & (age_test <= 240), "Selected model, assessed at 121-240 days"),
        metrics((age_test >= 241) & (age_test <= 365), "Selected model, assessed at 241-365 days"),
    ]

    # Row 6: RF trained on synthetic only, tested on all registry children
    syn_mask_all = source == "Synthetic"
    reg_mask_all = source == "Registry"
    from sklearn.ensemble import RandomForestClassifier as RF2
    rf_syn = RF2(n_estimators=300, max_depth=8, min_samples_leaf=5,
                 class_weight="balanced", random_state=args.seed).fit(X[syn_mask_all], y[syn_mask_all])
    proba_reg = rf_syn.predict_proba(X[reg_mask_all])[:, 1]
    pred_reg = (proba_reg >= 0.5).astype(int)
    y_reg = y[reg_mask_all]
    tn, fp, fn, tp = confusion_matrix(y_reg, pred_reg, labels=[0, 1]).ravel()
    rows20.append((
        "Random Forest trained on synthetic data only, tested on all registry children",
        len(y_reg), int(y_reg.sum()), (pred_reg == y_reg).mean(),
        tp / (tp + fp) if (tp + fp) else 0.0, tp / (tp + fn) if (tp + fn) else 0.0,
        tn / (tn + fp) if (tn + fp) else 0.0, roc_auc_score(y_reg, proba_reg),
    ))

    # Row 7: rule-based scorer on the same 938-row test set
    test_rows = [rows[i] for i in idx_test]
    rule_scores = np.array([dp.score_child(r) for r in test_rows])
    rule_pred = (rule_scores >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, rule_pred, labels=[0, 1]).ravel()
    rows20.append((
        "Rule-based scorer (fallback), full test set",
        len(y_test), int(y_test.sum()), (rule_pred == y_test).mean(),
        tp / (tp + fp) if (tp + fp) else 0.0, tp / (tp + fn) if (tp + fn) else 0.0,
        tn / (tn + fp) if (tn + fp) else 0.0, roc_auc_score(y_test, rule_scores),
    ))

    lines20 = ["Table 20. Supplementary Evaluation of the Predictive Model", ""]
    for label, n, n_at_risk, acc, prec, rec, spec, auc in rows20:
        lines20.append(f"{label}")
        lines20.append(f"  n(At-Risk)={n_at_risk:<6} accuracy={acc:.3f}  precision={prec:.3f}  "
                       f"recall={rec:.3f}  specificity={spec:.3f}  roc_auc={auc:.3f}")
    _write(os.path.join(tables_dir, "table20_supplementary_evaluation.txt"), "\n".join(lines20) + "\n")


# ---------------------------------------------------------------------------
# Table 21: example prediction profiles, scored by the saved production model
# ---------------------------------------------------------------------------
def table_21(args, tables_dir, assessment_date):
    BARANGAY = "Awai"

    def _profile(dob, given_offsets, date_registered=None):
        doses = {code: None for code in dp.VACCINE_CODES}
        for code, offset in given_offsets.items():
            doses[code] = dob + timedelta(days=offset)
        return dob, (date_registered or dob), doses

    schedule = {code: due_days for code, _, _, _, due_days in dp.VACCINE_SCHEDULE}

    dob_a = assessment_date - timedelta(days=122)
    due_a = [c for c, d in schedule.items() if d <= (assessment_date - dob_a).days]
    dob_a, reg_a, doses_a = _profile(dob_a, {c: schedule[c] for c in due_a})

    dob_b = assessment_date - timedelta(days=122)
    dob_b, reg_b, doses_b = _profile(dob_b, {"BCG": 0, "HEPA-BD": 0})

    dob_c = assessment_date - timedelta(days=215)
    due_c = [c for c, d in schedule.items() if d <= (assessment_date - dob_c).days]
    dob_c, reg_c, doses_c = _profile(
        dob_c, {c: schedule[c] + 40 for c in due_c}, date_registered=dob_c + timedelta(days=20))

    dob_d = assessment_date - timedelta(days=334)
    dob_d, reg_d, doses_d = _profile(
        dob_d, {"BCG": 0, "HEPA-BD": 0, "PENTA1": 42, "OPV1": 42, "PCV1": 42})

    dob_e = assessment_date - timedelta(days=730)
    due_e = [c for c, d in schedule.items() if c not in ("MMR1", "MMR2")]
    dob_e, reg_e, doses_e = _profile(dob_e, {c: schedule[c] for c in due_e})

    profiles = [
        ("A. 4 months old, all due doses on time", dob_a, reg_a, doses_a),
        ("B. 4 months old, birth doses only", dob_b, reg_b, doses_b),
        ("C. 7 months old, every dose about 40 days late", dob_c, reg_c, doses_c),
        ("D. 11 months old, stopped after the 6-week visit", dob_d, reg_d, doses_d),
        ("E. 2 years old, both MMR doses missing", dob_e, reg_e, doses_e),
    ]

    lines = ["Table 21. Example Prediction Results", ""]
    for name, dob, reg, doses_dict in profiles:
        dose_records = dp._dose_records_for(doses_dict)
        age_days = (assessment_date - dob).days
        feats = dp.compute_features(dob, reg, "F", dose_records, assessment_date,
                                     barangay_risk_rate=dp.BARANGAY_BASE_RISK.get(BARANGAY, 0.15))
        due, given, missed = feats["doses_due"], feats["doses_completed"], feats["doses_missed"]

        if dp.model_is_trained() and dp.ML_ASSESSMENT_FLOOR_DAYS <= age_days <= dp.CONTINUATION_WINDOW_DAYS:
            ml_result = dp._predict_with_model(feats)
        else:
            ml_result = None
        if ml_result is not None:
            score, artifact = ml_result
            engine = artifact.get("model_name", "ml")
        else:
            score = dp.score_child(feats)
            engine = "Rule-based scorer"
        label = "At-Risk" if score >= dp.RISK_THRESHOLD else "Not At-Risk"
        lines.append(f"{name}")
        lines.append(f"  Doses Due/Given/Missed: {due}/{given}/{missed}  Engine: {engine}  "
                     f"Result: {label} ({score:.3f})")
    _write(os.path.join(tables_dir, "table21_example_predictions.txt"), "\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="paper_artifacts")
    parser.add_argument("--n-synthetic", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--assessment-date", default="2026-10-10",
                        help="YYYY-MM-DD reference date used throughout the paper")
    parser.add_argument("--skip-model", action="store_true",
                        help="Skip Tables 17-21 and Figure 7 (data-only tables/figures run fast; "
                             "model training takes longer)")
    args = parser.parse_args()

    assessment_date = date.fromisoformat(args.assessment_date)
    tables_dir = os.path.join(args.out, "tables")
    figures_dir = os.path.join(args.out, "figures")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print("Loading real registry children from data/raw/*.csv ...")
    children = dp.load_and_merge_real_registries()
    print(f"Loaded {len(children)} children.\n")

    print("Data-only tables:")
    table_09(children, assessment_date, tables_dir)
    table_11(children, assessment_date, tables_dir)
    table_13(children, assessment_date, tables_dir)
    table_14(children, assessment_date, tables_dir)
    table_15(children, assessment_date, tables_dir)

    print("\nFigures:")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure_03(children, figures_dir, plt)
    figure_04(children, figures_dir, plt)
    figure_05(children, assessment_date, figures_dir, plt)
    figure_06(children, assessment_date, figures_dir, plt)

    if args.skip_model:
        print("\n--skip-model passed: skipping Tables 17-21 and Figure 7.")
        return

    print("\nModel-dependent tables (this retrains Logistic Regression / Random Forest / "
         "XGBoost a few times - may take a minute):")
    tables_17_18(children, args, tables_dir)

    print("\nTraining and saving the production model (data/risk_model.joblib) for "
         "Tables 19-21 and Figure 7 ...")
    dp.train_and_save_model(n_synthetic=args.n_synthetic, seed=args.seed, use_real_data=True)

    tables_19_20_figure_07(children, args, tables_dir, figures_dir, plt)
    table_21(args, tables_dir, assessment_date)

    print(f"\nDone. Everything is under {args.out}/")


if __name__ == "__main__":
    main()
