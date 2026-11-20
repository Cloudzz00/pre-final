"""Prints the predictive model's full performance report in the terminal,
with every table the manuscript and the defense rely on (manuscript table
numbers in brackets):

   1. Dataset: class distribution before/after balancing [Table 17],
      training/testing split [Table 19], real vs synthetic, class balance
   2. Model configuration [Table 18]
   3. Testing performance [Table 20], confusion matrix [Figure 7],
      classification report [Table 21], results by data source and class,
      Random Forest vs XGBoost
   4. Training vs testing (overfitting check)
   5. 5-fold cross-validation
   6. Feature importance [Figure 8]
   7. Supplementary evaluation [Table 22, upper part]
   8. Training-data comparison on real children [Table 22, lower part]
   9. Dropout-only (WHO) target test
  10. Example predictions [Table 23]
  11. Deployed model check (data/risk_model.joblib)

It retrains the three candidate models with exactly the settings used by
data_processor.train_and_save_model() (same data, split and seed), but never
overwrites the saved model.

Run from the ImmunoVision folder:
    python model_report.py
    python model_report.py --no-color      # plain output, e.g. for copying
    python model_report.py --skip-slow     # skip sections 8-9 (about 2 minutes)
"""
import argparse
import contextlib
import io
import os
import sys
import warnings

warnings.filterwarnings("ignore")

SEED = 7
METRICS = ["Accuracy", "Precision", "Recall", "F1-Score", "Specificity", "ROC-AUC"]


class Style:
    def __init__(self, enabled):
        self.enabled = enabled

    def _wrap(self, code, text):
        return f"\033[{code}m{text}\033[0m" if self.enabled else text

    def bold(self, t): return self._wrap("1", t)
    def dim(self, t): return self._wrap("2", t)
    def cyan(self, t): return self._wrap("36;1", t)
    def green(self, t): return self._wrap("32;1", t)
    def yellow(self, t): return self._wrap("33;1", t)


def visible_len(text):
    import re
    return len(re.sub(r"\033\[[0-9;]*m", "", text))


def pad(text, width, align):
    gap = width - visible_len(text)
    if align == "r":
        return " " * gap + text
    if align == "c":
        return " " * (gap // 2) + text + " " * (gap - gap // 2)
    return text + " " * gap


def table(headers, rows, aligns=None, highlight=None, style=None):
    """Box-drawn table. `highlight` is a set of row indexes to print in bold."""
    aligns = aligns or ["l"] + ["r"] * (len(headers) - 1)
    highlight = highlight or set()
    widths = [max(visible_len(str(x)) for x in [h] + [r[i] for r in rows if r is not None])
              for i, h in enumerate(headers)]
    line = lambda l, m, r: l + m.join("─" * (w + 2) for w in widths) + r
    out = [line("┌", "┬", "┐"),
           "│" + "│".join(" " + pad(style.bold(h), w, "c") + " " for h, w in zip(headers, widths)) + "│",
           line("├", "┼", "┤")]
    for i, r in enumerate(rows):
        if r is None:  # separator
            out.append(line("├", "┼", "┤"))
            continue
        cells = [str(x) for x in r]
        if i in highlight:
            cells = [style.green(c) for c in cells]
        out.append("│" + "│".join(" " + pad(c, w, a) + " " for c, w, a in zip(cells, widths, aligns)) + "│")
    out.append(line("└", "┴", "┘"))
    return "\n".join(out)


def section(title, style):
    bar = "═" * 78
    return f"\n{style.cyan(bar)}\n {style.cyan(title)}\n{style.cyan(bar)}"


def main():
    parser = argparse.ArgumentParser(description="ImmunoVision predictive model performance report")
    parser.add_argument("--no-color", action="store_true", help="disable colored output")
    parser.add_argument("--skip-slow", action="store_true",
                        help="skip the training-data comparison and dropout test (sections 8-9)")
    args = parser.parse_args()
    style = Style(enabled=not args.no_color and sys.stdout.isatty() and os.environ.get("NO_COLOR") is None)

    import random
    from datetime import date, timedelta
    import numpy as np
    import joblib
    import data_processor as dp
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                                 recall_score, roc_auc_score)
    from sklearn.model_selection import StratifiedKFold, train_test_split
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier

    print(style.bold("\nImmunoVision — Predictive Model Performance Report"))
    print(style.dim("Rebuilding the training dataset and retraining the candidate models..."))

    with contextlib.redirect_stdout(io.StringIO()):  # silence import notes from the loader
        real = dp.load_and_merge_real_registries()
        rows = dp.generate_training_dataset(3000, SEED, real)
        rows_unbalanced = dp.generate_training_dataset(3000, SEED, real, target_share=None)
    src_all = np.array([1 if r["source"] == "real" else 0 for r in rows])  # 1 = real, 0 = synthetic
    n_real, n_syn = int(src_all.sum()), int(len(rows) - src_all.sum())
    n_excluded = len(real) - n_real  # real children without a complete first year

    X = np.array([[r[c] for c in dp.FEATURE_COLUMNS] for r in rows], dtype=float)
    y = np.array([1 if r["label"] == "At-Risk" else 0 for r in rows], dtype=int)
    X_tr, X_te, y_tr, y_te, s_tr, s_te, i_tr, i_te = train_test_split(
        X, y, src_all, np.arange(len(rows)), test_size=0.2, random_state=SEED, stratify=y)
    scaler = StandardScaler().fit(X_tr)

    def evaluate(model, Xa, ya):
        proba = model.predict_proba(Xa)[:, 1]
        pred = (proba >= dp.RISK_THRESHOLD).astype(int)
        tn, fp, fn, tp = confusion_matrix(ya, pred, labels=[0, 1]).ravel()
        return {"Accuracy": accuracy_score(ya, pred), "Precision": precision_score(ya, pred, zero_division=0),
                "Recall": recall_score(ya, pred, zero_division=0), "F1-Score": f1_score(ya, pred, zero_division=0),
                "Specificity": tn / (tn + fp) if (tn + fp) else 0.0, "ROC-AUC": roc_auc_score(ya, proba),
                "cm": (tn, fp, fn, tp), "proba": proba}

    def build(name, y_fit):
        if name == "Logistic Regression":
            return LogisticRegression(max_iter=1000, class_weight="balanced")
        if name == "Random Forest":
            return RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5,
                                          class_weight="balanced", random_state=SEED)
        pos_weight = (len(y_fit) - y_fit.sum()) / max(1, y_fit.sum())
        return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.08, subsample=0.9,
                             colsample_bytree=0.9, scale_pos_weight=pos_weight, eval_metric="logloss",
                             random_state=SEED)

    names = ["Logistic Regression", "Random Forest", "XGBoost"]
    scaled = {"Logistic Regression"}
    results = {}
    for name in names:
        A, B = (scaler.transform(X_tr), scaler.transform(X_te)) if name in scaled else (X_tr, X_te)
        model = build(name, y_tr).fit(A, y_tr)
        results[name] = {"train": evaluate(model, A, y_tr), "test": evaluate(model, B, y_te), "model": model}

    selected = max(names, key=lambda n: (results[n]["test"]["Recall"], results[n]["test"]["ROC-AUC"]))

    cv = {}
    for name in names:
        folds = []
        for fit_idx, val_idx in StratifiedKFold(n_splits=5).split(X_tr, y_tr):
            A, B = X_tr[fit_idx], X_tr[val_idx]
            if name in scaled:
                s = StandardScaler().fit(A)
                A, B = s.transform(A), s.transform(B)
            folds.append(evaluate(build(name, y_tr[fit_idx]).fit(A, y_tr[fit_idx]), B, y_tr[val_idx]))
        cv[name] = folds

    f3 = lambda v: f"{v:.3f}"

    # 1. Dataset ---------------------------------------------------------------
    print(section("1. Dataset and Partitioning", style))

    def label_counts(rs, source):
        at = sum(1 for r in rs if r["source"] == source and r["label"] == "At-Risk")
        return at, sum(1 for r in rs if r["source"] == source) - at
    real_at, real_ok = label_counts(rows, "real")
    syn_at, syn_ok = label_counts(rows_unbalanced, "synthetic")
    removed = len(rows_unbalanced) - len(rows)
    print("  Class distribution before and after data balancing [Table 17]")
    print(table(["Dataset", "At-Risk", "Not At-Risk", "Total"], [
        ["Eligible real records", f"{real_at:,}", f"{real_ok:,}", f"{real_at + real_ok:,}"],
        ["Synthetic records", f"{syn_at:,}", f"{syn_ok:,}", f"{syn_at + syn_ok:,}"],
        ["Combined dataset before balancing", f"{real_at + syn_at:,}", f"{real_ok + syn_ok:,}",
         f"{len(rows_unbalanced):,}"],
        ["Removed by random undersampling (synthetic Not At-Risk only)", "0", f"-{removed:,}", f"-{removed:,}"],
        None,
        ["Final dataset after balancing", f"{y.sum():,}", f"{len(y) - y.sum():,}", f"{len(y):,}"],
    ], highlight={5}, style=style))
    print(style.dim(f"  Eligible = registry children whose first year is complete in the data: {n_real:,} of {len(real):,} "
                    f"({n_excluded:,} younger children excluded).\n  Removed rows are chosen at random with a fixed seed "
                    "and are not used for training or testing; no real record is removed."))

    print("\n  Training and testing split [Table 19]")
    print(table(["Set", "Records", "At-Risk", "Not At-Risk"], [
        ["Training (80%)", f"{len(y_tr):,}", f"{y_tr.sum():,}", f"{len(y_tr) - y_tr.sum():,}"],
        ["Testing (20%)", f"{len(y_te):,}", f"{y_te.sum():,}", f"{len(y_te) - y_te.sum():,}"],
        None,
        ["Total", f"{len(y):,}", f"{y.sum():,}", f"{len(y) - y.sum():,}"],
    ], style=style))
    print(style.dim(f"  Sources: {n_real:,} RHU registry children (complete first year; {n_excluded:,} younger children "
                    f"excluded) + {n_syn:,} synthetic children, balanced to {100 * y.mean():.0f}% At-Risk.\n  "
                    f"Stratified split, random_state = {SEED}.\n"
                    f"  Threshold: probability ≥ {dp.RISK_THRESHOLD:.2f} = At-Risk. "
                    "Selection rule: highest testing recall, ties broken by ROC-AUC."))

    def by_source(ys, src):
        out = []
        for label, mask in (("Real", src == 1), ("Synthetic", src == 0), ("Combined", np.ones(len(src), bool))):
            out.append([label, f"{mask.sum():,}", f"{ys[mask].sum():,}", f"{(ys[mask] == 0).sum():,}"])
        return out

    print("\n  Real and synthetic children in each set")
    print(table(["Set / Source", "Records", "At-Risk", "Not At-Risk"],
                [["Training (80%)", "", "", ""]] + [["  " + r[0]] + r[1:] for r in by_source(y_tr, s_tr)]
                + [None, ["Testing (20%)", "", "", ""]] + [["  " + r[0]] + r[1:] for r in by_source(y_te, s_te)]
                + [None, ["All rows", "", "", ""]]
                + [["  " + r[0]] + r[1:] for r in by_source(y, src_all)],
                style=style))
    print(style.dim("  Real and synthetic rows are mixed first, then split once; stratify keeps the At-Risk share "
                    "the same in both sets."))

    def balance_row(label, ys):
        pos, neg = int(ys.sum()), int(len(ys) - ys.sum())
        big, small = max(pos, neg), max(1, min(pos, neg))
        minority = "None (balanced)" if pos == neg else ("At-Risk" if pos < neg else "Not At-Risk")
        return [label, f"{pos:,} ({100 * pos / len(ys):.1f}%)", f"{neg:,} ({100 * neg / len(ys):.1f}%)",
                f"{big / small:.1f} : 1", minority]

    print("\n  Class balance")
    print(table(["Data", "At-Risk", "Not At-Risk", "Ratio", "Minority class"], [
        balance_row("Real registry only", y[src_all == 1]),
        balance_row("Synthetic only", y[src_all == 0]),
        balance_row("Combined (all rows)", y),
        None,
        balance_row("Training set", y_tr),
        balance_row("Testing set", y_te),
    ], style=style))
    pos_w = (len(y_tr) - y_tr.sum()) / max(1, y_tr.sum())
    w_pos = len(y_tr) / (2 * y_tr.sum())
    w_neg = len(y_tr) / (2 * (len(y_tr) - y_tr.sum()))
    combined = "balanced" if abs(y.mean() - 0.5) < 0.005 else "mildly imbalanced"
    print(style.dim(f"  Real data alone is severely imbalanced; the combined training data is {combined}.\n"
                    f"  Class weights: balanced weights At-Risk {w_pos:.2f} vs Not At-Risk {w_neg:.2f} (Logistic Regression,\n"
                    f"  Random Forest), scale_pos_weight {pos_w:.2f} (XGBoost); stratified split keeps the same mix in both sets."))

    # 2. Configuration ---------------------------------------------------------
    print(section("2. Model Configuration [Table 18]", style))
    print(table(["Component", "Configuration"], [
        ["Target variable", "12-month outcome: At-Risk if < 80% of first-year doses on time (within "
                            f"{dp.GRACE_PERIOD_DAYS} days) or 2+ never given"],
        [f"Input features ({len(dp.FEATURE_COLUMNS)})", ", ".join(dp.FEATURE_COLUMNS[:5]) + ","],
        ["", ", ".join(dp.FEATURE_COLUMNS[5:])],
        ["Not used as inputs", "sex, barangay risk rate (recorded, not model features)"],
        ["Assessment window", f"{dp.ML_ASSESSMENT_FLOOR_DAYS} to {dp.CONTINUATION_WINDOW_DAYS} days of age"],
        ["Training data", f"{len(y):,} rows: {n_real:,} registry + {n_syn:,} synthetic, "
                          f"{100 * y.mean():.0f}% At-Risk"],
        ["Data split", f"80% training / 20% testing, stratified (random_state = {SEED})"],
        ["Logistic Regression", "balanced class weights, standardized inputs, max_iter 1000"],
        ["Random Forest", "300 trees, max depth 8, min 5 samples per leaf, balanced class weights"],
        ["XGBoost", "300 trees, max depth 4, learning rate 0.08, subsample 0.9, scale_pos_weight"],
        ["Decision threshold", f"probability >= {dp.RISK_THRESHOLD:.2f} = At-Risk (>= 0.70 = High Risk)"],
        ["Selection rule", "highest testing recall; ties broken by ROC-AUC"],
        ["Validation", "5-fold stratified cross-validation on the training set"],
        ["Fallback", "rule-based scorer (6 weighted signals) outside the assessment window"],
    ], aligns=["l", "l"], style=style))

    # 3. Testing ---------------------------------------------------------------
    print(section("3. Testing Set Performance (held out, n = {:,}) [Table 20]".format(len(y_te)), style))
    rows_t = [[n + (" (selected)" if n == selected else "")] + [f3(results[n]["test"][m]) for m in METRICS]
              for n in names]
    print(table(["Model"] + METRICS, rows_t, highlight={names.index(selected)}, style=style))
    print(style.dim("  Precision, recall and F1-score refer to the At-Risk class."))

    tn, fp, fn, tp = results[selected]["test"]["cm"]
    print(f"\n  Confusion matrix — {style.bold(selected)} (testing set) [Figure 7]")
    print(table(["", "Predicted Not At-Risk", "Predicted At-Risk"], [
        [f"Actual Not At-Risk ({tn + fp})", f"{tn} (TN)", f"{fp} (FP)"],
        [f"Actual At-Risk ({fn + tp})", f"{fn} (FN)", f"{tp} (TP)"],
    ], aligns=["l", "c", "c"], style=style))

    def prf(tp_, fp_, fn_):
        p_ = tp_ / (tp_ + fp_) if (tp_ + fp_) else 0.0
        r_ = tp_ / (tp_ + fn_) if (tp_ + fn_) else 0.0
        return p_, r_, (2 * p_ * r_ / (p_ + r_) if (p_ + r_) else 0.0)
    neg = prf(tn, fn, fp)   # Not At-Risk as the class of interest
    pos = prf(tp, fp, fn)   # At-Risk
    n_neg, n_pos = tn + fp, fn + tp
    macro = [(a + b) / 2 for a, b in zip(neg, pos)]
    weighted = [(a * n_neg + b * n_pos) / (n_neg + n_pos) for a, b in zip(neg, pos)]
    print(f"\n  Classification report — {style.bold(selected)} (testing set) [Table 21]")
    print(table(["Class", "Precision", "Recall", "F1-Score", "Support"], [
        ["Not At-Risk"] + [f3(v) for v in neg] + [f"{n_neg:,}"],
        ["At-Risk"] + [f3(v) for v in pos] + [f"{n_pos:,}"],
        None,
        ["Accuracy", "", "", f3((tn + tp) / len(y_te)), f"{len(y_te):,}"],
        ["Macro average"] + [f3(v) for v in macro] + [f"{len(y_te):,}"],
        ["Weighted average"] + [f3(v) for v in weighted] + [f"{len(y_te):,}"],
    ], style=style))

    sel = results[selected]["test"]["proba"]
    src_rows = []
    for label, mask in (("Real children only", s_te == 1), ("Synthetic children only", s_te == 0),
                        ("Combined (all testing)", np.ones(len(s_te), bool))):
        yt, pt = y_te[mask], sel[mask]
        d = (pt >= dp.RISK_THRESHOLD).astype(int)
        tn_, fp_, fn_, tp_ = confusion_matrix(yt, d, labels=[0, 1]).ravel()
        src_rows.append([label, f"{mask.sum():,}", f3(accuracy_score(yt, d)), f3(precision_score(yt, d, zero_division=0)),
                         f3(recall_score(yt, d, zero_division=0)), f3(tn_ / (tn_ + fp_) if (tn_ + fp_) else 0.0),
                         f3(roc_auc_score(yt, pt)), f"{tp_}/{tp_ + fn_}", f"{tn_}/{tn_ + fp_}"])
    print(f"\n  Testing performance by data source — {style.bold(selected)}")
    print(table(["Testing children", "n", "Accuracy", "Precision", "Recall", "Specificity", "ROC-AUC",
                 "At-Risk found", "On-track cleared"], src_rows, highlight={0}, style=style))
    print(style.dim("  The real-only row is the best estimate of performance on actual RHU records."))

    tr = results[selected]["test"]
    print(f"\n  Performance on each class — {style.bold(selected)} (testing set)")
    print(table(["Class", "Children", "Correctly classified", "Rate"], [
        ["At-Risk", f"{fn + tp:,}", f"{tp:,}", f"{f3(tr['Recall'])} (recall)"],
        ["Not At-Risk", f"{tn + fp:,}", f"{tn:,}", f"{f3(tr['Specificity'])} (specificity)"],
    ], style=style))
    print(style.dim("  Both rates are high, so the model is not biased toward either class."))

    ens = [n for n in ("Random Forest", "XGBoost") if n in results]
    if len(ens) == 2:
        pa = results[ens[0]]["test"]["proba"] >= dp.RISK_THRESHOLD
        pb = results[ens[1]]["test"]["proba"] >= dp.RISK_THRESHOLD
        print("\n  Random Forest vs XGBoost on the testing set")
        print(table(["Model", "At-Risk found", "Missed", "False alarms", "Correctly cleared"],
                    [[n, f"{results[n]['test']['cm'][3]}", f"{results[n]['test']['cm'][2]}",
                      f"{results[n]['test']['cm'][1]}", f"{results[n]['test']['cm'][0]}"] for n in ens],
                    highlight={ens.index(selected)} if selected in ens else None, style=style))
        print(style.dim(f"  The two disagree on {int((pa != pb).sum())} of {len(y_te)} children: "
                        f"{int((pa & ~pb).sum())} flagged only by Random Forest, {int((pb & ~pa).sum())} only by XGBoost."))

    # 3. Training vs testing ---------------------------------------------------
    print(section("4. Training vs. Testing Performance (overfitting check)", style))
    rows_g, hl = [], set()
    for i, n in enumerate(names):
        if i:
            rows_g.append(None)
        tr, te = results[n]["train"], results[n]["test"]
        gap = max(abs(tr[m] - te[m]) for m in METRICS)
        if n == selected:
            hl |= {len(rows_g), len(rows_g) + 1}
        rows_g.append([n, "Training"] + [f3(tr[m]) for m in METRICS] + [""])
        rows_g.append(["", "Testing"] + [f3(te[m]) for m in METRICS] + [f3(gap)])
    print(table(["Model", "Set"] + METRICS + ["Max gap"], rows_g,
                aligns=["l", "l"] + ["r"] * (len(METRICS) + 1), highlight=hl, style=style))
    print(style.dim("  Max gap = largest training–testing difference across the six metrics."))

    # 4. Cross-validation ------------------------------------------------------
    print(section(f"5. Validation — 5-fold Stratified Cross-Validation ({selected})", style))
    folds = cv[selected]
    rows_f = [[str(k + 1)] + [f3(f[m]) for m in METRICS] for k, f in enumerate(folds)]
    rows_f.append(None)
    rows_f.append(["Mean (SD)"] + [f"{np.mean([f[m] for f in folds]):.3f} ({np.std([f[m] for f in folds]):.3f})"
                                    for m in METRICS])
    print(table(["Fold"] + METRICS, rows_f, highlight={len(rows_f) - 1}, style=style))

    print(f"\n  All candidate models — mean (SD) across folds")
    rows_c = [[n] + [f"{np.mean([f[m] for f in cv[n]]):.3f} ({np.std([f[m] for f in cv[n]]):.3f})"
                     for m in METRICS] for n in names]
    print(table(["Model"] + METRICS, rows_c, highlight={names.index(selected)}, style=style))

    # 6. Feature importance ----------------------------------------------------
    best = results[selected]["model"]
    if hasattr(best, "feature_importances_"):
        print(section(f"6. Feature Importance — {selected} [Figure 8]", style))
        imp = sorted(zip(dp.FEATURE_COLUMNS, best.feature_importances_), key=lambda kv: -kv[1])
        print(table(["Rank", "Feature", "Importance", "Share"],
                    [[str(k + 1), name, f3(v), f"{100 * v:.1f}%"] for k, (name, v) in enumerate(imp)],
                    aligns=["r", "l", "r", "r"], style=style))
        print(style.dim(f"  Top 3 together: {100 * sum(v for _n, v in imp[:3]):.1f}% of the model's decisions."))

    # 7. Supplementary evaluation ---------------------------------------------
    print(section("7. Supplementary Evaluation [Table 22, upper part]", style))

    def sub_metrics(label, yt, pt):
        d = (pt >= dp.RISK_THRESHOLD).astype(int)
        tn_, fp_, fn_, tp_ = confusion_matrix(yt, d, labels=[0, 1]).ravel()
        return [label, f"{int(yt.sum()):,}", f3(accuracy_score(yt, d)), f3(precision_score(yt, d, zero_division=0)),
                f3(recall_score(yt, d, zero_division=0)), f3(tn_ / (tn_ + fp_) if (tn_ + fp_) else 0.0),
                f3(roc_auc_score(yt, pt))]

    age_te = X_te[:, dp.FEATURE_COLUMNS.index("age_days")]
    supp = [sub_metrics("Selected model, registry children in the test set only", y_te[s_te == 1], sel[s_te == 1]),
            sub_metrics("Selected model, synthetic children in the test set only", y_te[s_te == 0], sel[s_te == 0])]
    for lo, hi in ((30, 120), (121, 240), (241, 365)):
        m = (age_te >= lo) & (age_te <= hi)
        supp.append(sub_metrics(f"Selected model, assessed at {lo}-{hi} days", y_te[m], sel[m]))
    syn_only = build("Random Forest", y[src_all == 0]).fit(X[src_all == 0], y[src_all == 0])
    supp.append(sub_metrics(f"Random Forest trained on synthetic data only, tested on all {n_real:,} registry children",
                            y[src_all == 1], syn_only.predict_proba(X[src_all == 1])[:, 1]))
    rule = np.array([dp.score_child(rows[i]) for i in i_te])
    supp.append(sub_metrics("Rule-based scorer (fallback), full test set", y_te, rule))
    print(table(["Evaluation", "n (At-Risk)", "Accuracy", "Precision", "Recall", "Specificity", "ROC-AUC"],
                supp, style=style))

    # 8-9. Comparisons on real children only (5-fold CV) ----------------------
    def given_by(c, code, days):
        d = c["doses"].get(code)
        return d is not None and (d - c["date_of_birth"]).days <= days

    last_dose = max(d for c in real for d in c["doses"].values() if d)
    eligible = [c for c in real if c["date_of_birth"] + timedelta(days=dp.CONTINUATION_WINDOW_DAYS) <= last_dose]

    def real_matrix(population, label_fn):
        rng = random.Random(SEED)  # each child assessed at one random age, as in training
        feats, labels = [], []
        for c in population:
            age = rng.randint(dp.ML_ASSESSMENT_FLOOR_DAYS, dp.CONTINUATION_WINDOW_DAYS)
            when = min(date(2026, 11, 8), c["date_of_birth"] + timedelta(days=age))
            f = dp.compute_features(c["date_of_birth"], c["date_registered"], dp._dose_records_for(c["doses"]), when)
            feats.append([f[k] for k in dp.FEATURE_COLUMNS])
            labels.append(1 if label_fn(c) else 0)
        return np.array(feats, float), np.array(labels, int)

    def cv_real(Xr, yr, option, Xs=None, ys=None):
        """5-fold CV on real children; synthetic or SMOTE rows are added to the
        training folds only, never to the test fold."""
        tot, aucs, probs = np.zeros(4, int), [], np.zeros(len(yr))
        for fit_idx, val_idx in StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xr, yr):
            A, b = Xr[fit_idx], yr[fit_idx]
            if option == "smote":
                A, b = SMOTE(random_state=SEED).fit_resample(A, b)
            elif option != "real":
                if option is not None:  # subsample synthetic Not At-Risk rows to reach the target share
                    need = int(round((b.sum() + ys.sum()) * (1 - option) / option)) - (len(b) - b.sum())
                    neg_idx = np.where(ys == 0)[0]
                    take = np.random.RandomState(SEED).choice(neg_idx, size=max(0, min(need, len(neg_idx))), replace=False)
                    keep = np.concatenate([np.where(ys == 1)[0], take])
                    A, b = np.vstack([A, Xs[keep]]), np.concatenate([b, ys[keep]])
                else:
                    A, b = np.vstack([A, Xs]), np.concatenate([b, ys])
            p = build("Random Forest", b).fit(A, b).predict_proba(Xr[val_idx])[:, 1]
            probs[val_idx] = p
            tot += confusion_matrix(yr[val_idx], (p >= dp.RISK_THRESHOLD).astype(int), labels=[0, 1]).ravel()
            aucs.append(roc_auc_score(yr[val_idx], p))
        tn_, fp_, fn_, tp_ = tot
        return [f3(tp_ / (tp_ + fn_)), f3(tn_ / (tn_ + fp_)), f3(tp_ / (tp_ + fp_) if tp_ + fp_ else 0),
                f3(f1_score(yr, (probs >= dp.RISK_THRESHOLD).astype(int))), f3(np.mean(aucs)),
                f3((tp_ + tn_) / len(yr)), f"{tp_:,}/{tp_ + fn_:,}", f"{fp_:,}/{tn_ + fp_:,}"]

    if args.skip_slow:
        print(style.dim("\n  Sections 8-9 skipped (--skip-slow)."))
    else:
        try:
            from imblearn.over_sampling import SMOTE
        except ImportError:
            SMOTE = None
        print(section("8. Training-Data Comparison on Real Children [Table 22, lower part]", style))
        current = lambda c: dp.final_outcome_label(c["date_of_birth"], dp._dose_records_for(c["doses"])) == "At-Risk"
        Xr, yr = real_matrix(eligible, current)
        with contextlib.redirect_stdout(io.StringIO()):
            syn_rows = dp.generate_training_dataset(3000, SEED, [], target_share=None)
        Xs = np.array([[r[k] for k in dp.FEATURE_COLUMNS] for r in syn_rows], float)
        ys = np.array([1 if r["label"] == "At-Risk" else 0 for r in syn_rows], int)
        options = [("Registry children only", "real"), ("Registry + all 3,000 synthetic", None),
                   ("Registry + synthetic 40/60", 0.40), ("Registry + synthetic 45/55", 0.45),
                   ("Registry + synthetic 50/50 (selected)", 0.50), ("Registry + synthetic 55/45", 0.55)]
        if SMOTE is not None:
            options.append(("Registry + SMOTE oversampling (50/50)", "smote"))
        comp = [[name] + cv_real(Xr, yr, opt, Xs, ys) for name, opt in options]
        print(table(["Training data", "Recall", "Specificity", "Precision", "F1-Score", "ROC-AUC", "Accuracy",
                     "At-Risk found", "False alarms"], comp, highlight={4}, style=style))
        print(style.dim(f"  5-fold CV on the {len(yr):,} registry children with a complete first year "
                        f"({int(yr.sum()):,} At-Risk, {int(len(yr) - yr.sum()):,} Not At-Risk). Synthetic or SMOTE rows are "
                        "added to the training folds only, so every option is judged on real records."
                        + ("" if SMOTE else "\n  SMOTE row skipped: install imbalanced-learn to include it.")))

        print(section("9. Dropout-Only Target Test (WHO definition, registry children only)", style))
        started = [c for c in eligible if given_by(c, "PENTA1", 365)]
        drop_rows = []
        for name, fn in (("Penta 1 given, MMR 1 not given by 12 months",
                          lambda c: not given_by(c, "MMR1", 365)),
                         ("Penta 1 given, Penta 3 not given by 12 months",
                          lambda c: not given_by(c, "PENTA3", 365))):
            Xd, yd = real_matrix(started, fn)
            res = cv_real(Xd, yd, "real")
            drop_rows.append([name, f"{len(yd):,}", f"{int(yd.sum()):,} ({100 * yd.mean():.1f}%)"] + res[:5])
        print(table(["Dropout target", "Children", "Dropouts", "Recall", "Specificity", "Precision", "F1-Score",
                     "ROC-AUC"], drop_rows, style=style))
        print(style.dim("  Random Forest, 5-fold CV, trained on registry data only. Dropout alone is much harder to "
                        "predict\n  from these records than on-time completion (recall about 0.9), so it is "
                        "recommended as future work."))

    # 10. Example predictions --------------------------------------------------
    print(section("10. Example Predictions (synthetic profiles, assessed 10 Oct 2026) [Table 23]", style))
    when = date(2026, 10, 10)
    sched = {code: age for code, _n, _v, _d, age in dp.VACCINE_SCHEDULE}

    def profile(age, given, registered_after=0):
        dob = when - timedelta(days=age)
        doses = {code: None for code in dp.VACCINE_CODES}
        doses.update({code: dob + timedelta(days=off) for code, off in given.items()})
        return dob, dob + timedelta(days=registered_after), doses

    profiles = [
        ("A. 4 months old, all due doses on time", profile(122, {c: a for c, a in sched.items() if a <= 122})),
        ("B. 4 months old, birth doses only", profile(122, {"BCG": 0, "HEPA-BD": 0})),
        ("C. 7 months old, every dose ~40 days late", profile(215, {c: a + 40 for c, a in sched.items() if a <= 215}, 20)),
        ("D. 11 months old, stopped after 6 weeks", profile(334, {"BCG": 0, "HEPA-BD": 0, "PENTA1": 42,
                                                                   "OPV1": 42, "PCV1": 42})),
        ("E. 2 years old, both MMR doses missing", profile(730, {c: a for c, a in sched.items()
                                                                 if c not in ("MMR1", "MMR2")})),
    ]
    ex = []
    for name, (dob, reg, doses) in profiles:
        f = dp.compute_features(dob, reg, dp._dose_records_for(doses), when)
        result = dp._predict_with_model(f) if (dp.model_is_trained()
                                               and dp.ML_ASSESSMENT_FLOOR_DAYS <= f["age_days"] <= dp.CONTINUATION_WINDOW_DAYS) else None
        if result is not None:
            score, art = result
            engine, factors = art.get("model_name", "model"), dp._explain(f, art)
        else:
            score, engine, factors = dp.score_child(f), "Rule-based scorer", dp._explain_rules(f)
        ex.append([name, f"{f['doses_due']}/{f['doses_completed']}/{f['doses_missed']}", engine,
                   f"{'At-Risk' if score >= dp.RISK_THRESHOLD else 'Not At-Risk'} ({score:.3f})",
                   "; ".join(x["factor"] for x in factors)])
    print(table(["Profile", "Due/Given/Missed", "Engine", "Result", "Top factors shown"], ex,
                aligns=["l", "c", "l", "l", "l"], style=style))
    print(style.dim("  Profile E is older than 12 months, so the rule-based scorer is used; that child appears on "
                    "the Vaccine Gaps page."))

    # 11. Deployed model check -------------------------------------------------
    print(section("11. Deployed Model Check (data/risk_model.joblib)", style))
    try:
        artifact = joblib.load(dp.MODEL_PATH)
        deployed = artifact["model"]
        Xd = scaler.transform(X_te) if artifact.get("needs_scaler") else X_te
        dep = evaluate(deployed, Xd, y_te)
        agree = int(((dep["proba"] >= dp.RISK_THRESHOLD) ==
                     (results[selected]["test"]["proba"] >= dp.RISK_THRESHOLD)).sum())
        print(table(["Deployed model"] + METRICS,
                    [[artifact.get("model_name", "?")] + [f3(dep[m]) for m in METRICS]], style=style))
        verdict = style.green("MATCH") if agree == len(y_te) else style.yellow("DIFFERS")
        print(f"  Same classification as the retrained {selected} on {agree} of {len(y_te)} "
              f"testing children: {verdict}")
    except FileNotFoundError:
        print(style.yellow("  No saved model found — run train_model.py first."))
    print()


if __name__ == "__main__":
    main()
