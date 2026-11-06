"""Prints the predictive model's performance report in the terminal:
dataset partitioning, training vs. testing performance, 5-fold
cross-validation, the selected model's confusion matrix, and a check that
the deployed model (data/risk_model.joblib) reproduces the testing results.

It retrains the three candidate models with exactly the settings used by
data_processor.train_and_save_model() (same data, split and seed), but never
overwrites the saved model.

Run from the ImmunoVision folder:
    python model_report.py
    python model_report.py --no-color      # plain output, e.g. for copying
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
    args = parser.parse_args()
    style = Style(enabled=not args.no_color and sys.stdout.isatty() and os.environ.get("NO_COLOR") is None)

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
        rows, _ = dp.generate_training_dataset(3000, SEED, real)
    n_real, n_syn = len(real), len(rows) - len(real)

    X = np.array([[r[c] for c in dp.FEATURE_COLUMNS] for r in rows], dtype=float)
    y = np.array([1 if r["label"] == "At-Risk" else 0 for r in rows], dtype=int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
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
        results[name] = {"train": evaluate(model, A, y_tr), "test": evaluate(model, B, y_te)}

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
    print(table(["Set", "Records", "At-Risk", "Not At-Risk"], [
        ["Training (80%)", f"{len(y_tr):,}", f"{y_tr.sum():,}", f"{len(y_tr) - y_tr.sum():,}"],
        ["Testing (20%)", f"{len(y_te):,}", f"{y_te.sum():,}", f"{len(y_te) - y_te.sum():,}"],
        None,
        ["Total", f"{len(y):,}", f"{y.sum():,}", f"{len(y) - y.sum():,}"],
    ], style=style))
    print(style.dim(f"  Sources: {n_real:,} RHU registry children + {n_syn:,} synthetic children. "
                    f"Stratified split, random_state = {SEED}.\n"
                    f"  Threshold: probability ≥ {dp.RISK_THRESHOLD:.2f} = At-Risk. "
                    "Selection rule: highest testing recall, ties broken by ROC-AUC."))

    # 2. Testing ---------------------------------------------------------------
    print(section("2. Testing Set Performance (held out, n = {:,})".format(len(y_te)), style))
    rows_t = [[n + (" (selected)" if n == selected else "")] + [f3(results[n]["test"][m]) for m in METRICS]
              for n in names]
    print(table(["Model"] + METRICS, rows_t, highlight={names.index(selected)}, style=style))
    print(style.dim("  Precision, recall and F1-score refer to the At-Risk class."))

    tn, fp, fn, tp = results[selected]["test"]["cm"]
    print(f"\n  Confusion matrix — {style.bold(selected)} (testing set)")
    print(table(["", "Predicted Not At-Risk", "Predicted At-Risk"], [
        [f"Actual Not At-Risk ({tn + fp})", f"{tn} (TN)", f"{fp} (FP)"],
        [f"Actual At-Risk ({fn + tp})", f"{fn} (FN)", f"{tp} (TP)"],
    ], aligns=["l", "c", "c"], style=style))

    # 3. Training vs testing ---------------------------------------------------
    print(section("3. Training vs. Testing Performance (overfitting check)", style))
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
    print(section(f"4. Validation — 5-fold Stratified Cross-Validation ({selected})", style))
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

    # 5. Deployed model check --------------------------------------------------
    print(section("5. Deployed Model Check (data/risk_model.joblib)", style))
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
