"""Reproduces Table 17 (Training and Testing Dataset Distribution) and
Table 18 (Predictive Model Performance Comparison) against the current,
full dataset — all 19 barangays, not the 13-barangay snapshot the paper
was originally written on.

Retrains all three candidates (Logistic Regression, Random Forest,
XGBoost) on the same 80/20 split train_and_save_model() uses, so Table 18
can include Specificity, which the saved model artifact doesn't carry for
the two candidates that weren't selected.

Run: python table_17_18.py [--n-synthetic 3000] [--seed 7]
"""
import argparse

import numpy as np

import data_processor as dp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-synthetic", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    real_children = dp.load_and_merge_real_registries()
    rows, _ = dp.generate_training_dataset(args.n_synthetic, args.seed, real_children)
    # Synthetic rows are appended first, then real children - see
    # generate_training_dataset() in data_processor.py.
    source = (["Synthetic"] * args.n_synthetic) + (["Registry"] * len(real_children))

    X = np.array([[r[c] for c in dp.FEATURE_COLUMNS] for r in rows], dtype=float)
    y = np.array([1 if r["label"] == "At-Risk" else 0 for r in rows], dtype=int)
    src = np.array(source)

    X_train, X_test, y_train, y_test, src_train, src_test = train_test_split(
        X, y, src, test_size=0.2, random_state=args.seed, stratify=y)

    def counts(y_arr, src_arr):
        from_registry = int((src_arr == "Registry").sum())
        from_synthetic = int((src_arr == "Synthetic").sum())
        at_risk = int(y_arr.sum())
        not_at_risk = int(len(y_arr) - at_risk)
        return len(y_arr), from_registry, from_synthetic, at_risk, not_at_risk

    print("Table 17. Training and Testing Dataset Distribution")
    print(f"{'Set':<16}{'Rows':>8}{'From Registry':>16}{'From Synthetic':>16}{'At-Risk':>10}{'Not At-Risk':>14}")
    tr = counts(y_train, src_train)
    te = counts(y_test, src_test)
    tot = counts(y, src)
    print(f"{'Training (80%)':<16}{tr[0]:>8}{tr[1]:>16}{tr[2]:>16}{tr[3]:>10}{tr[4]:>14}")
    print(f"{'Testing (20%)':<16}{te[0]:>8}{te[1]:>16}{te[2]:>16}{te[3]:>10}{te[4]:>14}")
    print(f"{'Total':<16}{tot[0]:>8}{tot[1]:>16}{tot[2]:>16}{tot[3]:>10}{tot[4]:>14}")
    print()

    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

    def evaluate(name, model, Xte):
        proba = model.predict_proba(Xte)[:, 1]
        pred = (proba >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        return {
            "name": name,
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
            "specificity": specificity,
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

    print(f"Table 18. Predictive Model Performance Comparison (Test Set, n = {len(y_test)})")
    print(f"{'Model':<24}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}{'F1-Score':>10}{'Specificity':>13}{'ROC-AUC':>10}")
    for r in results:
        label = r["name"] + (" (selected)" if r["name"] == selected else "")
        print(f"{label:<24}{r['accuracy']:>10.3f}{r['precision']:>11.3f}{r['recall']:>9.3f}"
              f"{r['f1']:>10.3f}{r['specificity']:>13.4f}{r['roc_auc']:>10.3f}")


if __name__ == "__main__":
    main()
