"""Generates the supplementary model-evaluation artifacts for the capstone
paper: confusion matrix, classification report, feature importance chart,
ROC / precision-recall curves, and a handful of worked example predictions.

train_model.py already prints accuracy/precision/recall/F1/ROC-AUC for all
three candidates to the console, but that output isn't something you can
paste into a paper as a figure or table - this script produces those.

Run (after train_model.py has already produced data/risk_model.joblib):
    pip install matplotlib   # one-time, only needed for this script
    python evaluate_model.py

Everything is written to ./model_eval/ by default (--out to change it):
    confusion_matrix.png          - confusion matrix heatmap
    classification_report.txt     - sklearn's per-class precision/recall/F1
    feature_importance.png        - bar chart (tree models) ...
    feature_coefficients.txt      - ... or coefficients (logistic regression)
    supplementary_evaluation.png  - ROC curve + precision-recall curve
    example_predictions.txt       - a few individual test-set predictions
                                     with the features that drove them

Pass --n-synthetic / --seed / --no-real-data matching whatever you trained
with (same defaults as train_model.py) to reproduce the exact same held-out
test set the console metrics above were computed on; different values still
give a valid evaluation, just on a different random split.
"""
import argparse
import os

import numpy as np

import data_processor as dp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-synthetic", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-real-data", action="store_true")
    parser.add_argument("--out", default="model_eval")
    parser.add_argument("--n-examples", type=int, default=8,
                         help="How many worked examples to write out")
    args = parser.parse_args()

    if not dp.model_is_trained():
        raise SystemExit(
            "No trained model found at data/risk_model.joblib - run train_model.py first.")

    import joblib
    from sklearn.metrics import (
        classification_report, confusion_matrix, ConfusionMatrixDisplay,
        roc_curve, auc, precision_recall_curve,
    )
    from sklearn.model_selection import train_test_split
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    artifact = joblib.load(dp.MODEL_PATH)
    model = artifact["model"]
    model_name = artifact["model_name"]
    needs_scaler = artifact["needs_scaler"]
    scaler = artifact["scaler"]
    feature_columns = artifact["feature_columns"]
    print(f"Evaluating saved model: {model_name} ({artifact.get('model_version', '?')})")

    real_children = [] if args.no_real_data else dp.load_and_merge_real_registries()
    rows, _ = dp.generate_training_dataset(args.n_synthetic, args.seed, real_children)
    X = np.array([[r[c] for c in feature_columns] for r in rows], dtype=float)
    y = np.array([1 if r["label"] == "At-Risk" else 0 for r in rows], dtype=int)

    X_train, X_test, y_train, y_test, rows_train, rows_test = train_test_split(
        X, y, rows, test_size=0.2, random_state=args.seed, stratify=y)

    X_test_eval = scaler.transform(X_test) if needs_scaler else X_test
    proba = model.predict_proba(X_test_eval)[:, 1]
    pred = (proba >= 0.5).astype(int)

    os.makedirs(args.out, exist_ok=True)
    labels = ["Not At-Risk", "At-Risk"]

    # 1. Confusion matrix
    cm = confusion_matrix(y_test, pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"Confusion Matrix – {model_name}")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "confusion_matrix.png"), dpi=150)
    plt.close(fig)

    # 2. Classification report
    report_text = classification_report(y_test, pred, target_names=labels, digits=3)
    with open(os.path.join(args.out, "classification_report.txt"), "w") as f:
        f.write(f"Classification Report – {model_name}\n")
        f.write("Test set size: %d (%d At-Risk, %d Not At-Risk)\n\n"
                % (len(y_test), int(y_test.sum()), int((1 - y_test).sum())))
        f.write(report_text)
    print(report_text)

    # 3. Feature importance (tree models) or coefficients (logistic regression)
    importance = artifact.get("feature_importance")
    if importance:
        items = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh([k for k, v in items][::-1], [v for k, v in items][::-1], color="#2563eb")
        ax.set_xlabel("Importance")
        ax.set_title(f"Feature Importance – {model_name}")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "feature_importance.png"), dpi=150)
        plt.close(fig)
    elif hasattr(model, "coef_"):
        items = sorted(zip(feature_columns, model.coef_[0].tolist()),
                        key=lambda kv: abs(kv[1]), reverse=True)
        with open(os.path.join(args.out, "feature_coefficients.txt"), "w") as f:
            f.write(f"Logistic Regression Coefficients – {model_name}\n")
            f.write("(positive = raises predicted risk, negative = lowers it)\n\n")
            for k, v in items:
                f.write(f"{k:28s} {v:+.4f}\n")

    # 4. Supplementary evaluation: ROC curve + precision-recall curve
    fpr, tpr, _ = roc_curve(y_test, proba)
    roc_auc = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(y_test, proba)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(fpr, tpr, color="#2563eb", label=f"AUC = {roc_auc:.3f}")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="#9ca3af")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend(loc="lower right")

    axes[1].plot(rec, prec, color="#16a34a")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")

    fig.suptitle(f"Supplementary Evaluation – {model_name}")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "supplementary_evaluation.png"), dpi=150)
    plt.close(fig)

    # 5. Example predictions
    n = min(args.n_examples, len(y_test))
    idx = np.random.RandomState(args.seed).choice(len(y_test), size=n, replace=False)
    lines = [f"Example Predictions – {model_name}", "=" * 60, ""]
    for i in idx:
        row = rows_test[i]
        actual = labels[y_test[i]]
        predicted = labels[pred[i]]
        mark = "CORRECT" if actual == predicted else "WRONG"
        lines.append(f"[{mark}] Barangay: {row['barangay']}")
        lines.append(f"  Predicted: {predicted}  ({proba[i] * 100:.1f}% risk)   Actual: {actual}")
        lines.append(
            f"  doses_missed={row['doses_missed']:.0f}  completion_rate={row['completion_rate']:.2f}  "
            f"avg_delay_days={row['avg_delay_days']:.1f}  days_since_last_dose={row['days_since_last_dose']:.0f}")
        lines.append("")
    with open(os.path.join(args.out, "example_predictions.txt"), "w") as f:
        f.write("\n".join(lines))

    print(f"\nSaved to {args.out}/:")
    print("  confusion_matrix.png")
    print("  classification_report.txt")
    print("  feature_importance.png" if importance else "  feature_coefficients.txt")
    print("  supplementary_evaluation.png  (ROC + precision-recall curves)")
    print("  example_predictions.txt")


if __name__ == "__main__":
    main()
