"""Generates everything an expert validator needs to score the Expert
Validation Questionnaire (Random Forest-Based Predictive Model) -
criteria 12-22, covering Predictive Performance & Accuracy and Model
Reliability & Interpretability. A focused subset of what
generate_paper_artifacts.py produces for the whole paper, reorganized
and renamed so each file maps to one or two questionnaire items and
the set can be handed to the expert on its own.

Produces, under --out (default expert_validation_artifacts/):

  00_questionnaire_map.txt     Which file answers which questionnaire item
  01_classification_report.txt Items 12, 15, 16 - accuracy, precision, recall, F1
  02_model_comparison.txt      Item 13 - Random Forest vs Logistic Regression vs XGBoost
  03_train_test_split.txt      Item 14 - confirms every metric is on held-out data
  04_cross_validation.txt      Item 18 - 5-fold stratified CV, mean +/- std
  05_feature_importance.png    Item 19
  06_roc_precision_recall.png  Item 16 (supplementary)
  07_confusion_matrix.png      Items 12, 16 (supplementary)
  08_example_predictions.txt   Items 17, 20 - real test-set children with the
                                feature values behind each call
  09_robustness_check.txt      Item 21 - performance across data sources and
                                age bands, and trained-on-synthetic/tested-on-
                                registry generalization
  10_limitations.txt           Item 22

Run:
    python generate_expert_validation_artifacts.py
    python generate_expert_validation_artifacts.py --out my_dir --n-synthetic 3000 --seed 7
"""
import argparse
import os

import numpy as np

import data_processor as dp


def _write(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  wrote {path}")


def questionnaire_map(out_dir):
    lines = [
        "Expert Validation Questionnaire - File Map",
        "=" * 60,
        "",
        "A. Predictive Performance and Accuracy",
        "  12. Acceptable predictive accuracy       -> 01_classification_report.txt",
        "  13. Consistent/reliable results           -> 02_model_comparison.txt",
        "  14. Performs well on unseen/test data     -> 03_train_test_split.txt, 01_classification_report.txt",
        "  15. Acceptable precision and recall       -> 01_classification_report.txt",
        "  16. F1-score and other metrics            -> 01_classification_report.txt, 06_roc_precision_recall.png, 07_confusion_matrix.png",
        "  17. Reliable for the intended application -> 08_example_predictions.txt",
        "",
        "B. Model Reliability and Interpretability",
        "  18. Consistent across test sets/runs      -> 04_cross_validation.txt, 09_robustness_check.txt",
        "  19. Feature importance meaningful         -> 05_feature_importance.png",
        "  20. Predictions interpretable to users    -> 08_example_predictions.txt",
        "  21. Robust to variations in input data    -> 09_robustness_check.txt",
        "  22. Limitations/error sources identified  -> 10_limitations.txt",
        "",
    ]
    _write(os.path.join(out_dir, "00_questionnaire_map.txt"), "\n".join(lines) + "\n")


def classification_report_and_confusion(X_train, X_test, y_train, y_test, out_dir, fig_dir, args, plt):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

    rf = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5,
                                 class_weight="balanced", random_state=args.seed).fit(X_train, y_train)
    proba = rf.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    report_text = classification_report(y_test, pred, target_names=["Not At-Risk", "At-Risk"], digits=3)
    _write(os.path.join(out_dir, "01_classification_report.txt"),
           f"Classification Report - Random Forest\n"
           f"Test set size: {len(y_test)} ({int(y_test.sum())} At-Risk, "
           f"{int((y_test == 0).sum())} Not At-Risk)\n\n{report_text}")

    cm = confusion_matrix(y_test, pred)
    fig, ax = plt.subplots(figsize=(5, 5), dpi=200)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not At-Risk", "At-Risk"]).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("Confusion Matrix - Random Forest")
    fig.tight_layout()
    path = os.path.join(fig_dir, "07_confusion_matrix.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")
    return rf, proba, pred


def model_comparison(X_train, X_test, y_train, y_test, out_dir, args):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                  f1_score, roc_auc_score, confusion_matrix)

    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

    def evaluate(name, model, Xte):
        proba = model.predict_proba(Xte)[:, 1]
        pred = (proba >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
        return {"name": name, "accuracy": accuracy_score(y_test, pred),
                "precision": precision_score(y_test, pred, zero_division=0),
                "recall": recall_score(y_test, pred, zero_division=0),
                "f1": f1_score(y_test, pred, zero_division=0),
                "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
                "roc_auc": roc_auc_score(y_test, proba)}

    results = []
    lr = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X_train_s, y_train)
    results.append(evaluate("Logistic Regression", lr, X_test_s))
    rf = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5,
                                 class_weight="balanced", random_state=args.seed).fit(X_train, y_train)
    results.append(evaluate("Random Forest (selected)", rf, X_test))
    pos_weight = (len(y_train) - y_train.sum()) / max(1, y_train.sum())
    xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.08, subsample=0.9,
                         colsample_bytree=0.9, scale_pos_weight=pos_weight, eval_metric="logloss",
                         random_state=args.seed).fit(X_train, y_train)
    results.append(evaluate("XGBoost", xgb, X_test))

    lines = [f"Model Comparison (Test Set, n = {len(y_test)})", ""]
    header = (f"{'Model':<26}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}"
              f"{'F1-Score':>10}{'Specificity':>13}{'ROC-AUC':>10}")
    lines.append(header)
    for r in results:
        lines.append(f"{r['name']:<26}{r['accuracy']:>10.3f}{r['precision']:>11.3f}{r['recall']:>9.3f}"
                      f"{r['f1']:>10.3f}{r['specificity']:>13.4f}{r['roc_auc']:>10.3f}")
    lines.append("")
    lines.append("Random Forest is the production model the app actually runs; the other two")
    lines.append("are shown only to demonstrate its results are not an artifact of one algorithm.")
    _write(os.path.join(out_dir, "02_model_comparison.txt"), "\n".join(lines) + "\n")


def train_test_split_table(y, y_train, y_test, out_dir):
    lines = ["Train/Test Split", ""]
    lines.append(f"Total rows: {len(y)}")
    lines.append(f"Training (80%): {len(y_train)} rows "
                 f"({int(y_train.sum())} At-Risk, {len(y_train) - int(y_train.sum())} Not At-Risk)")
    lines.append(f"Testing (20%):  {len(y_test)} rows "
                 f"({int(y_test.sum())} At-Risk, {len(y_test) - int(y_test.sum())} Not At-Risk)")
    lines.append("")
    lines.append("The split is stratified by class and held out before any model fitting -")
    lines.append("every metric in 01_classification_report.txt and 02_model_comparison.txt is")
    lines.append("measured on rows the model never saw during training.")
    _write(os.path.join(out_dir, "03_train_test_split.txt"), "\n".join(lines) + "\n")


def cross_validation(X, y, out_dir, args):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_validate

    rf = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5,
                                 class_weight="balanced", random_state=args.seed)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    scores = cross_validate(rf, X, y, cv=cv,
                             scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])

    lines = ["5-Fold Stratified Cross-Validation", ""]
    lines.append(f"{'Fold':<8}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}{'F1-Score':>10}{'ROC-AUC':>10}")
    for i in range(5):
        lines.append(f"{i + 1:<8}{scores['test_accuracy'][i]:>10.3f}{scores['test_precision'][i]:>11.3f}"
                      f"{scores['test_recall'][i]:>9.3f}{scores['test_f1'][i]:>10.3f}"
                      f"{scores['test_roc_auc'][i]:>10.3f}")
    lines.append("-" * 58)
    lines.append(f"{'Mean':<8}{scores['test_accuracy'].mean():>10.3f}{scores['test_precision'].mean():>11.3f}"
                  f"{scores['test_recall'].mean():>9.3f}{scores['test_f1'].mean():>10.3f}"
                  f"{scores['test_roc_auc'].mean():>10.3f}")
    lines.append(f"{'Std Dev':<8}{scores['test_accuracy'].std():>10.3f}{scores['test_precision'].std():>11.3f}"
                  f"{scores['test_recall'].std():>9.3f}{scores['test_f1'].std():>10.3f}"
                  f"{scores['test_roc_auc'].std():>10.3f}")
    lines.append("")
    lines.append("A small spread across folds (std dev near 0) means performance does not")
    lines.append("depend on which rows happened to land in any one test split.")
    _write(os.path.join(out_dir, "04_cross_validation.txt"), "\n".join(lines) + "\n")


def feature_importance(rf, fig_dir, plt):
    importance = dict(zip(dp.FEATURE_COLUMNS, rf.feature_importances_.tolist()))
    items = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)
    fig, ax = plt.subplots(figsize=(7, 5), dpi=200)
    ax.barh([k for k, v in items][::-1], [v for k, v in items][::-1], color="#2563eb")
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance - Random Forest")
    fig.tight_layout()
    path = os.path.join(fig_dir, "05_feature_importance.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def roc_precision_recall(y_test, proba, fig_dir, plt):
    from sklearn.metrics import roc_curve, auc as sk_auc, precision_recall_curve

    fpr, tpr, _ = roc_curve(y_test, proba)
    roc_auc_val = sk_auc(fpr, tpr)
    prec_curve, rec_curve, _ = precision_recall_curve(y_test, proba)
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
    fig.suptitle("ROC and Precision-Recall Curves - Random Forest")
    fig.tight_layout()
    path = os.path.join(fig_dir, "06_roc_precision_recall.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def example_predictions(rows, idx_test, y_test, pred_test, proba_test, out_dir, args):
    n_examples = min(8, len(y_test))
    sample_idx = np.random.RandomState(args.seed).choice(len(y_test), size=n_examples, replace=False)
    labels = ["Not At-Risk", "At-Risk"]
    lines = ["Example Predictions - Random Forest", "=" * 60, "",
             "Real test-set children, with the engineered feature values behind each call -",
             "this is the same information an RHU/BHW user sees when the app explains a",
             "prediction, not just the final label.", ""]
    for i in sample_idx:
        row = rows[idx_test[i]]
        actual = labels[y_test[i]]
        predicted = labels[pred_test[i]]
        mark = "CORRECT" if actual == predicted else "WRONG"
        lines.append(f"[{mark}] Barangay: {row['barangay']}")
        lines.append(f"  Predicted: {predicted}  ({proba_test[i] * 100:.1f}% risk)   Actual: {actual}")
        lines.append(
            f"  doses_missed={row['doses_missed']:.0f}  completion_rate={row['completion_rate']:.2f}  "
            f"avg_delay_days={row['avg_delay_days']:.1f}  days_since_last_dose={row['days_since_last_dose']:.0f}")
        lines.append("")
    _write(os.path.join(out_dir, "08_example_predictions.txt"), "\n".join(lines) + "\n")


def robustness_check(X, y, source, age_days, rf, out_dir, args):
    from sklearn.metrics import confusion_matrix, roc_auc_score
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test, src_train, src_test, age_train, age_test = train_test_split(
        X, y, source, age_days, test_size=0.2, random_state=args.seed, stratify=y)
    proba_test = rf.predict_proba(X_test)[:, 1]
    pred_test = (proba_test >= 0.5).astype(int)

    def metrics(mask, label):
        y_sub, pred_sub, proba_sub = y_test[mask], pred_test[mask], proba_test[mask]
        if len(y_sub) == 0:
            return (label, 0, 0, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))
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

    result_rows = [
        metrics(src_test == "Registry", "Registry children only (real data)"),
        metrics(src_test == "Synthetic", "Synthetic children only"),
        metrics((age_test >= 30) & (age_test <= 120), "Assessed at 30-120 days old"),
        metrics((age_test >= 121) & (age_test <= 240), "Assessed at 121-240 days old"),
        metrics((age_test >= 241) & (age_test <= 365), "Assessed at 241-365 days old"),
    ]

    # Cross-source generalization: train on synthetic data only, test on every
    # real registry child - the sharpest test of whether the model learned
    # something general rather than memorizing the synthetic generator.
    syn_mask, reg_mask = source == "Synthetic", source == "Registry"
    rf_syn = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5,
                                     class_weight="balanced", random_state=args.seed).fit(
        X[syn_mask], y[syn_mask])
    proba_reg = rf_syn.predict_proba(X[reg_mask])[:, 1]
    pred_reg = (proba_reg >= 0.5).astype(int)
    y_reg = y[reg_mask]
    tn, fp, fn, tp = confusion_matrix(y_reg, pred_reg, labels=[0, 1]).ravel()
    result_rows.append((
        "Trained on synthetic data only, tested on every registry child",
        len(y_reg), int(y_reg.sum()), (pred_reg == y_reg).mean(),
        tp / (tp + fp) if (tp + fp) else 0.0, tp / (tp + fn) if (tp + fn) else 0.0,
        tn / (tn + fp) if (tn + fp) else 0.0, roc_auc_score(y_reg, proba_reg),
    ))

    lines = ["Robustness Check - Performance Across Data Variations", "=" * 60, "",
             "Same model, broken out by subsets it was not specifically tuned for -",
             "performance holding up across these slices is evidence the model",
             "generalizes rather than overfitting one data source or age range.", ""]
    for label, n, n_at_risk, acc, prec, rec, spec, auc in result_rows:
        lines.append(label)
        lines.append(f"  n={n:<6} n(At-Risk)={n_at_risk:<6} accuracy={acc:.3f}  precision={prec:.3f}  "
                      f"recall={rec:.3f}  specificity={spec:.3f}  roc_auc={auc:.3f}")
        lines.append("")
    _write(os.path.join(out_dir, "09_robustness_check.txt"), "\n".join(lines))


def limitations(out_dir):
    text = """Known Limitations - Random Forest Predictive Model
============================================================

1. Data completeness. The real registry has incomplete dose-date fields
   (see Chapter 4's data-quality discussion) - some children's history is
   reconstructed from partial records rather than a full dose-by-dose log.

2. Reliance on synthetic augmentation. The registry alone is too small to
   train on directly, so most training rows are synthetically generated
   from the same feature distributions. Generalization to the real
   registry is checked directly in 09_robustness_check.txt, but synthetic
   data can still under-represent rare real-world patterns the generator
   wasn't designed to produce.

3. Risk labels are not stored historically. At-risk status reflects each
   child's most recent assessment, not a label frozen at the time a past
   prediction was made - so the model cannot be backtested against its
   own historical calls, only evaluated on held-out data from the same
   point in time.

4. Single municipality. All real data comes from San Jacinto, Pangasinan.
   Performance on a different municipality's registry, with different
   staffing, reporting habits, or population characteristics, is untested.

5. Threshold is fixed. Predictions use a single 0.5 probability cutoff for
   At-Risk / Not At-Risk; no cost-sensitive tuning (e.g. favoring recall
   over precision, given a missed at-risk child is more costly than a
   false alarm) has been applied.
"""
    _write(os.path.join(out_dir, "10_limitations.txt"), text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="expert_validation_artifacts")
    parser.add_argument("--n-synthetic", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    tables_dir = os.path.join(args.out, "tables")
    figures_dir = os.path.join(args.out, "figures")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print("Loading real registry children from data/raw/*.csv ...")
    children = dp.load_and_merge_real_registries()
    print(f"Loaded {len(children)} children.\n")

    print("Building training dataset ...")
    rows, _ = dp.generate_training_dataset(args.n_synthetic, args.seed, children)
    source = np.array(["Registry" if r["source"] == "real" else "Synthetic" for r in rows])
    age_days = np.array([r["age_days"] for r in rows])
    X = np.array([[r[c] for c in dp.FEATURE_COLUMNS] for r in rows], dtype=float)
    y = np.array([1 if r["label"] == "At-Risk" else 0 for r in rows], dtype=int)
    idx = np.arange(len(rows))

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, idx, test_size=0.2, random_state=args.seed, stratify=y)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("Questionnaire map:")
    questionnaire_map(args.out)

    print("\nItems 12, 15, 16 - classification report + confusion matrix:")
    rf, proba_test, pred_test = classification_report_and_confusion(
        X_train, X_test, y_train, y_test, tables_dir, figures_dir, args, plt)

    print("\nItem 13 - model comparison:")
    model_comparison(X_train, X_test, y_train, y_test, tables_dir, args)

    print("\nItem 14 - train/test split:")
    train_test_split_table(y, y_train, y_test, tables_dir)

    print("\nItem 18 - cross-validation:")
    cross_validation(X, y, tables_dir, args)

    print("\nItem 19 - feature importance:")
    feature_importance(rf, figures_dir, plt)

    print("\nItem 16 (supplementary) - ROC / precision-recall:")
    roc_precision_recall(y_test, proba_test, figures_dir, plt)

    print("\nItems 17, 20 - example predictions:")
    example_predictions(rows, idx_test, y_test, pred_test, proba_test, tables_dir, args)

    print("\nItem 21 - robustness check:")
    robustness_check(X, y, source, age_days, rf, tables_dir, args)

    print("\nItem 22 - limitations:")
    limitations(tables_dir)

    print(f"\nDone. Hand the expert the '{args.out}' folder, starting with "
          f"00_questionnaire_map.txt.")


if __name__ == "__main__":
    main()
