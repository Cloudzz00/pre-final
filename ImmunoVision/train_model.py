"""Trains the Continuation Predictor's classifier and saves it to
data/risk_model.joblib.

Combines the real RHU registries in data/raw/*.csv with a synthetic
population for data augmentation (see data_processor.generate_training_dataset),
assessed at a random age inside the model's serving window
(ML_ASSESSMENT_FLOOR_DAYS..CONTINUATION_WINDOW_DAYS) so it is never asked to
score a child older than anything it trained on. Picks the best of
LogisticRegression / RandomForest / XGBoost by recall then ROC-AUC - recall
matters most here since a missed at-risk child is the costly error.

Run: python train_model.py [--no-real-data] [--n-synthetic 3000] [--seed 7]

predict_for_child() in data_processor.py picks the saved model up
automatically on the next prediction - no other wiring needed. Existing
RiskAssessment rows are not retroactively recomputed; reseed (seed.py
--reset) or edit a child's record to refresh them under the new model.
"""
import argparse

import data_processor as dp

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-synthetic", type=int, default=3000,
                         help="Synthetic children to generate for training augmentation")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-real-data", action="store_true",
                         help="Skip loading data/raw/*.csv; train on synthetic data only")
    args = parser.parse_args()
    dp.train_and_save_model(n_synthetic=args.n_synthetic, seed=args.seed, use_real_data=not args.no_real_data)
