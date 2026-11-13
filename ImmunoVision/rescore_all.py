"""Re-scores every child with the current model (data/risk_model.joblib).

Run after retraining so the risk scores shown on the pages come from the new
model. Adds one new risk assessment per child; older ones are kept as history.

Run from the ImmunoVision folder:
    python rescore_all.py
"""
from collections import Counter

from app import create_app, _recompute_risk
from models import Child


def main():
    app = create_app()
    with app.app_context():
        children = Child.query.all()
        engines, labels = Counter(), Counter()
        for child in children:
            result = _recompute_risk(child)
            engines["model" if not result["model_version"].startswith("rules") else "rules"] += 1
            labels[result["label"]] += 1
        print(f"Re-scored {len(children)} children: {engines['model']} by the model, "
              f"{engines['rules']} by the rule-based scorer.")
        print(f"At-Risk: {labels['At-Risk']}, Not At-Risk: {labels['Not At-Risk']}")


if __name__ == "__main__":
    main()
