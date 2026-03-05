from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from features import load_and_clean_csv, build_ml_table, time_split_by_tank


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "hydroponic_lettuce.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def eval_one(artifact_path: Path, label_col: str, title: str):
    art = joblib.load(artifact_path)
    model = art["model"]
    feature_cols = art["feature_cols"]

    df_raw = load_and_clean_csv(str(DATA_PATH))
    df_ml = build_ml_table(df_raw, resample_rule="15min")
    _, test_df = time_split_by_tank(df_ml, test_ratio=0.2)

    X_test = test_df[feature_cols].values
    y_test = test_df[label_col].values

    pred = model.predict(X_test)

    print(f"\n==== {title} - Internal Eval (Time split per tank) ====")
    print(f"Model: {artifact_path}")
    print(f"Data : {DATA_PATH}")
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, pred))
    print("\nClassification report:")
    print(classification_report(y_test, pred, zero_division=0))
    print("\nAccuracy:", accuracy_score(y_test, pred))
    print("Macro-F1:", f1_score(y_test, pred, average="macro"))


def main():
    water_path = ARTIFACTS_DIR / "water_status_model.joblib"
    algae_path = ARTIFACTS_DIR / "algae_warning_model.joblib"

    if water_path.exists():
        eval_one(water_path, "water_status_label", "Water Status")
    else:
        print("Missing artifact:", water_path)

    if algae_path.exists():
        eval_one(algae_path, "algae_label", "Algae Warning")
    else:
        print("Missing artifact:", algae_path)


if __name__ == "__main__":
    main()
