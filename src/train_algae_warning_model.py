from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from features import load_and_clean_csv, build_ml_table, get_feature_columns, time_split_by_tank


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "hydroponic_lettuce.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df_raw = load_and_clean_csv(str(DATA_PATH))
    df_ml = build_ml_table(df_raw, resample_rule="15min")

    feature_cols = get_feature_columns()
    train_df, test_df = time_split_by_tank(df_ml, test_ratio=0.2)

    X_train = train_df[feature_cols].values
    y_train = train_df["algae_label"].values

    X_test = test_df[feature_cols].values
    y_test = test_df["algae_label"].values

    clf = RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
        min_samples_leaf=2,
    )
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)
    acc = accuracy_score(y_test, pred)
    macro_f1 = f1_score(y_test, pred, average="macro")

    artifact = {
        "model": clf,
        "feature_cols": feature_cols,
        "classes": list(clf.classes_),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "metrics": {
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
            "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
            "classification_report": classification_report(y_test, pred, zero_division=0),
            "test_rows": int(len(test_df)),
            "train_rows": int(len(train_df)),
            "tanks": sorted(df_ml["tank_id"].unique().tolist()),
        },
    }

    out_path = ARTIFACTS_DIR / "algae_warning_model.joblib"
    joblib.dump(artifact, out_path)

    print(f"Saved: {out_path}")
    print(f"Test tanks: {artifact['metrics']['tanks']}")
    print(f"Accuracy: {acc:.6f}")
    print(f"Macro-F1: {macro_f1:.6f}")


if __name__ == "__main__":
    main()
