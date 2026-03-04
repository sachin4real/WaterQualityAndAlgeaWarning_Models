import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit

from features import load_dataset, resample_15min, add_time_series_features

DATA_PATH = "data/hydroponic_lettuce.csv"
OUT_PATH = "artifacts/algae_risk_model.joblib"


def build_rf():
    return RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )


def main():
    os.makedirs("artifacts", exist_ok=True)

    df = load_dataset(DATA_PATH)
    df15 = resample_15min(df)
    df_feat, feature_cols = add_time_series_features(df15)

    X = df_feat[feature_cols].astype(float)
    y = df_feat["water_status_label"].astype(str)
    groups = df_feat["tank_id"].astype(str)

    # Tank-wise split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.33, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = build_rf()
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    macro_f1 = f1_score(y_test, pred, average="macro")

    print("\n=== Water Status Model (RF) ===")
    print("Train tanks:", sorted(groups.iloc[train_idx].unique().tolist()))
    print("Test tanks :", sorted(groups.iloc[test_idx].unique().tolist()))
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, pred))
    print("\nClassification report:")
    print(classification_report(y_test, pred, digits=4))
    print(f"\naccuracy: {acc:.4f}")
    print(f"macro F1: {macro_f1:.4f}")

    artifact = {
        "model": model,
        "features": feature_cols,
        "label_col": "water_status_label",
        "classes": sorted(y.unique().tolist()),
        "eval": {"accuracy": float(acc), "macro_f1": float(macro_f1)},
    }

    joblib.dump(artifact, OUT_PATH)
    print("Saved:", OUT_PATH)


if __name__ == "__main__":
    main()