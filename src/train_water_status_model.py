import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from features import load_dataset, resample_15min, add_time_series_features

DATA_PATH = "data/hydroponic_lettuce.csv"
OUT_PATH = "artifacts/water_status_model.joblib"


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

    # ✅ important: sort for time split
    df_feat = df_feat.sort_values(["tank_id", "timestamp"]).reset_index(drop=True)

    # ✅ time-based split per tank (train early, test later)
    train_parts = []
    test_parts = []
    for tank, g in df_feat.groupby("tank_id"):
        n = len(g)
        cut = int(n * 0.70)  # 70% train, 30% test
        train_parts.append(g.iloc[:cut])
        test_parts.append(g.iloc[cut:])

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    X_train = train_df[feature_cols].astype(float)
    y_train = train_df["water_status_label"].astype(str)

    X_test = test_df[feature_cols].astype(float)
    y_test = test_df["water_status_label"].astype(str)

    print("\n=== Water Status Model (RF) ===")
    print("Train tanks:", sorted(train_df["tank_id"].unique().tolist()))
    print("Test tanks :", sorted(test_df["tank_id"].unique().tolist()))
    print(f"Train rows: {len(train_df)} | Test rows: {len(test_df)}")

    model = build_rf()
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    macro_f1 = f1_score(y_test, pred, average="macro")

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
        "classes": sorted(df_feat["water_status_label"].astype(str).unique().tolist()),
        "eval": {"accuracy": float(acc), "macro_f1": float(macro_f1)},
        "split": {"type": "time_based_per_tank", "train_ratio": 0.70},
    }

    joblib.dump(artifact, OUT_PATH)
    print("Saved:", OUT_PATH)


if __name__ == "__main__":
    main()