import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from src.features import build_dataset, feature_columns

DATA_PATH = "data/hydroponic_2_tanks_realistic.csv"
OUT_PATH = "artifacts/algae_warning_model.joblib"

def split_by_tank(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    tanks = df["tank_id"].unique().tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(tanks)

    n_test = max(1, int(len(tanks) * test_size))
    test_tanks = set(tanks[:n_test])

    train_df = df[~df["tank_id"].isin(test_tanks)].copy()
    test_df = df[df["tank_id"].isin(test_tanks)].copy()
    return train_df, test_df, sorted(list(test_tanks))

def main():
    raw = pd.read_csv(DATA_PATH)
    df = build_dataset(raw)

    df = df.dropna(subset=["algae_label"])
    df["algae_label"] = df["algae_label"].astype(str)

    feats = feature_columns()

    train_df, test_df, test_tanks = split_by_tank(df)

    X_train = train_df[feats]
    y_train = train_df["algae_label"]

    X_test = test_df[feats]
    y_test = test_df["algae_label"]

    clf = RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )

    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)

    acc = accuracy_score(y_test, pred)
    mf1 = f1_score(y_test, pred, average="macro")

    bundle = {
        "model": clf,
        "features": feats,
        "labels": list(clf.classes_),
        "metrics": {
            "accuracy": float(acc),
            "macro_f1": float(mf1),
            "confusion_matrix": confusion_matrix(y_test, pred, labels=list(clf.classes_)).tolist(),
            "report": classification_report(y_test, pred, output_dict=True),
        },
        "data_used": DATA_PATH,
        "split": {
            "type": "tank-wise",
            "test_tanks": test_tanks,
        },
    }

    joblib.dump(bundle, OUT_PATH)
    print("Saved:", OUT_PATH)
    print("Test tanks:", test_tanks)
    print("Accuracy:", acc)
    print("Macro-F1:", mf1)

if __name__ == "__main__":
    main()