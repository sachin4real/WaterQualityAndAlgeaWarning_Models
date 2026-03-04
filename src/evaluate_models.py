import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from features import load_dataset, resample_15min, add_time_series_features

DATA_PATH = "data/synthetic_sl_hydroponic_lettuce.csv"

WATER_MODEL_PATH = "artifacts/water_status_model.joblib"
ALGAE_MODEL_PATH = "artifacts/algae_risk_model.joblib"


def eval_one(model_path: str, label_col: str):
    art = joblib.load(model_path)
    model = art["model"]
    feats = art["features"]

    df = load_dataset(DATA_PATH)
    df15 = resample_15min(df)
    df_feat, _ = add_time_series_features(df15)

    X = df_feat[feats].astype(float)
    y_true = df_feat[label_col].astype(str)
    y_pred = model.predict(X)

    print(f"\n==== Evaluate on full dataset: {model_path} ====")
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, digits=4))


def main():
    eval_one(WATER_MODEL_PATH, "water_status_label")
    eval_one(ALGAE_MODEL_PATH, "algae_label")


if __name__ == "__main__":
    main()