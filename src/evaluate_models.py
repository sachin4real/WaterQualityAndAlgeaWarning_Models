import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from features import load_dataset, resample_15min, add_time_series_features

# --------- SET YOUR PATHS HERE ---------
TRAIN_DATA = "data/hydroponic_lettuce.csv"                 # Dataset A (train)
EXTERNAL_DATA = "data/hydroponic_lettuce_balanced_sl_like.csv"  # Dataset B (test)
# --------------------------------------

WATER_MODEL_PATH = "artifacts/water_status_model.joblib"
ALGAE_MODEL_PATH = "artifacts/algae_risk_model.joblib"


def _prepare(df_path: str):
    df = load_dataset(df_path)
    df15 = resample_15min(df)
    df_feat, feature_cols = add_time_series_features(df15)
    return df_feat, feature_cols


def eval_one(model_path: str, label_col: str, data_path: str, title: str):
    art = joblib.load(model_path)
    model = art["model"]
    feats = art["features"]

    df_feat, _ = _prepare(data_path)
    X = df_feat[feats].astype(float)
    y_true = df_feat[label_col].astype(str)

    y_pred = model.predict(X)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    print(f"\n==== {title} ====")
    print(f"Model: {model_path}")
    print(f"Data : {data_path}")
    print("\nConfusion matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, digits=4))
    print(f"\naccuracy: {acc:.4f}")
    print(f"macro F1: {macro_f1:.4f}")


def main():
    # Internal evaluation (Dataset A)
    eval_one(WATER_MODEL_PATH, "water_status_label", TRAIN_DATA, "Water Status - Internal Eval (Dataset A)")
    eval_one(ALGAE_MODEL_PATH, "algae_label", TRAIN_DATA, "Algae Risk - Internal Eval (Dataset A)")

    # External evaluation (Dataset B)
    eval_one(WATER_MODEL_PATH, "water_status_label", EXTERNAL_DATA, "Water Status - External Eval (Dataset B)")
    eval_one(ALGAE_MODEL_PATH, "algae_label", EXTERNAL_DATA, "Algae Risk - External Eval (Dataset B)")


if __name__ == "__main__":
    main()