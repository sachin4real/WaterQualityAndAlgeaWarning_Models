import pandas as pd
import joblib
from src.features import build_dataset

WATER_MODEL_PATH = "artifacts/water_status_model.joblib"
ALGAE_MODEL_PATH = "artifacts/algae_warning_model.joblib"
DATA_PATH = "data/hydroponic_2_tanks_realistic.csv"

def main():
    water_bundle = joblib.load(WATER_MODEL_PATH)
    algae_bundle = joblib.load(ALGAE_MODEL_PATH)

    water_model = water_bundle["model"]
    algae_model = algae_bundle["model"]
    feats = water_bundle["features"]

    raw = pd.read_csv(DATA_PATH)
    df = build_dataset(raw)

    # use latest 5 rows of each tank for sanity
    for tank_id in df["tank_id"].unique():
        tail = df[df["tank_id"] == tank_id].sort_values("timestamp").tail(5)
        X = tail[feats]

        w_probs = water_model.predict_proba(X)
        a_probs = algae_model.predict_proba(X)

        w_pred = water_model.predict(X)
        a_pred = algae_model.predict(X)

        print("\nTank:", tank_id)
        print("Last timestamps:", tail["timestamp"].astype(str).tolist())
        print("Water preds:", w_pred.tolist())
        print("Algae preds:", a_pred.tolist())
        print("Water probs last:", dict(zip(water_model.classes_, w_probs[-1].tolist())))
        print("Algae probs last:", dict(zip(algae_model.classes_, a_probs[-1].tolist())))

    print("\nSaved metrics:")
    print("Water:", water_bundle.get("metrics", {}))
    print("Algae:", algae_bundle.get("metrics", {}))

if __name__ == "__main__":
    main()