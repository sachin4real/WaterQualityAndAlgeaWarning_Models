import pandas as pd

BASE_COLS = ["ph", "temp_c", "turb_ntu", "ec"]


def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # normalize headers (removes hidden spaces like "tank_id ")
    df.columns = [c.strip() for c in df.columns]

    # timestamp variants
    if "timestamp" not in df.columns and "created_at" in df.columns:
        df = df.rename(columns={"created_at": "timestamp"})

    # tank id variants
    if "tank_id" not in df.columns and "pond_id" in df.columns:
        df = df.rename(columns={"pond_id": "tank_id"})
    if "tank_id" not in df.columns and "Tank_ID" in df.columns:
        df = df.rename(columns={"Tank_ID": "tank_id"})
    if "tank_id" not in df.columns and "tankId" in df.columns:
        df = df.rename(columns={"tankId": "tank_id"})

    # ec variants
    if "ec" not in df.columns and "EC" in df.columns:
        df = df.rename(columns={"EC": "ec"})

    required = ["timestamp", "tank_id"] + BASE_COLS + ["water_status_label", "algae_label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}. Found: {df.columns.tolist()}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    df = df.dropna(subset=["timestamp", "tank_id"] + BASE_COLS)
    df["tank_id"] = df["tank_id"].astype(str)

    for c in BASE_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=BASE_COLS + ["water_status_label", "algae_label"])
    df = df.sort_values(["tank_id", "timestamp"]).reset_index(drop=True)
    return df


def resample_15min(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample per tank to 15-min interval (mean). Keeps labels by forward-fill.
    Ensures tank_id is preserved.
    """
    out = []
    for tank, g in df.groupby("tank_id"):
        g = g.sort_values("timestamp").copy()
        g = g.set_index("timestamp")

        num = g[BASE_COLS].resample("15min").mean()
        labs = g[["water_status_label", "algae_label"]].resample("15min").ffill()

        merged = pd.concat([num, labs], axis=1).reset_index()
        merged["tank_id"] = str(tank)  # force tank_id back
        out.append(merged)

    df15 = pd.concat(out, ignore_index=True)

    df15["tank_id"] = df15["tank_id"].astype(str)
    for c in BASE_COLS:
        df15[c] = pd.to_numeric(df15[c], errors="coerce")

    df15 = df15.dropna(subset=BASE_COLS + ["water_status_label", "algae_label", "tank_id"])
    df15 = df15.sort_values(["tank_id", "timestamp"]).reset_index(drop=True)
    return df15


def add_time_series_features(df15: pd.DataFrame):
    """
    Adds rolling mean/std and 30-min delta features per tank.
    Window: 2 steps = 30 min, 4 steps = 60 min.
    Returns: df_feat, feature_cols
    """

    def per_tank(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("timestamp").copy()

        for col in BASE_COLS:
            # 30 min (2 steps)
            g[f"{col}_rm2"] = g[col].rolling(2, min_periods=2).mean()
            g[f"{col}_rs2"] = g[col].rolling(2, min_periods=2).std(ddof=0)

            # 60 min (4 steps)
            g[f"{col}_rm4"] = g[col].rolling(4, min_periods=4).mean()

            # delta over 30 min (2 steps back)
            g[f"{col}_d2"] = g[col] - g[col].shift(2)

        return g

    frames = []
    for tank, g in df15.groupby("tank_id"):
        g2 = per_tank(g)
        g2["tank_id"] = str(tank)  # keep tank_id
        frames.append(g2)

    df_feat = pd.concat(frames, ignore_index=True)

    feature_cols = (
        BASE_COLS
        + [f"{c}_rm2" for c in BASE_COLS]
        + [f"{c}_rs2" for c in BASE_COLS]
        + [f"{c}_rm4" for c in BASE_COLS]
        + [f"{c}_d2" for c in BASE_COLS]
    )

    df_feat = df_feat.dropna(subset=feature_cols).reset_index(drop=True)

    if "tank_id" not in df_feat.columns:
        raise KeyError(f"'tank_id' missing after feature engineering. Columns: {df_feat.columns.tolist()}")

    return df_feat, feature_cols