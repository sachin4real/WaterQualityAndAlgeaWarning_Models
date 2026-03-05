from __future__ import annotations

import pandas as pd
import numpy as np

REQUIRED_COLS = [
    "timestamp", "tank_id", "ph", "temp_c", "turb_ntu", "ec",
    "water_status_label", "algae_label",
]

SENSOR_COLS = ["ph", "temp_c", "turb_ntu", "ec"]


def load_and_clean_csv(csv_path: str) -> pd.DataFrame:
    """Load CSV and enforce schema.

    Steps:
    - strip spaces in column names
    - parse timestamp
    - convert sensors to numeric
    - drop invalid rows
    - remove duplicate (tank_id, timestamp) keeping last
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    for c in SENSOR_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["timestamp", "tank_id"] + SENSOR_COLS + ["water_status_label", "algae_label"]).copy()

    # Normalize labels
    df["water_status_label"] = df["water_status_label"].astype(str).str.strip().str.upper()
    df["algae_label"] = df["algae_label"].astype(str).str.strip().str.upper()

    df = df.sort_values(["tank_id", "timestamp"])
    df = df.drop_duplicates(subset=["tank_id", "timestamp"], keep="last").reset_index(drop=True)
    return df


def resample_per_tank(df: pd.DataFrame, rule: str = "15min") -> pd.DataFrame:
    """Resample sensors per tank_id to a fixed interval."""
    out = []
    for tank_id, g in df.groupby("tank_id", sort=False):
        g = g.sort_values("timestamp").set_index("timestamp")

        sensors = g[SENSOR_COLS].resample(rule).mean()
        water_lab = g["water_status_label"].resample(rule).ffill()
        algae_lab = g["algae_label"].resample(rule).ffill()

        merged = sensors.join(water_lab).join(algae_lab)
        merged["tank_id"] = tank_id
        merged = merged.reset_index()
        out.append(merged)

    out_df = pd.concat(out, ignore_index=True)
    out_df = out_df.dropna(subset=SENSOR_COLS + ["water_status_label", "algae_label"])
    out_df = out_df.sort_values(["tank_id", "timestamp"]).reset_index(drop=True)
    return out_df


def add_time_series_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling mean/std and delta features per tank."""
    df = df.sort_values(["tank_id", "timestamp"]).reset_index(drop=True).copy()

    for c in SENSOR_COLS:
        df[f"{c}_rm2"] = df.groupby("tank_id")[c].transform(lambda s: s.rolling(2, min_periods=2).mean())
        df[f"{c}_rm4"] = df.groupby("tank_id")[c].transform(lambda s: s.rolling(4, min_periods=4).mean())
        df[f"{c}_rs2"] = df.groupby("tank_id")[c].transform(lambda s: s.rolling(2, min_periods=2).std())
        df[f"{c}_d2"]  = df.groupby("tank_id")[c].diff(2)

    return df


def get_feature_columns() -> list[str]:
    cols: list[str] = []
    for c in SENSOR_COLS:
        cols.extend([c, f"{c}_rm2", f"{c}_rm4", f"{c}_rs2", f"{c}_d2"])
    return cols


def build_ml_table(df_raw: pd.DataFrame, resample_rule: str = "15min") -> pd.DataFrame:
    """Full feature pipeline: clean -> resample -> features -> drop NaNs."""
    df_r = resample_per_tank(df_raw, rule=resample_rule)
    df_f = add_time_series_features(df_r)

    feature_cols = get_feature_columns()
    df_f = df_f.dropna(subset=feature_cols + ["water_status_label", "algae_label"]).reset_index(drop=True)
    return df_f


def time_split_by_tank(df: pd.DataFrame, test_ratio: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leakage-resistant split: for each tank, last X% (by time) goes to test."""
    train_parts = []
    test_parts = []

    for tank_id, g in df.groupby("tank_id", sort=False):
        g = g.sort_values("timestamp")
        n = len(g)
        cut = int(np.floor(n * (1 - test_ratio)))
        cut = max(cut, 1)
        train_parts.append(g.iloc[:cut])
        test_parts.append(g.iloc[cut:])

    return pd.concat(train_parts, ignore_index=True), pd.concat(test_parts, ignore_index=True)
