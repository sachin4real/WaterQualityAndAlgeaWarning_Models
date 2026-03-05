import pandas as pd

BASE = ["ph", "temp_c", "turb_ntu", "ec"]

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # parse timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # ensure tank_id string
    df["tank_id"] = df["tank_id"].astype(str)

    # numeric conversions
    for c in BASE:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # drop invalid rows
    df = df.dropna(subset=["timestamp", "tank_id"] + BASE)
    return df

def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    # keep last record for duplicate (tank_id, timestamp)
    df = df.sort_values(["tank_id", "timestamp"])
    df = df.drop_duplicates(["tank_id", "timestamp"], keep="last")
    return df

def resample_15m(df: pd.DataFrame, minutes: int = 15) -> pd.DataFrame:
    out = []
    for tank_id, g in df.groupby("tank_id"):
        g = g.sort_values("timestamp").set_index("timestamp")

        # mean for numeric columns
        r = g.resample(f"{minutes}min").mean(numeric_only=True)

        r["tank_id"] = tank_id
        r = r.reset_index()
        out.append(r)

    if not out:
        return df.iloc[0:0]
    return pd.concat(out, ignore_index=True)

def add_features(df: pd.DataFrame, rm2: int = 2, rm4: int = 4) -> pd.DataFrame:
    df = df.sort_values(["tank_id", "timestamp"]).copy()

    for c in BASE:
        df[f"{c}_rm2"] = df.groupby("tank_id")[c].transform(
            lambda s: s.rolling(rm2, min_periods=1).mean()
        )
        df[f"{c}_rs2"] = df.groupby("tank_id")[c].transform(
            lambda s: s.rolling(rm2, min_periods=1).std(ddof=0).fillna(0.0)
        )
        df[f"{c}_rm4"] = df.groupby("tank_id")[c].transform(
            lambda s: s.rolling(rm4, min_periods=1).mean()
        )
        df[f"{c}_d2"] = df.groupby("tank_id")[c].transform(
            lambda s: s.diff(rm2)
        ).fillna(0.0)

    return df

def feature_columns() -> list[str]:
    cols = []
    for c in BASE:
        cols += [c, f"{c}_rm2", f"{c}_rs2", f"{c}_rm4", f"{c}_d2"]
    return cols

def build_dataset(df_raw: pd.DataFrame, minutes: int = 15, rm2: int = 2, rm4: int = 4) -> pd.DataFrame:
    """
    Returns a dataframe with:
      timestamp, tank_id, base sensors, engineered features,
      plus original labels if present.
    """
    df = clean(df_raw)
    df = dedupe(df)
    df = resample_15m(df, minutes=minutes)
    df = add_features(df, rm2=rm2, rm4=rm4)
    return df