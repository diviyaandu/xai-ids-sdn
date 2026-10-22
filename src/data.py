from pathlib import Path
import numpy as np
import pandas as pd

RAW = Path("data/raw")
FILES = {"normal": "Normal_data.csv", "ovs": "OVS.csv", "meta2": "metasploitable-2.csv"}
ID_COLS = ["Flow ID", "Src IP", "Dst IP", "Timestamp"]
ENV_FEATURES = ["Init Bwd Win Byts", "Src Port", "Dst Port", "Fwd Header Len", "Bwd Header Len"]

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Label"] = df["Label"].str.strip()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce", format="mixed")
    return df

def load_raw() -> pd.DataFrame:
    dfs = []
    for scenario, fname in FILES.items():
        df = pd.read_csv(RAW / fname, low_memory=False)
        df.columns = df.columns.str.strip()
        df["scenario"] = scenario
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ID_COLS + ["Label", "scenario", "y"]]

def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ID_COLS + ["Label", "scenario", "y", "cls"]]

if __name__ == "__main__":
    df = load_raw()
    print("shape:", df.shape)
    print("\nlabel x scenario:\n", pd.crosstab(df["Label"], df["scenario"]))
    num = df.select_dtypes("number")
    print("\nnulls:", int(df.isna().sum().sum()), "| inf:", int(np.isinf(num).sum().sum()))
    print("duplicates:", int(df.drop(columns="scenario").duplicated().sum()))
    print("constant cols:", [c for c in num if num[c].nunique() <= 1])
    print("non-numeric cols:", list(df.select_dtypes(exclude="number").columns))