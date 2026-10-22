import pandas as pd
from src.data import load_raw, clean, dedupe, feature_cols

pd.set_option("display.width", 200)
df = clean(load_raw())
feats = feature_cols(df)

span = df.groupby(feats)["scenario"].nunique()
print("duplicate groups spanning >1 scenario:", int((span > 1).sum()))

d = dedupe(df)
print("rows before/after:", len(df), len(d))
print(pd.crosstab(d["Label"], d["scenario"]))
print(d.groupby("scenario")["Timestamp"].agg(["min", "max"]).to_string())