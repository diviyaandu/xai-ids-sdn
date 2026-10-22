from src.data import load_raw, clean, ID_COLS

df = clean(load_raw())
print(df.groupby("scenario")["Timestamp"].agg(["min", "max", lambda s: s.isna().sum()]))
print(df.groupby("scenario")["Label"].value_counts())

feats = [c for c in df.columns if c not in ID_COLS + ["Label", "scenario"]]
dup = df.duplicated(subset=feats, keep=False)
print("\nrows in duplicate groups:", int(dup.sum()))
print(df[dup].groupby(["scenario", "Label"]).size())
conf = df.groupby(feats)["Label"].nunique()
print("conflicting-label groups:", int((conf > 1).sum()))