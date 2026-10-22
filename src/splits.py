import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from src.data import load_raw, clean, feature_cols

def prepare() -> pd.DataFrame:
    df = clean(load_raw())
    feats = feature_cols(df)
    df["_h"] = pd.util.hash_pandas_object(df[feats], index=False)
    df = df.drop_duplicates(subset=["_h", "scenario"])
    df = df[df.groupby("_h")["scenario"].transform("nunique") == 1]
    df = df.drop(columns="_h").reset_index(drop=True)
    df["y"] = (df["Label"] != "Normal").astype(int)
    return df

def balanced_group_split(normal: pd.DataFrame):
    counts = normal["Src IP"].value_counts()
    a, b = [], []
    sa = sb = 0
    for ip, n in counts.items():
        if sa <= sb:
            a.append(ip); sa += n
        else:
            b.append(ip); sb += n
    return normal[normal["Src IP"].isin(a)], normal[normal["Src IP"].isin(b)]

def scenario_folds(df, seed=42):
    normal = df[df.scenario == "normal"]
    n1, n2 = balanced_group_split(normal)
    ovs, meta = df[df.scenario == "ovs"], df[df.scenario == "meta2"]
    return {"A_ovs->meta2": (pd.concat([ovs, n1]), pd.concat([meta, n2])),
            "B_meta2->ovs": (pd.concat([meta, n2]), pd.concat([ovs, n1]))}

def random_split(df, seed=42):
    return train_test_split(df, test_size=0.3, stratify=df["Label"], random_state=seed)



if __name__ == "__main__":
    df = prepare()
    print("rows:", len(df))
    for name, (tr, te) in scenario_folds(df).items():
        print(f"\n{name}: train={len(tr)} test={len(te)}")
        print(pd.concat([tr["Label"].value_counts().rename("train"),
                         te["Label"].value_counts().rename("test")], axis=1).fillna(0).astype(int))