import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from src.splits import prepare, scenario_folds
from src.data import feature_cols
from src.features import make_preproc

def xy(d, feats):
    X = d[feats].replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32")
    return X, d["y"].values

DROP_SETS = {
    "all": [],
    "-win": ["Init Bwd Win Byts"],
    "-win-hdr": ["Init Bwd Win Byts", "Fwd Header Len", "Bwd Header Len", "Init Fwd Win Byts"],
}

if __name__ == "__main__":
    from sklearn.metrics import f1_score, recall_score
    df = prepare()
    base = feature_cols(df)
    rows = []
    for name, (tr, te) in scenario_folds(df).items():
        for tag, drop in DROP_SETS.items():
            feats = [f for f in base if f not in drop]
            Xtr, ytr = xy(tr, feats); Xte, yte = xy(te, feats)
            pipe = Pipeline(make_preproc() + [
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])
            pipe.fit(Xtr, ytr)
            p = pipe.predict(Xte)
            rows.append((name, tag, f1_score(yte, p, average="macro"),
                         recall_score(yte, p, pos_label=0), recall_score(yte, p, pos_label=1)))
    import pandas as pd
    out = pd.DataFrame(rows, columns=["fold", "features", "macroF1", "recall_normal", "recall_attack"]).round(4)
    print(out.to_string(index=False))
    out.to_csv("reports/metrics/logreg_ablation.csv", index=False)