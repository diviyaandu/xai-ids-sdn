import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score
from xgboost import XGBClassifier
from src.splits import prepare, scenario_folds, random_split
from src.data import feature_cols
from src.features import make_preproc

MERGE = {"DoS": "DoS/DDoS", "DDoS": "DoS/DDoS"}
SHARED = ["BFA", "DoS/DDoS", "Normal", "Probe"]

def models():
    return {
        "LogReg": [("scale", StandardScaler()),
                   ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))],
        "RF": [("clf", RandomForestClassifier(n_estimators=200, max_depth=16, n_jobs=-1,
                                              class_weight="balanced", random_state=42))],
        "XGB": [("clf", XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                      n_jobs=-1, random_state=42))],
    }

def mat(d, feats, classes):
    X = d[feats].replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32")
    return X, d["cls"].map({c: i for i, c in enumerate(classes)}).values

if __name__ == "__main__":
    df = prepare(); df["cls"] = df["Label"].replace(MERGE); feats = feature_cols(df)
    assert not {"y", "cls", "Label", "scenario"} & set(feats), "label column in features"
    jobs = {k: (tr, te, SHARED) for k, (tr, te) in scenario_folds(df).items()}
    jobs["random_7cls"] = (*random_split(df), sorted(df["cls"].unique()))
    rows = []
    for sname, (tr, te, classes) in jobs.items():
        if sname != "random_7cls":
            tr, te = tr[tr.cls.isin(classes)], te[te.cls.isin(classes)]
        Xtr, ytr = mat(tr, feats, classes); Xte, yte = mat(te, feats, classes)
        present = sorted(set(yte))
        for mname, steps in models().items():
            pipe = Pipeline(make_preproc() + steps).fit(Xtr, ytr)
            p = pipe.predict(Xte)
            rec = recall_score(yte, p, labels=range(len(classes)), average=None, zero_division=0)
            rows.append({"split": sname, "model": mname,
                         "macroF1": f1_score(yte, p, labels=present, average="macro", zero_division=0),
                         **{f"R_{c}": (r if i in present else np.nan)
                            for i, (c, r) in enumerate(zip(classes, rec))}})    
    out = pd.DataFrame(rows).round(4)
    print(out.to_string(index=False))
    out.to_csv("reports/metrics/multiclass.csv", index=False)