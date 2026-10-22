import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, precision_score
from xgboost import XGBClassifier
from src.splits import prepare, scenario_folds, random_split
from src.data import feature_cols
from src.features import make_preproc
from src.train import xy
from sklearn.metrics import roc_auc_score, average_precision_score

def models(spw):
    return {
        "RF": RandomForestClassifier(n_estimators=200, max_depth=16, n_jobs=-1,
                                     class_weight="balanced", random_state=42),
        "XGB": XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                             scale_pos_weight=1, n_jobs=-1, random_state=42,
                             eval_metric="logloss"),
    }

if __name__ == "__main__":
    df = prepare(); feats = feature_cols(df)
    splits = dict(scenario_folds(df))
    rtr, rte = random_split(df)
    splits["random"] = (rtr, rte)
    rows = []
    for sname, (tr, te) in splits.items():
        Xtr, ytr = xy(tr, feats); Xte, yte = xy(te, feats)
        spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
        for mname, clf in models(spw).items():
            pipe = Pipeline(make_preproc() + [("clf", clf)]).fit(Xtr, ytr)
            p = pipe.predict(Xte)
            s = pipe.predict_proba(Xte)[:, 1]
            auc, ap = roc_auc_score(yte, s), average_precision_score(yte, s)
            rows.append((sname, mname, f1_score(yte, p, average="macro"),
                         precision_score(yte, p), recall_score(yte, p),
                         recall_score(yte, p, pos_label=0), auc, ap))
    out = pd.DataFrame(rows, columns=["split", "model", "macroF1", "prec_attack",
                                      "recall_attack", "recall_normal", "auc", "pr_auc"]).round(4)
    print(out.to_string(index=False))
    out.to_csv("reports/metrics/rf_xgb_binary.csv", index=False)