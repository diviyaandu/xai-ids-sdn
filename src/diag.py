import numpy as np, pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from src.splits import prepare, scenario_folds
from src.data import feature_cols
from src.train import xy

df = prepare(); feats = feature_cols(df)
for name, (tr, te) in scenario_folds(df).items():
    Xtr, ytr = xy(tr, feats); Xte, yte = xy(te, feats)
    res = {}
    for f in feats:
        m = DecisionTreeClassifier(max_depth=2, random_state=0).fit(Xtr[[f]], ytr)
        res[f] = roc_auc_score(yte, m.predict_proba(Xte[[f]])[:, 1])
    print(f"\n{name}: top single-feature test AUC")
    print(pd.Series(res).sort_values(ascending=False).head(12).round(4))