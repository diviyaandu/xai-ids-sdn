import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from src.splits import prepare, scenario_folds
from src.data import feature_cols
from src.features import make_preproc
from src.train import xy

W, P = ["Init Bwd Win Byts"], ["Dst Port"]
H = ["Fwd Header Len", "Bwd Header Len"]
SETS = {"all": [], "-win": W, "-win-port": W + P, "-win-port-hdr": W + P + H}
SEEDS = range(5)

df = prepare(); base = feature_cols(df); rows = []
for fold, (tr, te) in scenario_folds(df).items():
    for tag, drop in SETS.items():
        feats = [f for f in base if f not in drop]
        Xtr, ytr = xy(tr, feats); Xte, yte = xy(te, feats)
        for seed in SEEDS:
            pipe = Pipeline(make_preproc(seed=seed) + [("clf", RandomForestClassifier(
                n_estimators=100, max_depth=16, n_jobs=-1,
                class_weight="balanced", random_state=seed))]).fit(Xtr, ytr)
            p, s = pipe.predict(Xte), pipe.predict_proba(Xte)[:, 1]
            rows.append((fold, tag, seed, f1_score(yte, p, average="macro"),
                         roc_auc_score(yte, s), recall_score(yte, p, pos_label=0),
                         recall_score(yte, p)))
out = pd.DataFrame(rows, columns=["fold", "features", "seed", "macroF1", "auc",
                                  "recall_normal", "recall_attack"])
out.round(4).to_csv("reports/metrics/rf_ablation_seeds.csv", index=False)
agg = (out.groupby(["fold", "features"], sort=False)
          [["macroF1", "auc", "recall_normal", "recall_attack"]]
          .agg(["mean", "std"]).round(4))
print(agg.to_string())
agg.to_csv("reports/metrics/rf_ablation_agg.csv")