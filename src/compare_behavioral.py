import numpy as np, pandas as pd, shap
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from src.splits import prepare, scenario_folds
from src.data import feature_cols, ENV_FEATURES
from src.features import make_preproc
from src.train import xy

#ENV = ["Init Bwd Win Byts", "Src Port", "Dst Port", "Fwd Header Len", "Bwd Header Len"]
ENV = ENV_FEATURES

df = prepare(); base = feature_cols(df)
FEATS = {"full": base, "behavioral": [f for f in base if f not in ENV]}
rows, top = [], {}

for fold, (tr, te) in scenario_folds(df).items():
    for tag, feats in FEATS.items():
        Xtr, ytr = xy(tr, feats); Xte, yte = xy(te, feats)
        pre = Pipeline(make_preproc()).fit(Xtr, ytr)
        names = np.array(feats)[pre.named_steps["var"].get_support()][pre.named_steps["sel"].get_support()]
        auc, f1, rn, ra = [], [], [], []
        for seed in range(3):
            rf = RandomForestClassifier(n_estimators=100, max_depth=16, n_jobs=-1,
                                        class_weight="balanced", random_state=seed
                                        ).fit(pre.transform(Xtr), ytr)
            Xt = pre.transform(Xte); p = rf.predict(Xt)
            auc.append(roc_auc_score(yte, rf.predict_proba(Xt)[:, 1]))
            f1.append(f1_score(yte, p, average="macro"))
            rn.append(recall_score(yte, p, pos_label=0)); ra.append(recall_score(yte, p))
        rows.append((fold, tag, len(names), np.mean(f1), np.std(f1), np.mean(auc), np.std(auc),
                     np.mean(rn), np.mean(ra)))
        # SHAP top features on 300 flagged-attack test flows (last seed's model)
        idx = np.where(rf.predict_proba(Xt)[:, 1] >= 0.5)[0]
        idx = np.random.default_rng(0).choice(idx, min(300, len(idx)), replace=False)
        sv = shap.TreeExplainer(rf).shap_values(Xt[idx], check_additivity=False)
        sv = sv[1] if isinstance(sv, list) else (sv[..., 1] if sv.ndim == 3 else sv)
        top[(fold, tag)] = pd.Series(np.abs(sv).mean(0), index=names).nlargest(6).round(4)

out = pd.DataFrame(rows, columns=["fold", "features", "n_feat", "F1_mean", "F1_std",
                                  "AUC_mean", "AUC_std", "recall_normal", "recall_attack"]).round(4)
print(out.to_string(index=False))
out.to_csv("reports/metrics/behavioral_vs_full.csv", index=False)
for k, v in top.items():
    print(f"\n{k} top SHAP:"); print(v.to_string())