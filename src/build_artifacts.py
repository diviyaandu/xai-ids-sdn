import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from src.splits import prepare, scenario_folds
from src.data import feature_cols, ENV_FEATURES
from src.features import make_preproc
from src.train import xy
from src.reasons import normal_ref

df = prepare(); base = feature_cols(df)
tr, te = scenario_folds(df)["A_ovs->meta2"]
sets = {"full": base, "behavioral": [f for f in base if f not in ENV_FEATURES]}
models = {}
for tag, feats in sets.items():
    Xtr, ytr = xy(tr, feats)
    pre = Pipeline(make_preproc()).fit(Xtr, ytr)
    names = np.array(feats)[pre.named_steps["var"].get_support()][pre.named_steps["sel"].get_support()]
    Xt = pre.transform(Xtr)
    rf = RandomForestClassifier(n_estimators=200, max_depth=16, n_jobs=-1,
                                class_weight="balanced", random_state=42).fit(Xt, ytr)
    rf.set_params(n_jobs=1)
    models[tag] = dict(feats=feats, pre=pre, rf=rf, names=list(names), ref=normal_ref(Xt, ytr, names))
sample = te.sample(2000, random_state=0)[base + ["Label"]].reset_index(drop=True)
joblib.dump({"models": models, "sample": sample, "base": base}, "models/artifacts_foldA.joblib")
print("saved", {k: len(v["names"]) for k, v in models.items()}, "sample", sample.shape)