import time, numpy as np, pandas as pd, shap, joblib
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from src.splits import prepare, scenario_folds
from src.data import feature_cols
from src.features import make_preproc
from src.train import xy

df = prepare(); feats = feature_cols(df)
tr, te = scenario_folds(df)["A_ovs->meta2"]
Xtr, ytr = xy(tr, feats); Xte, yte = xy(te, feats)

pre = Pipeline(make_preproc()).fit(Xtr, ytr)
names = np.array(feats)[pre.named_steps["var"].get_support()][pre.named_steps["sel"].get_support()]
rf = RandomForestClassifier(n_estimators=200, max_depth=16, n_jobs=-1,
                            class_weight="balanced", random_state=42).fit(pre.transform(Xtr), ytr)
rf.set_params(n_jobs=1)
joblib.dump((pre, rf, list(names)), "models/rf_foldA.joblib")

Xs = pd.DataFrame(pre.transform(Xte), columns=names).sample(500, random_state=0)
expl = shap.TreeExplainer(rf)

def lean_predict(X):
    X = np.asarray(X, dtype=np.float32)
    out = []
    for t in rf.estimators_:
        v = t.tree_.predict(X).reshape(len(X), -1)
        out.append(v[:, 1] / v.sum(1))
    return np.mean(out, axis=0)
assert np.allclose(lean_predict(Xs), rf.predict_proba(Xs)[:, 1], atol=1e-4), "lean predict mismatch"

def timeit(fn, n=3):
    fn()  # warm-up
    t = []
    for _ in range(n):
        s = time.perf_counter(); fn(); t.append(time.perf_counter() - s)
    return np.median(t)

x1 = Xs.iloc[[0]]
T = {
    "sklearn_predict": (timeit(lambda: rf.predict_proba(x1), 20),
                        timeit(lambda: rf.predict_proba(Xs)) / 500),
    "lean_predict":    (timeit(lambda: lean_predict(x1), 20),
                        timeit(lambda: lean_predict(Xs)) / 500),
    "shap":            (timeit(lambda: expl.shap_values(x1, check_additivity=False), 20),
                        timeit(lambda: expl.shap_values(Xs, check_additivity=False)) / 500),
}
res = pd.DataFrame({k: [v[0] * 1e3, v[1] * 1e3] for k, v in T.items()},
                   index=["single flow (ms)", "per flow, batch 500 (ms)"])
res["shap_vs_sklearn_x"] = res.shap / res.sklearn_predict
res["shap_vs_lean_x"] = res.shap / res.lean_predict
print(res.round(4).to_string())
res.to_csv("reports/metrics/shap_overhead.csv")

sv = expl.shap_values(Xs, check_additivity=False)
sv = sv[1] if isinstance(sv, list) else (sv[..., 1] if sv.ndim == 3 else sv)
print(pd.Series(np.abs(sv).mean(0), index=names).sort_values(ascending=False).head(10).round(4))