import time, joblib, numpy as np, pandas as pd, shap
from src.splits import prepare, scenario_folds
from src.data import feature_cols
from src.train import xy

THR = 0.5
pre, rf, names = joblib.load("models/rf_foldA.joblib")
rf.set_params(n_jobs=1)
df = prepare(); feats = feature_cols(df)
_, te = scenario_folds(df)["A_ovs->meta2"]
Xte, yte = xy(te.sample(5000, random_state=0), feats)
X = pd.DataFrame(pre.transform(Xte), columns=names)
expl = shap.TreeExplainer(rf)

def sv1(v):
    return v[1] if isinstance(v, list) else (v[..., 1] if v.ndim == 3 else v)

def run(explain_all=False):
    t = time.perf_counter()
    p = rf.predict_proba(X)[:, 1]
    flagged = np.ones(len(X), bool) if explain_all else p >= THR
    sv = sv1(expl.shap_values(X[flagged], check_additivity=False)) if flagged.any() else None
    return time.perf_counter() - t, p, flagged, sv

t0 = time.perf_counter(); p = rf.predict_proba(X)[:, 1]; t_pred = time.perf_counter() - t0
t_gate, p, flagged, sv = run()
t_all, *_ = run(explain_all=True)
n = len(X)
print(pd.DataFrame({
    "ms_per_flow": [t_pred / n * 1e3, t_gate / n * 1e3, t_all / n * 1e3],
    "flows_explained": [0, int(flagged.sum()), n]},
    index=["predict only", "predict + SHAP on flagged", "predict + SHAP on all"]).round(3))
print("flagged fraction:", round(flagged.mean(), 3))

Xf = X[flagged].reset_index(drop=True)
pf = p[flagged]
for i in range(3):
    top = np.argsort(-sv[i])[:3]
    print(f"\nAttack detected ({pf[i]:.0%} score). Reasons:")
    for j in top:
        print(f"- {names[j]} = {Xf.iloc[i, j]:.4g} (raises attack score by {sv[i, j]:.3f})")

print("\nGating benefit vs attack rate (predict + SHAP on flagged, ms/flow):")
rng = np.random.default_rng(0)
atk, nrm = np.where(yte == 1)[0], np.where(yte == 0)[0]
for rate in [0.05, 0.20, 0.50]:
    n_atk = int(rate * 1000); n_nrm = 1000 - n_atk
    idx = np.r_[rng.choice(atk, n_atk, replace=False), rng.choice(nrm, min(n_nrm, len(nrm)), replace=False)]
    Xm = X.iloc[idx]
    t = time.perf_counter()
    pm = rf.predict_proba(Xm.values)[:, 1]; fm = pm >= THR
    if fm.any():
        expl.shap_values(Xm[fm], check_additivity=False)
    print(f"attack rate {rate:.0%}: {(time.perf_counter() - t) / len(Xm) * 1e3:.3f} ms/flow, flagged {fm.mean():.1%}")