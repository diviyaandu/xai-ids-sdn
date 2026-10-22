import numpy as np, pandas as pd
from src.data import ENV_FEATURES

def normal_ref(X_t, y, names):
    N = pd.DataFrame(X_t[y == 0], columns=names)
    return N.quantile([.25, .5, .75]).T.set_axis(["q1", "p50", "q3"], axis=1)

def pos_shap(v):
    return v[1] if isinstance(v, list) else (v[..., 1] if v.ndim == 3 else v)

def reasons(x, sv, names, ref, k=3):
    out = []
    for j in np.argsort(-sv)[:k]:
        if sv[j] <= 0:
            break
        n, v, r = names[j], x[j], ref.loc[names[j]]
        lvl = "above" if v > r.q3 else "below" if v < r.q1 else "typical"
        tag = " [environment-sensitive]" if n in ENV_FEATURES else ""
        out.append(f"{n} = {v:.4g} ({lvl} for normal traffic, IQR {r.q1:.4g}-{r.q3:.4g}, "
                   f"median {r.p50:.4g}); raises attack score by {sv[j]:.3f}{tag}")
    return out