import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib, numpy as np, pandas as pd, shap, streamlit as st
from src.reasons import reasons, pos_shap

st.set_page_config(page_title="XAI IDS for SDN", layout="wide")
ART = ROOT / "models" / "artifacts_foldA.joblib"
if not ART.exists():
    st.error("Run `python -m src.build_artifacts` first."); st.stop()

@st.cache_resource
def load():
    a = joblib.load(ART)
    return a, {k: shap.TreeExplainer(v["rf"]) for k, v in a["models"].items()}

a, expl = load()
base = a["base"]

st.title("Explainable IDS for SDN (InSDN)")
st.caption("Trained on the OVS session; the sample flows come from the held-out meta2 session. "
           "Scores are uncalibrated RF attack probabilities. Attack-type prediction is not in this demo.")

tab1, tab2 = st.tabs(["Detect + explain", "Measured results"])

with tab1:
    tag = st.sidebar.radio("Model", ["full", "behavioral"])
    thr = st.sidebar.slider("Attack threshold", 0.1, 0.9, 0.5, 0.05)
    mode = st.sidebar.radio("Input", ["Sample flows (held-out session)", "Upload CSV (InSDN format)"])
    if tag == "full":
        st.info("Full RF: best cross-session detection (Fold A AUC ~0.98), but its top features are "
                "environment-tied (TCP window, ports, header lengths).")
    else:
        st.warning("Behavioral RF: environment-tied features removed. Measured cross-session AUC is about "
                   "0.47 (Fold A) and 0.58 (Fold B), roughly chance. Reasons are behavioral but detection is unreliable.")

    if mode.startswith("Sample"):
        df = a["sample"]
    else:
        up = st.file_uploader("CSV", type="csv")
        if up is None:
            st.stop()
        df = pd.read_csv(up, low_memory=False).head(5000)
        df.columns = df.columns.str.strip()
        missing = [c for c in base if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing[:5]}..."); st.stop()
    m = a["models"][tag]
    X = df[m["feats"]].replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32")
    Xt = m["pre"].transform(X)
    p = m["rf"].predict_proba(Xt)[:, 1]
    flagged = p >= thr

    c1, c2, c3 = st.columns(3)
    c1.metric("Flows", len(df)); c2.metric("Flagged", int(flagged.sum()))
    c3.metric("Flagged %", f"{flagged.mean():.1%}")
    table = pd.DataFrame({"attack_score": p.round(3), "verdict": np.where(flagged, "ATTACK", "normal")})
    if "Label" in df:
        table["true_label"] = df["Label"].values
    st.dataframe(table, height=250, use_container_width=True)

    idx = np.where(flagged)[0]
    if len(idx) == 0:
        st.success("No flows flagged."); st.stop()
    i = st.selectbox("Explain flagged flow (row)", idx)
    t0 = time.perf_counter()
    sv = pos_shap(expl[tag].shap_values(Xt[[i]], check_additivity=False))[0]
    ms = (time.perf_counter() - t0) * 1e3
    st.subheader(f"Attack detected (score {p[i]:.0%})")
    if "Label" in df:
        st.caption(f"Ground truth (sample mode): {df['Label'].iloc[i]}")
    for line in reasons(Xt[i], sv, m["names"], m["ref"]):
        st.write("- " + line)
    st.caption(f"SHAP computed for this flow only, on flagged flows only: {ms:.1f} ms.")
    st.bar_chart(pd.Series(sv, index=m["names"]).sort_values().tail(10))

with tab2:
    for f, title in [("rf_xgb_binary.csv", "Binary: random vs scenario splits"),
                     ("rf_ablation_agg.csv", "RF feature ablation (5 seeds)"),
                     ("behavioral_vs_full.csv", "Full vs behavioral RF"),
                     ("multiclass.csv", "Multiclass"),
                     ("shap_overhead.csv", "SHAP overhead")]:
        path = ROOT / "reports" / "metrics" / f
        if path.exists():
            st.subheader(title)
            st.dataframe(pd.read_csv(path), use_container_width=True)