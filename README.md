# Explainable AI-Based Intrusion Detection for SDN

An ML-based Intrusion Detection System (IDS) for Software-Defined Networks, built on the InSDN dataset. It classifies flows as normal or attack, explains individual alerts with SHAP, and tests whether the detection performance and the explanations hold up **across capture sessions**, not just on a random split.

Course project: Cryptography and Network Security.
Motivation: the SoK paper *"Software-Defined Networks Security With Machine Learning"* (Örsdemir et al., IEEE Access, 2026) reports questionable validity of near-perfect accuracy, outdated datasets, overfitting, missing efficiency measurements and missing explainability. This project takes those gaps as motivation. It does not implement the paper's hybrid architecture.

## Dataset

https://ieeexplore.ieee.org/document/9187858

## Key findings (as measured)

- **Random splits are misleading.** Macro-F1 is 0.9998 for RF and XGBoost on a random split, but 0.47-0.82 on scenario-based splits (train on one capture session, test on another).
- **RF is the most stable model under shift.** XGBoost has the worst cross-session AUC (0.69 / 0.73). Logistic Regression has a competitive macro-F1 in one fold. Complexity did not help under shift.
- **SHAP exposed a validity problem that accuracy hid.** The top features of the best-performing RF are environment-tied: `Init Bwd Win Byts`, `Dst Port`, header lengths and (in Fold B) `Src Port`.
- **Removing environment-tied features removes cross-session detection.** The behavioral-only RF has AUC 0.47 (Fold A) and 0.58 (Fold B), roughly chance. In this setup, detection that transfers and explanations based on traffic behavior trade off.
- **SHAP is costly.** About 6-12 ms per flow vs about 0.03-0.16 ms for prediction alone (roughly two orders of magnitude), so the dashboard explains flagged flows only.

These findings come from two folds, a limited number of seeds and one dataset. They are evidence within those limits, not general claims about InSDN or XAI-based IDS.

## Dataset

[InSDN](https://ieeexplore.ieee.org/document/9187858) (Elsayed et al., IEEE Access, 2020): SDN-native flows (Mininet + ONOS), 80+ CICFlowMeter features. The CSVs are not redistributed. Download them (Kaggle or IEEE DataPort) into `data/raw/`:

- `Normal_data.csv`
- `OVS.csv`
- `metasploitable-2.csv`

Raw size: 343,889 flows. After cleaning: 223,303.

## Methodology

**Preprocessing and leakage control**
- Dropped identifiers: `Flow ID`, `Src IP`, `Dst IP`, `Timestamp`.
- Removed duplicate feature vectors. Removed vectors that appear in more than one capture session. Duplicates were heavy in DDoS (about 120k rows).
- Constant-column removal and feature selection (RF importance, top 30) are fitted **on training data only**, inside a scikit-learn `Pipeline`.
- Timestamps were not used for splitting because they were unreliable (date parsing ambiguity).

**Splits**
- `A_ovs->meta2` and `B_meta2->ovs`: train on one attack session, test on the other. Normal traffic exists in one session only, so it is split by `Src IP` group (host-disjoint, size-balanced greedily; resulting split is 79/21).
- `random`: stratified 70/30, reported only as an optimistic reference.
- Multiclass: 7 classes on a random split. Cross-scenario multiclass uses 4 shared classes (Normal, DoS/DDoS, Probe, BFA); DoS and DDoS are merged because only 25 DDoS rows remain in the OVS session after deduplication.

**Models:** Logistic Regression (baseline), Random Forest, XGBoost.

**Metrics:** macro-F1, precision, recall (per class), ROC-AUC, PR-AUC, confusion matrices. Accuracy is not used as a headline metric.

**XAI:** SHAP `TreeExplainer` on RF. Text reasons compare each top feature with the interquartile range of Normal training traffic and tag environment-sensitive features.

## Results

Binary detection (macro-F1 / ROC-AUC; single seed, 0.5 threshold):

| Split | RF | XGBoost |
|---|---|---|
| Random | 0.9998 / 1.0000 | 0.9998 / 1.0000 |
| A: OVS -> meta2 | 0.7463 / 0.9813 | 0.4654 / 0.6909 |
| B: meta2 -> OVS | 0.8210 / 0.8453 | 0.4698 / 0.7257 |

Logistic Regression macro-F1: 0.7034 (A), 0.8254 (B). AUC was not computed.

RF with 5 seeds (varying model and selector seeds, folds fixed): Fold A 0.7732 ± 0.0462, Fold B 0.6811 ± 0.1132 macro-F1. Fold B is unstable, so feature-set rankings there are inconclusive.

Full vs behavioral RF (3 seeds, environment-tied features removed: `Init Bwd Win Byts`, `Src Port`, `Dst Port`, `Fwd/Bwd Header Len`):

| Fold | Features | Macro-F1 | AUC |
|---|---|---|---|
| A | full | 0.7548 ± 0.0357 | 0.9826 |
| A | behavioral | 0.4136 ± 0.0111 | 0.4689 |
| B | full | 0.6414 ± 0.1631 | 0.8409 |
| B | behavioral | 0.5734 ± 0.0006 | 0.5836 |

Multiclass macro-F1: random 7-class RF 0.9516, XGBoost 0.9622, LogReg 0.4481. Cross-scenario 4-class: 0.26-0.45, with per-class recall that swings between folds. Attack-type identification does not transfer across sessions in this setup.

SHAP overhead (RF, 200 trees, 1 thread, batch): about 6-12 ms per flow vs 0.03-0.16 ms for prediction. Timings varied by about 25% between identical runs and were taken on a shared machine. Single-flow timings are not reliable and are not reported. Gated SHAP (flagged flows only) at a 5% attack rate: 6.3 ms per flow average, with 62% of flows flagged, because of the model's high false-alarm rate on the held-out Normal traffic.

All metric CSVs are in `reports/metrics/`.

## Setup

Python 3.13 (tested). Windows PowerShell commands shown.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

Place the InSDN CSVs in `data/raw/`.

## Reproduce

Run from the repository root, in this order:

```powershell
python -m src.data               # load + EDA summary
python -m src.eda_checks         # label/duplicate checks
python -m src.eda_checks2        # post-dedup counts
python -m src.splits             # inspect scenario folds
python -m src.train              # Logistic Regression baseline + ablation
python -m src.models             # RF / XGBoost: random vs scenario splits
python -m src.ablate_rf          # RF feature ablation, 5 seeds
python -m src.explain            # SHAP + overhead (also saves models/rf_foldA.joblib)
python -m src.explain_flagged    # gated SHAP + overhead vs attack rate
python -m src.multiclass         # multiclass evaluation
python -m src.compare_behavioral # full vs behavioral RF
python -m src.build_artifacts    # models for the dashboard
streamlit run app/streamlit_app.py
```

Run timing scripts on an otherwise idle machine. Runtime is roughly 2-15 minutes per script.

## Dashboard

`app/streamlit_app.py` provides two tabs.
- **Detect + explain:** choose the Full or Behavioral RF, set the attack threshold, use sample flows from the held-out session or upload an InSDN-format CSV. It shows the per-flow attack score, the top SHAP reasons with Normal-range context and an `[environment-sensitive]` tag, and a SHAP bar chart for flagged flows.
- **Measured results:** the metric tables from `reports/metrics/`.

The attack score is an uncalibrated RF probability, not a confidence.

## Project structure

```
xai-ids-sdn/
├── data/{raw,processed}/
├── src/
│   ├── data.py               # loading, cleaning, dedupe, ENV_FEATURES
│   ├── splits.py             # prepare(), scenario folds, random split
│   ├── features.py           # variance filter + RF feature selection
│   ├── train.py              # Logistic Regression baseline + ablation
│   ├── models.py             # RF / XGBoost binary evaluation
│   ├── multiclass.py         # multiclass evaluation
│   ├── ablate_rf.py          # RF feature-ablation across seeds
│   ├── compare_behavioral.py # full vs behavioral RF
│   ├── explain.py            # SHAP + overhead measurement
│   ├── explain_flagged.py    # gated SHAP + text reasons
│   ├── reasons.py            # SHAP -> readable reasons
│   ├── build_artifacts.py    # models for the dashboard
│   ├── diag.py               # single-feature shortcut diagnostic
│   └── eda_checks*.py
├── app/streamlit_app.py
├── notebooks/
├── reports/{figures,metrics}/
├── models/
├── requirements.txt
└── README.md
```

## Limitations

- Normal traffic comes from a single capture session, so Normal recall measures host shift, not full environment shift. The Normal split is 79/21 by flow count.
- DDoS has only 25 OVS rows after deduplication; per-class DDoS results are not meaningful. U2R (17 flows), Web-Attack (192) and BOTNET (164) appear only in the random multiclass split.
- Seeds vary the model and selector only. Fold design is fixed, so split-level variance is not captured.
- The ablation feature sets were chosen after viewing SHAP output on test folds, so those comparisons are exploratory.
- The decision threshold (0.5) is untuned, and the reported scores are uncalibrated. The binary AUC for Logistic Regression is not computed.
- The XGBoost multiclass model has no class weighting. Hyperparameters were not tuned for any model.
- SHAP reasons are faithful to the model but not necessarily to attack semantics. The current reasons describe capture-session artifacts such as TCP window size and ports. The "unusually high packet rate" style of output is not implemented.
- The dashboard does not predict the attack type.
- CPU and memory usage are not measured yet; only inference and SHAP latency are.
- SDN controller integration (ONOS/Ryu) and live traffic are not implemented.

## Future work

- Measure CPU and memory overhead (`psutil`) and model size.
- Add attack-type prediction to the dashboard.
- Repeat the analysis with additional datasets or capture sessions to separate environment shift from attack behavior.
- Optional SDN controller integration.

## References

- M. S. Elsayed, N.-A. Le-Khac, A. D. Jurcut, "InSDN: A Novel SDN Intrusion Dataset," *IEEE Access*, vol. 8, 2020.
- A. Örsdemir, U. Tefek, E. Esiner, A. Küpçü, "SoK: Software-Defined Networks Security With Machine Learning," *IEEE Access*, vol. 14, 2026.