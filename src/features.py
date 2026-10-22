import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.ensemble import RandomForestClassifier

def make_preproc(selector_k_est=100, seed=42):
    return [
        ("var", VarianceThreshold(0.0)),
        ("sel", SelectFromModel(
            RandomForestClassifier(n_estimators=selector_k_est, max_depth=12,
                                   n_jobs=-1, random_state=seed),
            max_features=30, threshold=-np.inf)),
    ]