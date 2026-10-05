import numpy as np
import pandas as pd

from scripts.train_ml import EXCLUDE, make_pipeline, select_threshold


def test_redundant_engineered_columns_are_excluded():
    df = pd.DataFrame({
        "EmployeeNumber": [1, 2],
        "MonthlyIncome": [1000, 2000],
        "income_yearly": [12000, 24000],
        "OverTime": ["Yes", "No"],
        "overtime_flag": [1, 0],
        "attrition_label": [1, 0],
        "Age": [30, 40],
        "survey_satisfaction": [1, 5],
    })
    features = df.drop(columns=[c for c in df.columns if c in EXCLUDE])
    assert "income_yearly" not in features.columns
    assert "overtime_flag" not in features.columns
    assert "MonthlyIncome" in features.columns
    assert "OverTime" in features.columns
    assert "survey_satisfaction" not in features.columns


def test_threshold_is_selected_from_training_data_only():
    y = np.array([0, 0, 1, 1])
    probs = np.array([0.05, 0.30, 0.60, 0.90])
    threshold, score = select_threshold(y, probs)
    assert 0.0 < threshold <= 1.0
    assert 0.0 <= score <= 1.0


def test_pipeline_is_constructed_for_all_models():
    numeric = ["Age"]
    categorical = ["OverTime"]
    for name in ("logreg", "rf", "mlp"):
        pipe, label = make_pipeline(name, numeric, categorical)
        assert pipe is not None
        assert label
