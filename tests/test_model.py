import numpy as np
import pandas as pd

from aeroready import config, model, risk


def test_training_data_is_reproducible():
    a = model.generate_training_data(500, seed=7)
    b = model.generate_training_data(500, seed=7)
    pd.testing.assert_frame_equal(a, b)
    assert set(a["failure_within_50h"].unique()) <= {0, 1}


def test_metrics_are_held_out_and_reasonable(engine):
    m = engine.metrics
    total = engine.n_train + engine.n_val + engine.n_test
    assert engine.n_train > 0 and engine.n_val > 0 and engine.n_test > 0
    assert abs(engine.n_train / total - 0.60) < 0.01
    assert abs(engine.n_val / total - 0.20) < 0.01
    assert abs(engine.n_test / total - 0.20) < 0.01
    assert 0.5 < m["roc_auc"] <= 1.0
    # held-out metric is reported separately from training performance
    assert m["train_roc_auc"] >= m["ml_test"]["roc_auc"] - 0.05


def test_method_is_selected_on_validation_metrics(engine):
    """The method must match the validation-based rule - not the test-set ROC-AUC."""
    m = engine.metrics
    assert engine.method == model.select_method(m["val_roc_auc_ml"], m["val_roc_auc_rules"])
    assert m["active_method"] == engine.method


def test_select_method_rule():
    floor, margin = config.MIN_ACCEPTABLE_AUC, config.ML_MIN_AUC_ADVANTAGE
    assert model.select_method(0.90, 0.80) == "ml"
    assert model.select_method(0.80, 0.90) == "rules"          # ML must beat the baseline on validation
    assert model.select_method(floor - 0.01, floor - 0.2) == "rules"  # below the absolute floor
    assert model.select_method(0.85 + margin, 0.85) == "ml"


def test_selection_never_looks_at_the_test_split(monkeypatch):
    """Scrambling the TEST labels must change neither the validation AUCs nor the chosen method."""
    base = model.train_engine(n=2500, seed=11)
    real_split = model.split_data

    def scrambled(X, y, seed=config.RANDOM_SEED):
        X_tr, X_va, X_te, y_tr, y_va, y_te = real_split(X, y, seed)
        return X_tr, X_va, X_te, y_tr, y_va, 1 - y_te  # test labels deliberately wrong

    monkeypatch.setattr(model, "split_data", scrambled)
    other = model.train_engine(n=2500, seed=11)
    assert other.method == base.method
    assert other.metrics["val_roc_auc_ml"] == base.metrics["val_roc_auc_ml"]
    assert other.metrics["val_roc_auc_rules"] == base.metrics["val_roc_auc_rules"]
    assert other.metrics["ml_test"]["roc_auc"] != base.metrics["ml_test"]["roc_auc"]  # test metrics did change


def test_splits_are_disjoint_stratified_60_20_20():
    data = model.generate_training_data(3000, seed=5)
    X, y = data[config.FEATURES], data["failure_within_50h"]
    parts = model.split_data(X, y, seed=5)
    X_tr, X_va, X_te, y_tr, y_va, y_te = parts
    ids = [set(p.index) for p in (X_tr, X_va, X_te)]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    assert sum(len(i) for i in ids) == len(X)
    assert abs(len(X_tr) / len(X) - 0.6) < 0.01 and abs(len(X_va) / len(X) - 0.2) < 0.01
    for yy in (y_tr, y_va, y_te):
        assert abs(yy.mean() - y.mean()) < 0.01


def test_both_methods_are_reported_on_the_test_split(engine):
    m = engine.metrics
    for key in ("ml_test", "rules_test"):
        assert {"roc_auc", "average_precision", "brier", "confusion"} <= set(m[key])
    assert m["rule_based_roc_auc"] == m["rules_test"]["roc_auc"]


def test_model_version_was_bumped_for_new_evaluation_design():
    assert config.MODEL_VERSION >= 5


def test_score_semantics_do_not_claim_calibrated_probability(engine):
    text = engine.score_semantics.lower()
    assert "not" in text and "probability" in text
    fallback = model.RiskEngine(model=None, method="rules", reference=engine.reference)
    assert "weighted index" in fallback.score_semantics and "70%" in fallback.score_semantics


def test_risk_is_monotonic_in_every_feature(engine):
    base = dict(life_used_pct=50, hours_since_maintenance=250, age_years=10, prior_faults=1,
                temp_deviation=3, vibration=2.5, pressure_deviation=2)
    for feat in config.FEATURES:
        lo, hi, _ = config.FEATURE_RANGES[feat]
        grid = [dict(base, **{feat: v}) for v in np.linspace(lo, hi, 15)]
        p = engine.predict(pd.DataFrame(grid))
        assert np.all(np.diff(p) >= -1e-9), f"risk decreases as {feat} increases"


def test_risk_category_bands():
    assert model.risk_category(0.0) == "Low"
    assert model.risk_category(0.19) == "Low"
    assert model.risk_category(0.20) == "Medium"
    assert model.risk_category(0.45) == "High"
    assert model.risk_category(0.70) == "Critical"
    assert model.risk_category(1.0) == "Critical"


def test_rule_score_bounded_and_fallback(engine):
    data = model.generate_training_data(300, seed=1)[config.FEATURES]
    s = model.rule_score(data)
    assert s.min() >= 0 and s.max() <= 1
    fallback = model.RiskEngine(model=None, method="rules", reference=engine.reference)
    assert np.allclose(fallback.predict(data), s)
    expl = fallback.explain(data.iloc[0].to_dict())
    assert np.isclose(expl["contribution"].sum(), s[0])


def test_assessment_responds_to_sensor_changes(engine):
    healthy = dict(life_used_pct=40, hours_since_maintenance=150, age_years=6, prior_faults=0,
                   temp_deviation=2, vibration=2.0, pressure_deviation=1.5)
    worse = dict(healthy, temp_deviation=35, vibration=11, pressure_deviation=20)
    a, b = risk.assess(engine, healthy), risk.assess(engine, worse)
    assert b["score"] > a["score"]
    assert config.SEVERITY_RANK[b["level"]] > config.SEVERITY_RANK[a["level"]]
    assert b["factors"], "a high-risk assessment should name contributing factors"
    assert "Largest differences" in b["text"]


def test_missing_values_are_imputed(engine):
    row = dict(life_used_pct=np.nan, hours_since_maintenance=100, age_years=np.nan, prior_faults=0,
               temp_deviation=2, vibration=2, pressure_deviation=1)
    assert 0 <= engine.predict_one(row) <= 1
