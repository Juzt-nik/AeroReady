"""Model output must be decision support, never an airworthiness or grounding determination."""
import re

import pytest

from aeroready import config, db, risk, ui, workorders

FORBIDDEN = [r"ground aircraft", r"ground the aircraft", r"within 24 ?h", r"must be (replaced|grounded)",
             r"airworthy", r"cleared (for|to) fl"]


@pytest.mark.parametrize("level", config.RISK_LEVELS)
@pytest.mark.parametrize("telemetry", ["Fresh", "Stale", "No data"])
def test_recommendation_wording(level, telemetry):
    rec = risk.recommend(level, ["vibration", "temp_deviation"], {"life_used_pct": 120}, telemetry)
    text = rec["text"]
    assert "does not determine aircraft serviceability" in text
    for pattern in FORBIDDEN:
        assert not re.search(pattern, text, re.I), (level, pattern)
    if level != "Low":
        assert "authorized maintenance personnel" in text and "simulated maintenance risk" in text
    assert "grounding" not in rec, "recommendations must not carry a grounding decision"


def test_recommendation_keeps_severity_action_and_window_visible():
    rec = risk.recommend("High", ["vibration"], {"life_used_pct": 50})
    assert rec["action"] == "Inspect & Service" and rec["due_days"] == 3
    assert risk.recommend("Critical", [], {"life_used_pct": 50})["action"] == "Replace"
    assert risk.recommend("High", [], {"life_used_pct": 105})["action"] == "Replace"  # beyond rated life


def test_exact_wording_for_high_risk():
    text = risk.recommend("High", [], {})["text"]
    assert text.startswith("High simulated maintenance risk. Prioritize review by authorized maintenance personnel "
                           "under approved procedures.")


def test_no_grounding_in_policy_assessment_or_scored_frame(fresh_db, engine):
    assert all(len(v) == 2 for v in risk.CATEGORY_POLICY.values())
    a = risk.assess(engine, dict(life_used_pct=120, hours_since_maintenance=1400, age_years=25, prior_faults=8,
                                 temp_deviation=50, vibration=18, pressure_deviation=35))
    assert a["level"] == "Critical" and "grounding" not in a
    assert "grounding" not in risk.score_components(engine, db.table("components")).columns


def test_disclaimer_and_score_labels_are_not_probability_claims():
    text = config.DISCLAIMER.lower()
    assert "not calibrated failure probabilities" in text and "airworthiness" in text
    assert "authorized" in text


def test_work_order_does_not_ground_unless_explicitly_requested(fresh_db):
    res = workorders.create_work_order("FT-201", "FT-201-HP1", "Replace", "Critical")
    row = db.query_one("SELECT grounding FROM work_orders WHERE wo_id = ?", (res["wo_id"],))
    assert row["grounding"] == 0
