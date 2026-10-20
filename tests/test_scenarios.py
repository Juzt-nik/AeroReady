"""Scenario fairness and baseline interpretation for the availability simulator."""
from datetime import date

import pytest

from aeroready import availability, db, risk, workorders


@pytest.fixture
def results(fresh_db, engine):
    scored = risk.score_components(engine, db.table("components"))
    return scored, availability.run_scenarios(scored, horizon=30, today=date.today())


def test_proactive_and_reactive_use_identical_inputs(results):
    """Same fleet, horizon, existing work orders and at-risk population - only the intervention differs."""
    scored, res = results
    pro, rea = res["proactive"]["inputs"], res["reactive"]["inputs"]
    assert pro == rea
    assert pro["tails"] == tuple(sorted(db.table("aircraft")["tail_no"]))
    assert pro["horizon"] == 30
    expected = tuple(sorted(availability.at_risk_unaddressed(scored)["component_id"]))
    assert pro["at_risk_components"] == expected
    assert len(expected) > 0, "the demo dataset should contain unaddressed High/Critical components"
    active = db.query_df("SELECT component_id FROM work_orders WHERE status IN ('Open','Awaiting Parts',"
                         "'Scheduled','In Progress')")
    assert not set(expected) & set(active["component_id"]), "components with a work order are not 'unaddressed'"


def test_at_risk_population_is_exactly_high_and_critical(results):
    scored, res = results
    risky = availability.at_risk_unaddressed(scored)
    assert set(risky["risk_level"]) <= {"High", "Critical"}
    assert set(scored[scored["risk_level"].isin(["High", "Critical"])]["component_id"]) >= set(risky["component_id"])


def test_existing_work_orders_only_baseline_excludes_predicted_failures(results):
    """The baseline contains no synthetic at-risk jobs, so it must not be read as predictive-maintenance value."""
    _, res = results
    assert res["current"]["inputs"]["at_risk_components"] == ()
    assert availability.SCENARIOS["current"] == "Existing work orders only"
    assumption = availability.ASSUMPTIONS["current"]
    assert "EXCLUDED" in assumption and "do not compare" in assumption.lower()
    # every scenario starts from the same existing work orders
    base = res["current"]["inputs"]["existing_work_orders"]
    assert all(r["inputs"]["existing_work_orders"] == base for r in res.values())


def test_scenario_groups_separate_intervention_from_resources():
    assert availability.INTERVENTION_SCENARIOS == ["reactive", "proactive"]
    assert "current" in availability.RESOURCE_SCENARIOS
    assert not set(availability.INTERVENTION_SCENARIOS) & set(availability.RESOURCE_SCENARIOS)
    assert set(availability.INTERVENTION_SCENARIOS) | set(availability.RESOURCE_SCENARIOS) == set(availability.SCENARIOS)


def test_intervention_comparison_is_reactive_minus_proactive(results):
    _, res = results
    cmp = availability.intervention_comparison(res)
    r, p = (res[k]["metrics"]["aircraft_days_lost"] for k in ("reactive", "proactive"))
    assert (cmp["reactive_days"], cmp["proactive_days"], cmp["improvement"]) == (r, p, r - p)
    assert cmp["same_inputs"] is True
    assert "not a forecast" in cmp["text"].lower() and "user-configured" in cmp["text"].lower()


@pytest.mark.parametrize("reactive,proactive,word", [(60, 40, "fewer"), (40, 60, "worse"), (50, 50, "same")])
def test_intervention_text_states_direction_explicitly(reactive, proactive, word):
    inputs = {"at_risk_components": ("A-1",)}
    res = {"reactive": {"metrics": {"aircraft_days_lost": reactive}, "inputs": inputs},
           "proactive": {"metrics": {"aircraft_days_lost": proactive}, "inputs": inputs}}
    cmp = availability.intervention_comparison(res)
    assert cmp["improvement"] == reactive - proactive
    assert word in cmp["text"]


def test_addressing_a_risk_removes_it_from_both_strategies(fresh_db, engine):
    scored = risk.score_components(engine, db.table("components"))
    target = availability.at_risk_unaddressed(scored).iloc[0]
    workorders.create_work_order(target["tail_no"], target["component_id"], "Inspect & Service", target["risk_level"])
    res = availability.run_scenarios(scored, horizon=30)
    for key in ("proactive", "reactive"):
        assert target["component_id"] not in res[key]["inputs"]["at_risk_components"]
    assert res["proactive"]["inputs"] == res["reactive"]["inputs"]
