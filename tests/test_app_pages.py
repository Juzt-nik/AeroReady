"""Headless UI smoke tests: every page renders without exceptions, and key interactions work."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from aeroready import db

ROOT = Path(__file__).resolve().parent.parent
PAGES = ["app.py", "views/dashboard.py", "views/predictive.py", "views/digital_twin.py", "views/planner.py",
         "views/inventory.py", "views/simulator.py", "views/data_hub.py"]


def run(page, **state):
    at = AppTest.from_file(str(ROOT / page), default_timeout=120)
    for k, v in state.items():
        at.session_state[k] = v
    return at.run()


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(fresh_db, page):
    at = run(page)
    assert not at.exception, [e.value for e in at.exception]


def test_twin_slider_changes_risk_and_creates_work_order(fresh_db):
    at = run("views/digital_twin.py", sel_tail="FT-201", sel_comp="FT-201-HP1")

    def gauge():
        for m in at.markdown:
            if "ar-gauge" in m.value:
                return int(m.value.split("ar-gauge' style='color:")[1].split(">")[1].split("<")[0])

    low = gauge()
    at.slider(key="tw_FT-201-HP1_vibration").set_value(15.0).run()
    at.slider(key="tw_FT-201-HP1_temp_deviation").set_value(40.0).run()
    assert not at.exception
    assert gauge() > low
    at.button(key="tw_FT-201-HP1_create").click().run()
    assert not at.exception
    assert db.query_one("SELECT COUNT(*) n FROM work_orders WHERE component_id='FT-201-HP1'")["n"] == 1


def test_planner_generate_and_apply(fresh_db):
    at = run("views/planner.py")
    at.button[[b.label for b in at.button].index("Generate suggested schedule")].click().run()
    assert not at.exception
    at.button[[b.label for b in at.button].index("Apply suggested schedule")].click().run()
    assert not at.exception
    assert db.query_one("SELECT COUNT(*) n FROM work_orders WHERE status='Scheduled'")["n"] > 1


# --------------------------------------------------------------------------- regression: review findings
def _text(at):
    return " ".join(m.value for m in at.markdown) + " " + " ".join(c.value for c in at.caption)


def test_simulator_separates_intervention_from_resource_comparisons(fresh_db):
    at = run("views/simulator.py")
    assert not at.exception, [e.value for e in at.exception]
    names = [s.value for s in at.subheader]
    assert any("proactive vs reactive" in n for n in names)
    assert any("Resource planning" in n for n in names)
    # default chart = reactive vs proactive only
    intervention = at.multiselect(key="plot_intervention")
    assert intervention.value == ["reactive", "proactive"]
    text = _text(at)
    assert "not forecasts of actual fleet availability" in text
    assert "not a forecast of actual fleet availability" in text  # appended to the proactive-vs-reactive reading
    assert "Existing work orders only" in text


def test_grounding_is_never_preset_by_the_model(fresh_db):
    at = run("views/digital_twin.py", sel_tail="FT-201", sel_comp="FT-201-HP1")
    at.slider(key="tw_FT-201-HP1_vibration").set_value(18.0).run()
    at.slider(key="tw_FT-201-HP1_temp_deviation").set_value(50.0).run()
    assert not at.exception
    box = at.checkbox(key="tw_FT-201-HP1_ground")
    assert box.value is False
    assert "Simulated work-order decision" in box.label
    text = _text(at)
    assert "does not determine aircraft serviceability" in text
    assert "Ground aircraft and replace" not in text


def test_bulk_work_orders_do_not_set_grounding(fresh_db):
    before = set(db.table("work_orders")["wo_id"])
    at = run("views/predictive.py")
    at.button[[b.label for b in at.button].index("Create all recommended work orders")].click().run()
    assert not at.exception
    new = db.table("work_orders")
    new = new[~new["wo_id"].isin(before)]
    assert len(new) > 0 and (new["grounding"] == 0).all()


def test_data_hub_shows_sources_and_simulated_connector_is_idempotent(fresh_db):
    at = run("views/data_hub.py")
    assert not at.exception, [e.value for e in at.exception]
    go = at.button(key="sim_go")
    go.click().run()
    assert not at.exception
    n1 = db.query_one("SELECT COUNT(*) n FROM sensor_readings WHERE source='sim-api-v1'")["n"]
    assert n1 == 5
    at.button(key="sim_go").click().run()
    assert db.query_one("SELECT COUNT(*) n FROM sensor_readings WHERE source='sim-api-v1'")["n"] == n1
