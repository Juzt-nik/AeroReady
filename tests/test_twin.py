"""Digital-twin lifecycle, state derivation and trend guard rails."""
from datetime import date, datetime, timedelta

import pandas as pd
import pytest

from aeroready import config, db, risk, twin, workorders

CID, TAIL = "FT-201-HP1", "FT-201"  # hydraulic pump -> T-HYD


def score(engine):
    return risk.score_components(engine, db.table("components")).set_index("component_id")


def state_of(engine, cid=CID):
    sc = score(engine).reset_index()
    return twin.twin_states(sc)[cid]


def test_component_state_rules():
    now = datetime(2026, 6, 15, 12, 0)
    f = twin.component_state
    assert f("Low", None, None, now) == "Serviceable"
    assert f("High", None, None, now) == "Degraded"
    assert f("Critical", "Awaiting Parts", None, now) == "Under inspection"
    assert f("Low", "Scheduled", None, now) == "Under inspection"
    assert f("Critical", "In Progress", None, now) == "In maintenance"
    recent = (now - timedelta(days=1)).isoformat(sep=" ")
    old = (now - timedelta(days=config.RETURNED_TO_SERVICE_DAYS + 1)).isoformat(sep=" ")
    assert f("Low", None, recent, now) == "Returned to service"
    assert f("Low", None, old, now) == "Serviceable"
    assert f("High", None, old, now) == "Degraded"
    assert set(twin.STATE_NOTES) == set(config.TWIN_STATES)


def test_lifecycle_updates_state_history_record_and_risk(fresh_db, engine):
    assert state_of(engine) == "Serviceable"
    workorders.record_reading(CID, 40.0, 15.0, 30.0)  # simulated deterioration, stored as a new reading
    degraded = score(engine).loc[CID]
    assert degraded["risk_level"] in ("High", "Critical")
    assert state_of(engine) == "Degraded"
    src = db.query_one("SELECT source, recorded_at FROM sensor_readings WHERE component_id=? ORDER BY reading_id DESC", (CID,))
    assert src["source"] == "digital-twin"

    n_readings = db.query_one("SELECT COUNT(*) n FROM sensor_readings WHERE component_id=?", (CID,))["n"]
    n_records = db.query_one("SELECT COUNT(*) n FROM maintenance_records WHERE component_id=?", (CID,))["n"]
    wo = workorders.create_work_order(TAIL, CID, "Inspect & Service", degraded["risk_level"])["wo_id"]
    assert state_of(engine) == "Under inspection"
    workorders.assign_and_schedule(wo, "T-HYD", date.today())
    workorders.start_work(wo)
    assert state_of(engine) == "In maintenance"
    workorders.complete_work_order(wo, "serviced")

    after = score(engine).loc[CID]
    assert state_of(engine) == "Returned to service"
    assert after["risk_score"] < degraded["risk_score"], "completing work must lower the subsequent assessment"
    comp = db.query_one("SELECT * FROM components WHERE component_id=?", (CID,))
    assert comp["hours_since_maintenance"] == 0
    assert comp["vibration"] < 15.0 and comp["temp_deviation"] < 40.0
    # history: exactly one new post-maintenance reading, matching the component's current condition
    assert db.query_one("SELECT COUNT(*) n FROM sensor_readings WHERE component_id=?", (CID,))["n"] == n_readings + 1
    post = db.query_one("SELECT * FROM sensor_readings WHERE component_id=? AND source='post-maintenance'", (CID,))
    assert (post["vibration"], post["temp_deviation"]) == (comp["vibration"], comp["temp_deviation"])
    # maintenance record tied to the work order
    assert db.query_one("SELECT COUNT(*) n FROM maintenance_records WHERE component_id=?", (CID,))["n"] == n_records + 1
    assert db.query_one("SELECT COUNT(*) n FROM maintenance_records WHERE work_order_id=?", (wo,))["n"] == 1


def hist(n, start=datetime(2026, 6, 1), sources=None, step_h=24):
    rows = [{"recorded_at": start + timedelta(hours=step_h * i), "temp_deviation": 2 + 0.5 * i,
             "vibration": 2 + 0.2 * i, "pressure_deviation": 1 + 0.1 * i,
             "source": (sources[i] if sources else "telemetry")} for i in range(n)]
    return pd.DataFrame(rows)


def test_trend_requires_minimum_valid_samples():
    few = twin.trend_slopes(hist(config.MIN_TREND_SAMPLES - 1))
    assert few["sufficient"] is False and few["slopes"] == {} and "at least" in few["reason"]
    enough = twin.trend_slopes(hist(config.MIN_TREND_SAMPLES))
    assert enough["sufficient"] is True
    assert enough["slopes"]["vibration"] == pytest.approx(0.2, rel=1e-6)
    assert twin.trend_slopes(hist(0).reindex(columns=hist(1).columns))["sufficient"] is False


def test_trend_needs_distinct_timestamps():
    h = hist(8)
    h["recorded_at"] = h["recorded_at"].iloc[0]  # eight readings, one instant
    assert twin.trend_slopes(h)["sufficient"] is False


def test_readings_before_maintenance_reset_are_not_extrapolated():
    sources = ["telemetry"] * 8 + ["post-maintenance"] + ["digital-twin"] * 2
    h = hist(11, sources=sources)
    valid = twin.valid_history(h)
    assert len(valid) == 3 and valid["source"].iloc[0] == "post-maintenance"
    t = twin.trend_slopes(h)
    assert t["sufficient"] is False and t["n_samples"] == 3  # only 3 valid samples since the reset
