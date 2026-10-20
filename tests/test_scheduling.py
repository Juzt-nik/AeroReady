"""Scheduling constraints: parts-ready dates, qualified teams, no double booking."""
from datetime import date, timedelta

import pytest

from aeroready import config, db, planner, workorders
from test_planner_availability import TEAMS, TODAY, job


def test_jobs_never_start_before_parts_are_ready():
    jobs = [job(1, release=TODAY + timedelta(days=9)), job(2, release=TODAY), job(3, release=TODAY + timedelta(days=3))]
    plan = planner.build_schedule(jobs, TEAMS, TODAY).set_index("wo_id")
    for j in jobs:
        assert plan.at[j.wo_id, "start"] >= j.release


def test_teams_are_qualified_and_never_double_booked():
    jobs = [job(i, ctype="Generator", action="Replace" if i % 2 else "Inspect & Service", hours=16) for i in range(8)]
    plan = planner.build_schedule(jobs, TEAMS, TODAY)
    assert not planner.has_team_overlap(plan)
    for r in plan.itertuples():
        team = TEAMS[TEAMS["team_id"] == r.team_id].iloc[0].to_dict()
        assert workorders.team_can_do(team, "Generator", r.action)


def test_seeded_awaiting_parts_orders_wait_for_their_parts(fresh_db):
    jobs = planner.load_jobs(date.today())
    waiting = [j for j in jobs if j.status == "Awaiting Parts"]
    assert waiting, "the demo dataset has work orders awaiting parts"
    plan = planner.build_schedule(jobs, db.table("teams"), date.today()).set_index("wo_id")
    for j in waiting:
        assert j.release > date.today()
        assert plan.at[j.wo_id, "start"] >= j.release
    with pytest.raises(workorders.WorkOrderError):
        workorders.assign_and_schedule(waiting[0].wo_id, "T-HYD", date.today())


def test_planner_describes_itself_as_a_heuristic():
    assert "NOT a globally optimal" in planner.__doc__
