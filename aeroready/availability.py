"""Fleet availability: current status derivation and the scenario simulator.

Aircraft status rule (used everywhere in the app):
* "Under Maintenance" - the aircraft has a work order In Progress.
* "Unavailable"       - otherwise, it has an active work order flagged as grounding
                        (e.g. a Critical finding waiting for parts or a team).
* "Serviceable"       - neither of the above.
Fleet availability % = serviceable aircraft / total aircraft.

Simulator rule: on day d an aircraft is unavailable if d falls inside the [start, end)
window of any of its scheduled jobs, or if it has a grounding job and
down_from <= d < end (an unscheduled grounding job grounds it for the whole horizon).
"""
from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta

import numpy as np
import pandas as pd

from . import config, db
from .planner import Job, build_schedule, load_jobs
from .workorders import default_hours, default_part


# --------------------------------------------------------------------------- current status
def aircraft_status(aircraft: pd.DataFrame, work_orders: pd.DataFrame) -> pd.DataFrame:
    active = work_orders[work_orders["status"].isin(config.ACTIVE_WO_STATUSES)]
    in_prog = active[active["status"] == "In Progress"]
    grounded = active[(active["grounding"] == 1) & (active["status"] != "In Progress")]
    status, reason = [], []
    for tail in aircraft["tail_no"]:
        ip = in_prog[in_prog["tail_no"] == tail]
        gr = grounded[grounded["tail_no"] == tail]
        if len(ip):
            status.append("Under Maintenance")
            reason.append("In progress: " + ", ".join(ip["wo_id"]))
        elif len(gr):
            status.append("Unavailable")
            reason.append("Grounded pending " + ", ".join(f"{w} ({s})" for w, s in zip(gr["wo_id"], gr["status"])))
        else:
            status.append("Serviceable")
            reason.append("")
    out = aircraft.copy()
    out["status"] = status
    out["status_reason"] = reason
    open_counts = active.groupby("tail_no").size()
    out["active_work_orders"] = out["tail_no"].map(open_counts).fillna(0).astype(int)
    return out


def fleet_kpis(status_df: pd.DataFrame) -> dict:
    total = len(status_df)
    counts = status_df["status"].value_counts()
    serviceable = int(counts.get("Serviceable", 0))
    return {
        "total": total,
        "serviceable": serviceable,
        "under_maintenance": int(counts.get("Under Maintenance", 0)),
        "unavailable": int(counts.get("Unavailable", 0)),
        "availability_pct": round(100 * serviceable / total, 1) if total else 0.0,
    }


# --------------------------------------------------------------------------- simulation
def daily_availability(plan: pd.DataFrame, tails: list[str], today: date, horizon: int) -> pd.DataFrame:
    days = [today + timedelta(days=i) for i in range(horizon)]
    down = {t: np.zeros(horizon, dtype=bool) for t in tails}
    for r in plan.to_dict("records"):
        if r["tail_no"] not in down:
            continue
        mask = down[r["tail_no"]]
        start, end = r["start"], r["end"]
        scheduled = start is not None and not pd.isna(start)
        if scheduled:
            for i, d in enumerate(days):
                if start <= d < end:
                    mask[i] = True
        if r["grounding"]:
            from_day = r.get("down_from") if r.get("down_from") is not None and not pd.isna(r.get("down_from")) else today
            until = end if scheduled else days[-1] + timedelta(days=1)
            for i, d in enumerate(days):
                if from_day <= d < until:
                    mask[i] = True
    available = [sum(not down[t][i] for t in tails) for i in range(horizon)]
    return pd.DataFrame({
        "date": days,
        "available": available,
        "availability_pct": [round(100 * a / len(tails), 1) if tails else 0 for a in available],
    })


def summarise(daily: pd.DataFrame, n_aircraft: int) -> dict:
    return {
        "mean_availability_pct": round(float(daily["availability_pct"].mean()), 1),
        "min_availability_pct": float(daily["availability_pct"].min()),
        "end_availability_pct": float(daily["availability_pct"].iloc[-1]),
        "aircraft_days_lost": int(n_aircraft * len(daily) - daily["available"].sum()),
    }


def at_risk_unaddressed(scored: pd.DataFrame) -> pd.DataFrame:
    """High/Critical components with no active work order - the population BOTH the proactive and the
    reactive scenario act on, so the two strategies are compared on identical inputs."""
    active = db.query_df("SELECT component_id FROM work_orders WHERE status IN "
                         "('Open','Awaiting Parts','Scheduled','In Progress')")
    if scored.empty:
        return scored
    risky = scored[scored["risk_level"].isin(["High", "Critical"]) & ~scored["component_id"].isin(active["component_id"])]
    return risky.sort_values(["risk_score", "component_id"], ascending=[False, True])


def _unaddressed_jobs(scored: pd.DataFrame, today: date, mode: str, fail_days: dict,
                      repair_multiplier: float, stock: dict) -> list[Job]:
    """Synthetic jobs for High/Critical components that have no active work order.

    mode="proactive": planned now using the proposed action. Critical findings are ASSUMED to ground the
                      aircraft in this simulation (a scenario assumption, not a model decision).
    mode="reactive":  no action until an assumed failure ``fail_days[level]`` days out,
                      then an unplanned replacement grounds the aircraft until repaired.
    ``stock`` (part -> free qty) is consumed as parts are claimed.
    """
    risky = at_risk_unaddressed(scored)
    parts = db.table("spare_parts").set_index("part_number")
    jobs = []
    for r in risky.itertuples():
        if mode == "proactive":
            action = r.recommended_action or "Inspect & Service"
            event_day = today
            hours = default_hours(r.component_type, action)
            grounding = r.risk_level == "Critical"
        else:
            action = "Replace"
            event_day = today + timedelta(days=int(fail_days[r.risk_level]))
            hours = default_hours(r.component_type, action) * repair_multiplier
            grounding = True
        part = default_part(r.component_type, action)
        if stock.get(part, 0) >= 1:
            stock[part] -= 1
            release, why = event_day, "parts in stock"
        else:
            release = event_day + timedelta(days=int(parts.loc[part, "lead_time_days"]))
            why = f"part {part} not in stock - {int(parts.loc[part, 'lead_time_days'])}-day lead time"
        jobs.append(Job(
            wo_id=f"SIM-{r.component_id}", tail_no=r.tail_no, component_id=r.component_id,
            component_type=r.component_type, action=action, severity=r.risk_level,
            risk_score=float(r.risk_score), grounding=grounding, est_hours=hours, release=release,
            release_reason=why, created_at="9999", status="Proposed",
            down_from=event_day if grounding else None))
    return jobs


SCENARIOS = {
    "current": "Existing work orders only",
    "proactive": "Proactive: act now on all unaddressed High/Critical predictions",
    "reactive": "Reactive: wait for failures of unaddressed High/Critical components",
    "expedited": "Existing work orders + all missing parts expedited",
    "extra_team": "Existing work orders + one extra general-purpose team",
    "fifo": "Existing work orders, first-in-first-out (no risk prioritisation)",
}

# Two separate questions, never mixed in one comparison:
#   1. Does earlier intervention help?   proactive vs reactive (same fleet, horizon, work orders, at-risk set)
#   2. Does more capacity help?          existing commitments vs parts / team / ordering changes
INTERVENTION_SCENARIOS = ["reactive", "proactive"]
RESOURCE_SCENARIOS = ["current", "expedited", "extra_team", "fifo"]

ASSUMPTIONS = {
    "current": "Only the work orders that already exist, scheduled with the priority heuristic. Parts arrive on "
               "the purchase-order date, or after the lead time if nothing is on order. Unaddressed High/Critical "
               "predictions are EXCLUDED, so they cause no downtime inside the horizon. This is a baseline of "
               "existing commitments, not a 'do nothing' case - do not compare it with proactive maintenance "
               "as evidence of predictive-maintenance benefit; use proactive vs reactive for that.",
    "proactive": "Existing work orders, plus a planned work order now for every unaddressed High/Critical component, "
                 "using its proposed action. Critical findings are assumed to ground the aircraft until the work "
                 "is done (a scenario assumption, not an airworthiness decision).",
    "reactive": "Same existing work orders and the same unaddressed High/Critical components as the proactive "
                "scenario, but no preventive action. Each component is assumed to fail after the configured number "
                "of days and then needs an unplanned replacement whose labour is multiplied by the configured "
                "factor. The aircraft is grounded from failure to repair. The failure timings are user-set "
                "illustrative assumptions, not model predictions.",
    "expedited": "Same as existing work orders only, but every Awaiting Parts work order is assumed to have its "
                 "parts today. A resource-planning scenario.",
    "extra_team": "Same as existing work orders only, with one additional fictional team qualified for all systems "
                  "and actions. A resource-planning scenario.",
    "fifo": "Same work orders as the existing-work-orders scenario, ordered by creation time instead of "
            "risk/severity. A resource-planning scenario.",
}

SCENARIO_NOTICE = ("Scenario comparison under stated assumptions - not a forecast of actual fleet availability. "
                   "Failure timing in the reactive scenario is user-configured, not predicted by the model.")


def intervention_comparison(res: dict) -> dict:
    """Proactive vs reactive, on identical inputs. ``improvement`` > 0 means fewer simulated unavailable
    aircraft-days under the proactive scenario; < 0 means the proactive scenario did worse under these assumptions."""
    reactive_days = res["reactive"]["metrics"]["aircraft_days_lost"]
    proactive_days = res["proactive"]["metrics"]["aircraft_days_lost"]
    improvement = reactive_days - proactive_days
    same_inputs = res["reactive"]["inputs"] == res["proactive"]["inputs"]
    if improvement > 0:
        verdict = (f"Under these assumptions the proactive strategy loses **{improvement}** fewer aircraft-days "
                   f"({proactive_days} vs {reactive_days}).")
    elif improvement < 0:
        verdict = (f"Under these assumptions the proactive strategy performs **worse** by {-improvement} "
                   f"aircraft-days ({proactive_days} vs {reactive_days}).")
    else:
        verdict = f"Under these assumptions the two strategies lose the same number of aircraft-days ({reactive_days})."
    return {"reactive_days": reactive_days, "proactive_days": proactive_days, "improvement": improvement,
            "same_inputs": same_inputs, "at_risk_components": res["proactive"]["inputs"]["at_risk_components"],
            "text": f"{verdict} {SCENARIO_NOTICE}"}


def run_scenarios(scored: pd.DataFrame, horizon: int = 30, today: date | None = None,
                  fail_days: dict | None = None, repair_multiplier: float = 1.5,
                  keys: list[str] | None = None) -> dict:
    today = today or date.today()
    fail_days = fail_days or {"Critical": 3, "High": 10}
    teams = db.table("teams")
    tails = db.table("aircraft")["tail_no"].tolist()
    risk_lookup = dict(zip(scored["component_id"], scored["risk_score"])) if not scored.empty else {}
    base_jobs = load_jobs(today, keep_scheduled=True, risk_lookup=risk_lookup)
    parts = db.table("spare_parts")
    def free_stock():
        return dict(zip(parts["part_number"], (parts["qty_on_hand"] - parts["qty_reserved"]).clip(lower=0)))

    results = {}
    for key in keys or list(SCENARIOS):
        jobs, t, order = list(base_jobs), teams, "priority"
        if key == "proactive":
            jobs += _unaddressed_jobs(scored, today, "proactive", fail_days, repair_multiplier, free_stock())
        elif key == "reactive":
            jobs += _unaddressed_jobs(scored, today, "reactive", fail_days, repair_multiplier, free_stock())
        elif key == "expedited":
            jobs = [j if j.status != "Awaiting Parts" else replace(j, release=today,
                                                                   release_reason="parts expedited (scenario)")
                    for j in jobs]
        elif key == "extra_team":
            t = pd.concat([teams, pd.DataFrame([{
                "team_id": "T-SIM", "name": "Additional team (scenario)",
                "systems": "Propulsion,Hydraulics,Electrical", "hours_per_day": 8, "service_only": 0}])],
                ignore_index=True)
        elif key == "fifo":
            order = "fifo"
        plan = build_schedule(jobs, t, today, order=order)
        daily = daily_availability(plan, tails, today, horizon)
        results[key] = {
            "name": SCENARIOS[key], "assumptions": ASSUMPTIONS[key], "plan": plan, "daily": daily,
            "metrics": summarise(daily, len(tails)),
            # what this scenario actually consumed - lets tests and the UI verify like-for-like comparisons
            "inputs": {
                "tails": tuple(sorted(tails)), "horizon": len(daily), "today": today,
                "existing_work_orders": tuple(sorted(j.wo_id for j in jobs if not j.wo_id.startswith("SIM-"))),
                "at_risk_components": tuple(sorted(j.component_id for j in jobs if j.wo_id.startswith("SIM-"))),
            },
        }
    return results
