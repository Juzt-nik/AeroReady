"""Simulation-based component digital twin: lifecycle state and trend guard rails.

This is a software simulation, not a validated engineering twin: there is no physical or empirical
degradation model behind it. The state shown for a component is DERIVED from the existing work-order
workflow (``workorders.py``) and the risk score, so there is no second, conflicting state mechanism:

* ``In maintenance``      - a work order is In Progress.
* ``Under inspection``    - a work order has been raised (Open / Awaiting Parts / Scheduled) but work has
                            not started.
* ``Returned to service`` - a work order was completed within ``config.RETURNED_TO_SERVICE_DAYS``.
* ``Degraded``            - no work order, and the simulated risk category is High or Critical.
* ``Serviceable``         - none of the above.

Completing a work order (``workorders.complete_work_order``) restores the component's simulated
condition, writes a ``post-maintenance`` sensor reading and a maintenance record in one transaction,
so state, history and subsequent risk assessment change together.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from . import config, db

STATE_NOTES = {
    "Serviceable": "No open work and no High/Critical simulated risk.",
    "Degraded": "High/Critical simulated risk and no work order yet.",
    "Under inspection": "A work order has been raised; work has not started.",
    "In maintenance": "A work order is in progress.",
    "Returned to service": "A work order was completed recently; condition was restored in the simulation.",
}


def component_state(risk_level: str, active_wo_status: str | None, last_completed_at: str | None,
                    now: datetime | None = None) -> str:
    now = now or datetime.now()
    if active_wo_status == "In Progress":
        return "In maintenance"
    if active_wo_status in config.ACTIVE_WO_STATUSES:
        return "Under inspection"
    if last_completed_at:
        try:
            done = datetime.fromisoformat(last_completed_at)
        except ValueError:
            done = None
        if done and now - done <= timedelta(days=config.RETURNED_TO_SERVICE_DAYS):
            return "Returned to service"
    return "Degraded" if risk_level in ("High", "Critical") else "Serviceable"


def twin_states(scored: pd.DataFrame, now: datetime | None = None) -> dict[str, str]:
    """component_id -> twin state for every scored component."""
    active = db.query_df("SELECT component_id, status FROM work_orders WHERE status IN "
                         "('Open','Awaiting Parts','Scheduled','In Progress')")
    # a component has at most one active work order, but prefer In Progress defensively
    active_by = {r.component_id: r.status for r in active.sort_values("status", ascending=False).itertuples()}
    done = db.query_df("SELECT component_id, MAX(completed_at) AS completed_at FROM work_orders "
                       "WHERE status = 'Completed' GROUP BY component_id")
    done_by = dict(zip(done["component_id"], done["completed_at"]))
    return {r.component_id: component_state(r.risk_level, active_by.get(r.component_id),
                                             done_by.get(r.component_id), now)
            for r in scored.itertuples()}


def valid_history(hist: pd.DataFrame) -> pd.DataFrame:
    """Readings that describe the component's CURRENT build state: those since its latest post-maintenance
    reset. Earlier readings describe the pre-maintenance condition and must not be extrapolated forward."""
    if hist.empty:
        return hist
    resets = hist.loc[hist["source"] == "post-maintenance", "recorded_at"]
    return hist[hist["recorded_at"] >= resets.max()] if len(resets) else hist


def trend_slopes(hist: pd.DataFrame, window_days: int = 14, min_samples: int | None = None) -> dict:
    """Per-day linear slope of each sensor channel over the recent window, with a data-sufficiency check.

    Returns {'sufficient': bool, 'n_samples': int, 'slopes': {channel: slope}, 'reason': str}. When there
    are too few valid samples nothing is extrapolated - the caller shows "insufficient data".
    """
    need = config.MIN_TREND_SAMPLES if min_samples is None else min_samples
    h = valid_history(hist)
    if h.empty:
        return {"sufficient": False, "n_samples": 0, "slopes": {}, "reason": "no readings on record"}
    recent = h[h["recorded_at"] >= h["recorded_at"].max() - pd.Timedelta(days=window_days)]
    n = len(recent)
    t = (recent["recorded_at"] - recent["recorded_at"].min()).dt.total_seconds() / 86400
    if n < need or t.nunique() < need:
        return {"sufficient": False, "n_samples": n, "slopes": {},
                "reason": f"{n} valid reading(s) in the last {window_days} days; at least {need} distinct "
                          "timestamps are needed to estimate a trend"}
    slopes = {f: float(np.polyfit(t, recent[f], 1)[0]) for f in config.SENSOR_FEATURES}
    return {"sufficient": True, "n_samples": n, "slopes": slopes, "reason": ""}
