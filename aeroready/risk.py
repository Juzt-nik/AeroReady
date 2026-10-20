"""Fleet-level risk scoring and decision-support recommendations.

Turns component records into model features, scores them with the RiskEngine and attaches a
category, the largest differences from a healthy reference profile and a proposed action. The
score is a prototype maintenance-risk score on synthetic data, and recommendations never make
grounding or serviceability decisions.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from . import config, ingestion
from .model import RiskEngine, risk_category

# Phrased as indicators for review, never as maintenance instructions: the model only sees a few
# synthetic inputs and cannot know the cause of a deviation.
FACTOR_ADVICE = {
    "vibration": "vibration is elevated",
    "temp_deviation": "temperature deviation is elevated",
    "pressure_deviation": "pressure deviation is elevated",
    "life_used_pct": "a high share of rated life has been consumed",
    "hours_since_maintenance": "the interval since last maintenance is long",
    "prior_faults": "the fault history shows repeat faults",
    "age_years": "the component is relatively old",
}

# category -> (proposed work-order action, suggested review window in days)
# The model NEVER presets a grounding decision. Grounding is a simulated work-order decision that
# the user must set explicitly (see ui.work_order_panel); it is not part of a recommendation.
CATEGORY_POLICY = {
    "Critical": ("Replace", 1),
    "High": ("Inspect & Service", 3),
    "Medium": ("Inspect & Service", 14),
    "Low": (None, None),
}

SERVICEABILITY_NOTICE = "This model does not determine aircraft serviceability."

MIN_FACTOR_CONTRIBUTION = 0.02


def to_features(components: pd.DataFrame) -> pd.DataFrame:
    """Derive model features from component records (handles missing values)."""
    df = pd.DataFrame(index=components.index)
    rated = components["rated_life_hours"].replace(0, np.nan)
    df["life_used_pct"] = (components["operating_hours"] / rated * 100).round(1)
    for col in ["hours_since_maintenance", "age_years", "temp_deviation", "vibration",
                "pressure_deviation", "prior_faults"]:
        df[col] = pd.to_numeric(components[col], errors="coerce")
    return df[config.FEATURES]


def top_factors(contrib_row: pd.Series, k: int = 3) -> list[str]:
    s = contrib_row.sort_values(ascending=False)
    return [f for f, v in s.items() if v >= MIN_FACTOR_CONTRIBUTION][:k]


def recommend(level: str, factors: list[str], features: dict | pd.Series, telemetry: str = "Fresh") -> dict:
    """Decision-support text for one component.

    Returns the proposed work-order ``action`` and suggested review window (``due_days``) together with
    ``text``. None of it is an airworthiness or grounding decision: those rest with authorized
    maintenance personnel under approved procedures.
    """
    action, due_days = CATEGORY_POLICY[level]
    beyond_life = float(features.get("life_used_pct", 0) or 0) >= 100
    if action == "Inspect & Service" and beyond_life:
        action = "Replace"  # beyond rated life in this simulation: servicing alone is not proposed
    if level == "Low":
        text = ("Low simulated maintenance risk. No action is proposed by the model; continue routine monitoring "
                "under approved procedures.")
    else:
        lead = {
            "Critical": "Critical simulated maintenance risk. Prioritize immediate review by authorized "
                        "maintenance personnel under approved procedures.",
            "High": "High simulated maintenance risk. Prioritize review by authorized maintenance personnel "
                    "under approved procedures.",
            "Medium": "Medium simulated maintenance risk. Include in the next scheduled review by authorized "
                      "maintenance personnel.",
        }[level]
        reasons = "; ".join(FACTOR_ADVICE[f] for f in factors)
        drivers = (f" Largest differences from the healthy reference profile: {reasons} "
                   "(an association in the model, not a confirmed cause)." if reasons else
                   " The score reflects the combined effect of several moderate indicators.")
        extra = " Component is beyond its rated life in this simulation." if beyond_life else ""
        text = f"{lead}{drivers}{extra}"
    if telemetry != "Fresh":
        what = "No sensor reading is on record" if telemetry == "No data" else "The latest sensor reading is stale"
        text += (f" {what}, so this score may not reflect current condition - it is not evidence that the "
                 "component is healthy.")
    return {"action": action, "due_days": due_days, "text": f"{text} {SERVICEABILITY_NOTICE}"}


def score_components(engine: RiskEngine, components: pd.DataFrame, now: datetime | None = None) -> pd.DataFrame:
    """Return the components frame with risk columns and telemetry freshness added."""
    if components.empty:
        return components.assign(risk_prob=[], risk_score=[], risk_level=[], health_index=[],
                                 top_factors=[], recommended_action=[], recommendation=[],
                                 due_days=[], life_used_pct=[], reading_age_days=[], telemetry_status=[])
    comps = components.reset_index(drop=True)
    X = to_features(comps)
    probs = engine.predict(X)
    contrib = engine.contributions(X)
    out = comps.copy()
    out["life_used_pct"] = X["life_used_pct"]
    out["risk_prob"] = probs
    out["risk_score"] = (probs * 100).round(1)
    out["risk_level"] = [risk_category(p) for p in probs]
    out["health_index"] = (100 - out["risk_score"]).round(1)
    fresh = ingestion.telemetry_status(out["last_reading_at"], now=now)
    out["reading_age_days"] = fresh["reading_age_days"]
    out["telemetry_status"] = fresh["telemetry_status"]
    factors, actions, texts, dues = [], [], [], []
    for i in range(len(out)):
        f = top_factors(contrib.iloc[i])
        rec = recommend(out.at[i, "risk_level"], f, X.iloc[i], out.at[i, "telemetry_status"])
        factors.append(", ".join(config.FEATURE_LABELS[x] for x in f))
        actions.append(rec["action"])
        texts.append(rec["text"])
        dues.append(rec["due_days"])
    out["top_factors"] = factors
    out["recommended_action"] = actions
    out["recommendation"] = texts
    out["due_days"] = dues
    return out


def assess(engine: RiskEngine, features: dict, telemetry: str = "Fresh") -> dict:
    """Single interactive assessment used by the what-if and digital-twin screens."""
    prob = engine.predict_one(features)
    level = risk_category(prob)
    explanation = engine.explain(features)
    factors = [r.feature for r in explanation.itertuples() if r.contribution >= MIN_FACTOR_CONTRIBUTION][:3]
    rec = recommend(level, factors, features, telemetry)
    return {
        "prob": prob,
        "score": round(prob * 100, 1),
        "level": level,
        "health_index": round(100 - prob * 100, 1),
        "explanation": explanation,
        "factors": factors,
        **rec,
    }
