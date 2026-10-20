"""Traceable, idempotent data ingestion plus telemetry-staleness checks.

Every external source - a CSV upload or a connector - goes through the same two steps:

1. **Validation** with :func:`aeroready.csv_io.validate` (types, ranges, known components).
2. **Persistence** with :func:`persist_sensor_readings` / :func:`persist_maintenance_records`,
   which are *idempotent*: each sensor reading carries a stable ``source_record_id`` and a
   record whose (component, timestamp, source) was already stored is skipped, so importing
   the same file twice never duplicates readings.

Each attempt (accepted or rejected) is written to ``ingestion_log`` so the Data Hub can show
source, record count, latest reading and validation status.

Only a SIMULATED connector ships with the prototype. It demonstrates the interface; it is not a
real aircraft-system or IoT integration. A real integration needs the source system's interface
specification, data definitions (units, timestamps, time zones), credentials and error-handling
rules, none of which can be verified by adding an API-shaped function.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime
from typing import Protocol

import numpy as np
import pandas as pd

from . import config, db

CSV_SOURCE = "csv-import"


# --------------------------------------------------------------------------- identifiers + log
def stable_record_id(source: str, component_id: str, recorded_at: str, external_id: str | None = None) -> str:
    """Deterministic id for one reading: the source's own id if it has one, else a hash of the natural key."""
    if external_id:
        return f"{source}:{external_id}"
    digest = hashlib.sha1(f"{source}|{component_id}|{recorded_at}".encode()).hexdigest()[:20]
    return f"{source}:{digest}"


def log_ingestion(source: str, kind: str, received: int, inserted: int, skipped: int, status: str,
                  message: str = "") -> None:
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO ingestion_log (ts, source, kind, rows_received, rows_inserted, rows_skipped, status, message) "
            "VALUES (?,?,?,?,?,?,?,?)", (db.now_iso(), source, kind, received, inserted, skipped, status, message))


# --------------------------------------------------------------------------- idempotent persistence
def persist_sensor_readings(clean: pd.DataFrame, source: str, external_ids: dict | None = None) -> dict:
    """Insert validated readings, skipping any already stored. Returns {'inserted', 'skipped_duplicates'}.

    ``clean`` is the output of ``csv_io.validate('sensor_readings', ...)``. ``external_ids`` optionally maps
    (component_id, recorded_at) -> the source system's own record id.
    """
    external_ids = external_ids or {}
    inserted = skipped = 0
    with db.connect() as conn:
        comps = {r["component_id"]: dict(r) for r in conn.execute("SELECT * FROM components")}
        existing_ids = {r[0] for r in conn.execute(
            "SELECT source_record_id FROM sensor_readings WHERE source_record_id IS NOT NULL")}
        existing_keys = {(r[0], r[1]) for r in conn.execute(
            "SELECT component_id, recorded_at FROM sensor_readings WHERE source = ?", (source,))}
        for r in clean.sort_values("recorded_at").to_dict("records"):
            cid, ts = r["component_id"], r["recorded_at"]
            rid = stable_record_id(source, cid, ts, external_ids.get((cid, ts)))
            if rid in existing_ids or (cid, ts) in existing_keys:
                skipped += 1
                continue
            c = comps[cid]
            op = r["operating_hours"] if pd.notna(r["operating_hours"]) else c["operating_hours"]
            hsm = (r["hours_since_maintenance"] if pd.notna(r["hours_since_maintenance"])
                   else c["hours_since_maintenance"])
            conn.execute(
                "INSERT INTO sensor_readings (component_id, recorded_at, operating_hours, hours_since_maintenance, "
                "temp_deviation, vibration, pressure_deviation, source, source_record_id) VALUES (?,?,?,?,?,?,?,?,?)",
                (cid, ts, op, hsm, r["temp_deviation"], r["vibration"], r["pressure_deviation"], source, rid))
            existing_ids.add(rid)
            existing_keys.add((cid, ts))
            inserted += 1
            # the newest reading becomes the component's current state; older back-fill does not
            if ts >= (c["last_reading_at"] or ""):
                conn.execute(
                    "UPDATE components SET temp_deviation=?, vibration=?, pressure_deviation=?, operating_hours=?, "
                    "hours_since_maintenance=?, last_reading_at=? WHERE component_id=?",
                    (r["temp_deviation"], r["vibration"], r["pressure_deviation"], op, hsm, ts, cid))
                c.update(last_reading_at=ts, operating_hours=op, hours_since_maintenance=hsm)
        db.log_activity(conn, "Data import",
                        f"{source}: {inserted} new sensor reading(s), {skipped} duplicate(s) skipped")
    log_ingestion(source, "sensor_readings", len(clean), inserted, skipped, "Accepted",
                  f"{inserted} inserted, {skipped} duplicate(s) skipped")
    return {"inserted": inserted, "skipped_duplicates": skipped}


def persist_maintenance_records(clean: pd.DataFrame, source: str = CSV_SOURCE) -> dict:
    """Insert validated maintenance records, skipping exact repeats of (component, date, type, description)."""
    inserted = skipped = 0
    with db.connect() as conn:
        tails = dict(conn.execute("SELECT component_id, tail_no FROM components").fetchall())
        seen = {tuple(r) for r in conn.execute(
            "SELECT component_id, performed_on, record_type, description FROM maintenance_records")}
        for r in clean.to_dict("records"):
            key = (r["component_id"], r["performed_on"], r["record_type"], r["description"])
            if key in seen:
                skipped += 1
                continue
            seen.add(key)
            conn.execute(
                "INSERT INTO maintenance_records (component_id, tail_no, performed_on, record_type, description, "
                "hours_at, team_id, work_order_id) VALUES (?,?,?,?,?,?,?,NULL)",
                (r["component_id"], tails[r["component_id"]], r["performed_on"], r["record_type"],
                 r["description"], None if pd.isna(r["hours_at"]) else float(r["hours_at"]),
                 None if pd.isna(r["team_id"]) else r["team_id"]))
            inserted += 1
        db.log_activity(conn, "Data import",
                        f"{source}: {inserted} new maintenance record(s), {skipped} duplicate(s) skipped")
    log_ingestion(source, "maintenance_records", len(clean), inserted, skipped, "Accepted",
                  f"{inserted} inserted, {skipped} duplicate(s) skipped")
    return {"inserted": inserted, "skipped_duplicates": skipped}


# --------------------------------------------------------------------------- connector interface
class Connector(Protocol):
    """A source of sensor readings. ``fetch`` returns raw records in the source's own shape and
    ``normalize`` maps one of them onto the AeroReady sensor-reading schema."""

    source: str

    def fetch(self) -> list[dict]: ...

    def normalize(self, raw: dict) -> dict: ...


def ingest(connector: Connector) -> dict:
    """Pull from ``connector`` and run the records through the SAME validation and persistence as CSV imports.

    Returns {'rows', 'skipped_duplicates', 'errors'}; nothing is written when validation fails.
    """
    from . import csv_io  # local import: csv_io imports this module

    try:
        raw = connector.fetch()
        normalized = [connector.normalize(r) for r in raw]
    except (ValueError, KeyError, TypeError) as exc:
        log_ingestion(connector.source, "sensor_readings", 0, 0, 0, "Rejected", f"Malformed payload: {exc}")
        return {"rows": 0, "skipped_duplicates": 0, "errors": [f"Malformed payload: {exc}"]}
    if not normalized:
        log_ingestion(connector.source, "sensor_readings", 0, 0, 0, "Rejected", "Source returned no records")
        return {"rows": 0, "skipped_duplicates": 0, "errors": ["The source returned no records."]}

    ext = {(n["component_id"], n["recorded_at"]): n.pop("_external_id", None) for n in normalized}
    df = pd.DataFrame(normalized).astype(str).replace({"None": "", "nan": ""})
    clean, errors = csv_io.validate("sensor_readings", df)
    if errors:
        log_ingestion(connector.source, "sensor_readings", len(df), 0, 0, "Rejected",
                      f"{len(errors)} validation error(s); first: {errors[0]}")
        return {"rows": 0, "skipped_duplicates": 0, "errors": errors}
    # external ids are keyed on the validated (normalised) timestamp
    ext = {(c, pd.to_datetime(t).strftime("%Y-%m-%d %H:%M:%S")): v for (c, t), v in ext.items()}
    res = persist_sensor_readings(clean, connector.source, ext)
    return {"rows": res["inserted"], "skipped_duplicates": res["skipped_duplicates"], "errors": []}


@dataclass
class SimulatedJsonApiConnector:
    """SIMULATED JSON/API telemetry source (no network, no real aircraft system).

    Its payload uses a different shape and field names from the AeroReady schema, so ``normalize``
    does real mapping work. ``payload`` may be supplied directly (e.g. in tests); otherwise a
    deterministic payload is generated for ``as_of`` - fetching twice for the same day yields the
    same record ids, which is what makes re-ingestion idempotent.
    """

    source: str = "sim-api-v1"
    payload: str | None = None
    as_of: date | None = None
    component_ids: tuple[str, ...] | None = None
    max_components: int = 5
    seed: int = config.RANDOM_SEED

    def _generate(self) -> dict:
        as_of = self.as_of or date.today()
        comps = db.query_df("SELECT component_id, temp_deviation, vibration, pressure_deviation FROM components "
                            "ORDER BY component_id")
        if self.component_ids is not None:
            comps = comps[comps["component_id"].isin(self.component_ids)]
        records = []
        for r in comps.head(self.max_components).to_dict("records"):
            digest = hashlib.sha1(f"{self.seed}|{r['component_id']}|{as_of}".encode()).digest()
            rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
            ts = datetime(as_of.year, as_of.month, as_of.day, 6, 30)
            records.append({
                "message_id": f"SIM-{as_of:%Y%m%d}-{r['component_id']}",
                "asset": {"component": r["component_id"]},
                "observed": ts.isoformat(),
                "metrics": {
                    "temp_c_dev": round(float(np.clip(r["temp_deviation"] + rng.normal(0, 0.8), -10, 60)), 1),
                    "vib_mm_s": round(float(np.clip(r["vibration"] + rng.normal(0, 0.25), 0, 20)), 2),
                    "press_pct_dev": round(float(np.clip(r["pressure_deviation"] + rng.normal(0, 0.5), 0, 40)), 1),
                },
            })
        return {"schema": "sim-telemetry/1", "records": records}

    def fetch(self) -> list[dict]:
        data = json.loads(self.payload) if self.payload is not None else self._generate()
        if not isinstance(data, dict) or not isinstance(data.get("records"), list):
            raise ValueError("expected a JSON object with a 'records' list")
        return data["records"]

    def normalize(self, raw: dict) -> dict:
        m = raw["metrics"]
        return {
            "component_id": str(raw["asset"]["component"]).strip(),
            "recorded_at": str(raw["observed"]).strip(),
            "temp_deviation": m["temp_c_dev"],
            "vibration": m["vib_mm_s"],
            "pressure_deviation": m["press_pct_dev"],
            "_external_id": raw.get("message_id"),
        }


# --------------------------------------------------------------------------- traceability summary
def source_summary() -> pd.DataFrame:
    """One row per sensor-data source: record count, latest reading and last import status."""
    readings = db.query_df(
        "SELECT source, COUNT(*) AS records, COUNT(DISTINCT component_id) AS components, "
        "MAX(recorded_at) AS latest_reading FROM sensor_readings GROUP BY source ORDER BY source")
    log = db.query_df("SELECT source, ts, status, message FROM ingestion_log "
                      "WHERE kind = 'sensor_readings' ORDER BY log_id")
    if log.empty:
        last = pd.DataFrame(columns=["source", "last_import_at", "validation_status", "last_import_message"])
    else:
        last = (log.groupby("source").tail(1)
                .rename(columns={"ts": "last_import_at", "status": "validation_status", "message": "last_import_message"}))
    out = readings.merge(last, on="source", how="outer")
    out["records"] = out["records"].fillna(0).astype(int)
    out["components"] = out["components"].fillna(0).astype(int)
    out["validation_status"] = out["validation_status"].fillna("Loaded with dataset (no import log)")
    return out


# --------------------------------------------------------------------------- staleness
def telemetry_status(last_reading_at: pd.Series, now: datetime | None = None,
                     threshold_days: float | None = None) -> pd.DataFrame:
    """Age and status of each component's newest sensor reading.

    status is "No data" (never reported), "Stale" (older than the threshold) or "Fresh". Missing or stale
    telemetry must never be read as a healthy component - it means the condition is not currently known.
    """
    now = now or datetime.now()
    threshold = config.STALE_READING_DAYS if threshold_days is None else threshold_days
    ts = pd.to_datetime(last_reading_at, errors="coerce")
    age = (pd.Timestamp(now) - ts).dt.total_seconds() / 86400
    status = np.where(ts.isna(), "No data", np.where(age > threshold, "Stale", "Fresh"))
    return pd.DataFrame({"reading_age_days": age.round(1), "telemetry_status": status}, index=last_reading_at.index)
