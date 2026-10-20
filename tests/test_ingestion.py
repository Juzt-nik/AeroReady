"""Traceable, idempotent ingestion; connector interface; stale telemetry."""
import json
from datetime import datetime, timedelta

import pandas as pd

from aeroready import config, csv_io, db, ingestion, risk

SENSOR_CSV = (b"component_id,recorded_at,temp_deviation,vibration,pressure_deviation\n"
              b"FT-201-HP1,2099-01-01 08:00:00,12.5,6.1,4.0\n"
              b"FT-201-HP1,2099-01-01 09:00:00,13.5,6.4,4.2\n")


def n_readings(source=None):
    sql, params = "SELECT COUNT(*) n FROM sensor_readings", ()
    if source:
        sql, params = sql + " WHERE source = ?", (source,)
    return db.query_one(sql, params)["n"]


# --------------------------------------------------------------------------- idempotent CSV import
def test_repeated_csv_import_does_not_duplicate_readings(fresh_db):
    before = n_readings()
    first = csv_io.import_csv("sensor_readings", SENSOR_CSV)
    assert (first["rows"], first["skipped_duplicates"], first["errors"]) == (2, 0, [])
    mid = n_readings()
    assert mid == before + 2
    second = csv_io.import_csv("sensor_readings", SENSOR_CSV)
    assert (second["rows"], second["skipped_duplicates"], second["errors"]) == (0, 2, [])
    assert n_readings() == mid
    c = db.query_one("SELECT vibration, last_reading_at FROM components WHERE component_id='FT-201-HP1'")
    assert (c["vibration"], c["last_reading_at"]) == (6.4, "2099-01-01 09:00:00")


def test_duplicate_rows_inside_one_file_are_inserted_once(fresh_db):
    data = SENSOR_CSV + b"FT-201-HP1,2099-01-01 08:00:00,12.5,6.1,4.0\n"
    res = csv_io.import_csv("sensor_readings", data)
    assert (res["rows"], res["skipped_duplicates"]) == (2, 1)


def test_older_backfill_does_not_overwrite_current_state(fresh_db):
    csv_io.import_csv("sensor_readings", SENSOR_CSV)
    old = b"component_id,recorded_at,temp_deviation,vibration,pressure_deviation\nFT-201-HP1,2098-12-31 00:00:00,1,1,1\n"
    assert csv_io.import_csv("sensor_readings", old)["rows"] == 1
    assert db.query_one("SELECT vibration FROM components WHERE component_id='FT-201-HP1'")["vibration"] == 6.4


def test_repeated_maintenance_import_is_idempotent(fresh_db):
    data = (b"component_id,performed_on,record_type,description,team_id\n"
            b"FT-201-HP1,2026-02-01,Inspection,Borescope check,T-HYD\n")
    assert csv_io.import_csv("maintenance_records", data)["rows"] == 1
    again = csv_io.import_csv("maintenance_records", data)
    assert (again["rows"], again["skipped_duplicates"]) == (0, 1)
    assert db.query_one("SELECT COUNT(*) n FROM maintenance_records WHERE description='Borescope check'")["n"] == 1


def test_source_and_record_id_are_stored(fresh_db):
    csv_io.import_csv("sensor_readings", SENSOR_CSV)
    rows = db.query_df("SELECT source, source_record_id FROM sensor_readings WHERE source = 'csv-import'")
    assert len(rows) == 2 and rows["source_record_id"].is_unique and rows["source_record_id"].notna().all()


# --------------------------------------------------------------------------- traceability
def test_source_summary_and_log_show_status_counts_and_latest(fresh_db):
    csv_io.import_csv("sensor_readings", SENSOR_CSV)
    csv_io.import_csv("sensor_readings", b"component_id,recorded_at,temp_deviation,vibration,pressure_deviation\n"
                                         b"NOPE,2099,1,1,1\n")  # rejected
    src = ingestion.source_summary().set_index("source")
    assert {"telemetry", "csv-import"} <= set(src.index)
    assert src.at["csv-import", "records"] == 2
    assert src.at["csv-import", "latest_reading"] == "2099-01-01 09:00:00"
    assert src.at["csv-import", "validation_status"] == "Rejected"  # most recent attempt
    log = db.query_df("SELECT status, rows_inserted FROM ingestion_log WHERE source='csv-import' ORDER BY log_id")
    assert log["status"].tolist() == ["Accepted", "Rejected"] and log["rows_inserted"].tolist() == [2, 0]
    assert src.at["telemetry", "records"] > 0


# --------------------------------------------------------------------------- simulated connector
def payload(message_id="M-1", comp="FT-201-HP1", ts="2099-02-01T06:30:00", vib=5.5):
    return json.dumps({"records": [{"message_id": message_id, "asset": {"component": comp}, "observed": ts,
                                    "metrics": {"temp_c_dev": 11.0, "vib_mm_s": vib, "press_pct_dev": 3.0}}]})


def test_connector_uses_same_validation_and_persistence_as_csv(fresh_db):
    c = ingestion.SimulatedJsonApiConnector(payload=payload())
    res = ingestion.ingest(c)
    assert res == {"rows": 1, "skipped_duplicates": 0, "errors": []}
    r = db.query_one("SELECT * FROM sensor_readings WHERE source = 'sim-api-v1'")
    assert r["source_record_id"] == "sim-api-v1:M-1" and r["vibration"] == 5.5
    assert db.query_one("SELECT vibration FROM components WHERE component_id='FT-201-HP1'")["vibration"] == 5.5
    again = ingestion.ingest(ingestion.SimulatedJsonApiConnector(payload=payload()))
    assert (again["rows"], again["skipped_duplicates"]) == (0, 1)
    assert n_readings("sim-api-v1") == 1


def test_connector_rejects_invalid_records_with_the_csv_rules(fresh_db):
    before = n_readings()
    bad_range = ingestion.ingest(ingestion.SimulatedJsonApiConnector(payload=payload(vib=99)))
    unknown = ingestion.ingest(ingestion.SimulatedJsonApiConnector(payload=payload(comp="NOPE-1")))
    assert any("vibration" in e for e in bad_range["errors"])
    assert any("unknown 'component_id'" in e for e in unknown["errors"])
    assert n_readings() == before
    statuses = db.query_df("SELECT status FROM ingestion_log WHERE source='sim-api-v1'")["status"].tolist()
    assert statuses == ["Rejected", "Rejected"]


def test_connector_handles_malformed_payloads(fresh_db):
    for bad in ('{"nope": 1}', "not json", json.dumps({"records": [{"asset": {}}]}), json.dumps({"records": []})):
        res = ingestion.ingest(ingestion.SimulatedJsonApiConnector(payload=bad))
        assert res["rows"] == 0 and res["errors"], bad


def test_generated_payload_is_deterministic_so_reingestion_is_idempotent(fresh_db):
    day = datetime(2099, 3, 1).date()
    first = ingestion.ingest(ingestion.SimulatedJsonApiConnector(as_of=day, max_components=4))
    second = ingestion.ingest(ingestion.SimulatedJsonApiConnector(as_of=day, max_components=4))
    assert (first["rows"], first["skipped_duplicates"]) == (4, 0)
    assert (second["rows"], second["skipped_duplicates"]) == (0, 4)


# --------------------------------------------------------------------------- stale telemetry
def test_telemetry_status_classification():
    now = datetime(2026, 6, 15, 12, 0)
    last = pd.Series([(now - timedelta(days=1)).isoformat(sep=" "), (now - timedelta(days=30)).isoformat(sep=" "),
                      None, ""], index=list("abcd"))
    out = ingestion.telemetry_status(last, now=now, threshold_days=7)
    assert out["telemetry_status"].to_dict() == {"a": "Fresh", "b": "Stale", "c": "No data", "d": "No data"}
    assert out.loc["b", "reading_age_days"] == 30.0


def test_stale_and_missing_readings_are_flagged_not_treated_as_healthy(fresh_db, engine):
    old = (datetime.now() - timedelta(days=config.STALE_READING_DAYS + 5)).isoformat(sep=" ", timespec="seconds")
    with db.connect() as conn:
        conn.execute("UPDATE components SET last_reading_at = ? WHERE component_id = 'FT-201-HP1'", (old,))
        conn.execute("UPDATE components SET last_reading_at = NULL WHERE component_id = 'FT-201-APU1'")
    sc = risk.score_components(engine, db.table("components")).set_index("component_id")
    assert sc.at["FT-201-HP1", "telemetry_status"] == "Stale"
    assert sc.at["FT-201-APU1", "telemetry_status"] == "No data"
    fresh = sc[~sc.index.isin(["FT-201-HP1", "FT-201-APU1"])]["telemetry_status"]
    assert (fresh == "Fresh").all()
    for cid in ("FT-201-HP1", "FT-201-APU1"):
        text = sc.at[cid, "recommendation"].lower()
        assert "not evidence" in text and "healthy" in text
        assert "no action required" not in text


def test_legacy_database_is_migrated_in_place(tmp_path):
    """Databases created before source_record_id/ingestion_log existed must upgrade without data loss."""
    import sqlite3
    path = tmp_path / "legacy.db"
    legacy = sqlite3.connect(path)
    legacy.executescript(db.SCHEMA.replace(",\n    source_record_id TEXT", ""))
    legacy.execute("DROP TABLE ingestion_log")
    legacy.execute("INSERT INTO aircraft VALUES ('T-1','Type','Base','Sqn',2010,100)")
    legacy.execute("INSERT INTO components VALUES ('C-1','T-1','Fuel Pump','#1','S1','2020-01-01',10,3000,10,2,0,0,1,1,1,NULL)")
    legacy.execute("INSERT INTO sensor_readings (component_id, recorded_at, operating_hours, hours_since_maintenance, "
                   "temp_deviation, vibration, pressure_deviation, source) VALUES ('C-1','2026-01-01 00:00:00',1,1,1,1,1,'telemetry')")
    legacy.commit()
    legacy.close()
    old = db.get_db_path()
    db.set_db_path(path)
    try:
        db.init_schema()
        db.init_schema()  # idempotent
        cols = {r[1] for r in sqlite3.connect(path).execute("PRAGMA table_info(sensor_readings)")}
        assert "source_record_id" in cols
        assert db.query_one("SELECT COUNT(*) n FROM sensor_readings")["n"] == 1  # data preserved
        data = b"component_id,recorded_at,temp_deviation,vibration,pressure_deviation\nC-1,2026-02-01 00:00:00,1,1,1\n"
        assert csv_io.import_csv("sensor_readings", data)["rows"] == 1
        assert csv_io.import_csv("sensor_readings", data)["skipped_duplicates"] == 1
    finally:
        db.set_db_path(old)
