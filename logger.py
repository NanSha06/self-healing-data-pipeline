"""
logger.py
---------
Dedicated audit logging module for the Self-Healing Data Pipeline.

Provides a single AuditLogger class that writes a detailed, queryable
record of every pipeline event to SQLite — separate from the console/file
logs produced by loguru.

Three tables are maintained:
  pipeline_runs   — one row per end-to-end run (status, counts, duration)
  repair_attempts — one row per LLM heal attempt (diagnosis, fix_code, outcome)
  issue_events    — one row per individual validation/anomaly issue detected

Usage:
    from logger import AuditLogger
    from pipeline import PipelineResult
    from healer   import HealResult

    audit = AuditLogger("pipeline_audit.db")

    # After a pipeline run completes
    audit.log_run(pipeline_result)

    # After a heal cycle completes
    audit.log_heal(heal_result, run_id="abc123")

    # After validation/anomaly detection
    audit.log_issues(validation_report, anomaly_report, run_id="abc123")

    # Query helpers
    df = audit.get_runs(limit=20)
    df = audit.get_repairs(run_id="abc123")
    df = audit.get_issues(run_id="abc123")
    stats = audit.get_stats()
"""

import os
import json
import sqlite3
import pandas as pd
from datetime import datetime
from loguru import logger
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from pipeline  import PipelineResult
    from healer    import HealResult
    from validator import ValidationReport
    from anomaly   import AnomalyReport


# ─────────────────────────────────────────────
#  Schema definitions
# ─────────────────────────────────────────────

_CREATE_PIPELINE_RUNS = """
CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id              TEXT    PRIMARY KEY,
    input_path          TEXT,
    output_path         TEXT,
    status              TEXT,
    total_rows          INTEGER,
    rows_written        INTEGER,
    validation_issues   INTEGER,
    anomalies_detected  INTEGER,
    issues_healed       INTEGER,
    issues_remaining    INTEGER,
    heal_attempts       INTEGER,
    duration_seconds    REAL,
    error_message       TEXT,
    timestamp           TEXT
)
"""

_CREATE_REPAIR_ATTEMPTS = """
CREATE TABLE IF NOT EXISTS repair_attempts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT,
    batch_id        TEXT,
    attempt_number  INTEGER,
    diagnosis       TEXT,
    fix_code        TEXT,
    confidence      REAL,
    outcome         TEXT,
    error_message   TEXT,
    issues_before   INTEGER,
    issues_after    INTEGER,
    timestamp       TEXT,
    FOREIGN KEY (run_id) REFERENCES pipeline_runs(run_id)
)
"""

_CREATE_ISSUE_EVENTS = """
CREATE TABLE IF NOT EXISTS issue_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT,
    source          TEXT,     -- 'validation' or 'anomaly'
    column_name     TEXT,
    issue_type      TEXT,
    severity        TEXT,     -- null for validation issues
    description     TEXT,
    affected_count  INTEGER,
    affected_rows   TEXT,     -- JSON array
    sample_values   TEXT,     -- JSON array
    timestamp       TEXT,
    FOREIGN KEY (run_id) REFERENCES pipeline_runs(run_id)
)
"""


# ─────────────────────────────────────────────
#  AuditLogger
# ─────────────────────────────────────────────

class AuditLogger:
    """
    Writes structured audit records to a SQLite database.

    Maintains three tables:
      - pipeline_runs     : high-level run summary
      - repair_attempts   : per-attempt LLM repair details
      - issue_events      : per-issue validation and anomaly records
    """

    def __init__(self, db_path: str = "pipeline_audit.db"):
        self.db_path = db_path
        self._init_db()
        logger.debug(f"AuditLogger initialised — db={db_path}")

    # ── Schema init ─────────────────────────────

    def _init_db(self) -> None:
        """Create all tables if they don't already exist."""
        with self._connect() as conn:
            conn.execute(_CREATE_PIPELINE_RUNS)
            conn.execute(_CREATE_REPAIR_ATTEMPTS)
            conn.execute(_CREATE_ISSUE_EVENTS)
        logger.debug("Audit DB schema initialised.")

    # ── Write methods ───────────────────────────

    def log_run(self, result: "PipelineResult") -> None:
        """
        Write or update a pipeline run record.
        Safe to call multiple times — uses INSERT OR REPLACE.
        """
        try:
            d = result.to_dict()
            with self._connect() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO pipeline_runs (
                        run_id, input_path, output_path, status,
                        total_rows, rows_written, validation_issues,
                        anomalies_detected, issues_healed, issues_remaining,
                        heal_attempts, duration_seconds, error_message, timestamp
                    ) VALUES (
                        :run_id, :input_path, :output_path, :status,
                        :total_rows, :rows_written, :validation_issues,
                        :anomalies_detected, :issues_healed, :issues_remaining,
                        :heal_attempts, :duration_seconds, :error_message, :timestamp
                    )
                """, d)
            logger.debug(f"Run logged — run_id={result.run_id} status={result.status}")
        except Exception as e:
            logger.warning(f"AuditLogger.log_run failed: {e}")

    def log_heal(self, heal_result: "HealResult", run_id: str) -> None:
        """
        Write one repair_attempts row per HealAttempt inside a HealResult.
        """
        try:
            ts = datetime.now().isoformat()
            rows = []
            for attempt in heal_result.attempts:
                rows.append((
                    run_id,
                    heal_result.batch_id,
                    attempt.attempt_number,
                    attempt.diagnosis,
                    attempt.fix_code,
                    attempt.confidence,
                    attempt.outcome,
                    attempt.error_message,
                    attempt.issues_before,
                    attempt.issues_after,
                    ts,
                ))

            with self._connect() as conn:
                conn.executemany("""
                    INSERT INTO repair_attempts (
                        run_id, batch_id, attempt_number, diagnosis, fix_code,
                        confidence, outcome, error_message,
                        issues_before, issues_after, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, rows)

            logger.debug(
                f"Repair attempts logged — run_id={run_id} "
                f"batch={heal_result.batch_id} count={len(rows)}"
            )
        except Exception as e:
            logger.warning(f"AuditLogger.log_heal failed: {e}")

    def log_issues(
        self,
        run_id: str,
        validation_report: Optional["ValidationReport"] = None,
        anomaly_report: Optional["AnomalyReport"] = None,
    ) -> None:
        """
        Write one issue_events row per ValidationIssue and Anomaly.
        """
        try:
            ts   = datetime.now().isoformat()
            rows = []

            if validation_report:
                for issue in validation_report.issues:
                    rows.append((
                        run_id,
                        "validation",
                        issue.column,
                        issue.issue_type,
                        None,                               # no severity for validation issues
                        issue.description,
                        issue.affected_count,
                        json.dumps(issue.affected_rows[:20]),
                        json.dumps([str(v) for v in issue.sample_values[:5]]),
                        ts,
                    ))

            if anomaly_report:
                for anomaly in anomaly_report.anomalies:
                    rows.append((
                        run_id,
                        "anomaly",
                        anomaly.column,
                        anomaly.anomaly_type,
                        anomaly.severity,
                        anomaly.description,
                        anomaly.affected_count,
                        json.dumps(anomaly.affected_rows[:20]),
                        json.dumps([str(v) for v in anomaly.sample_values[:5]]),
                        ts,
                    ))

            if rows:
                with self._connect() as conn:
                    conn.executemany("""
                        INSERT INTO issue_events (
                            run_id, source, column_name, issue_type, severity,
                            description, affected_count, affected_rows,
                            sample_values, timestamp
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, rows)

            logger.debug(
                f"Issues logged — run_id={run_id} count={len(rows)}"
            )
        except Exception as e:
            logger.warning(f"AuditLogger.log_issues failed: {e}")

    # ── Query methods ───────────────────────────

    def get_runs(self, limit: int = 20, status: Optional[str] = None) -> pd.DataFrame:
        """
        Return recent pipeline runs.

        Args:
            limit  : max rows to return
            status : optional filter e.g. 'success', 'failed', 'quarantined'
        """
        where = f"WHERE status = '{status}'" if status else ""
        sql   = f"""
            SELECT run_id, timestamp, status, total_rows, validation_issues,
                   anomalies_detected, issues_healed, issues_remaining,
                   heal_attempts, duration_seconds, input_path
            FROM pipeline_runs
            {where}
            ORDER BY timestamp DESC
            LIMIT {limit}
        """
        with self._connect() as conn:
            return pd.read_sql(sql, conn)

    def get_repairs(self, run_id: Optional[str] = None, limit: int = 50) -> pd.DataFrame:
        """
        Return repair attempt records.

        Args:
            run_id : filter to a specific run (optional)
            limit  : max rows to return
        """
        where = f"WHERE run_id = '{run_id}'" if run_id else ""
        sql   = f"""
            SELECT run_id, batch_id, attempt_number, outcome, confidence,
                   diagnosis, issues_before, issues_after, error_message, timestamp
            FROM repair_attempts
            {where}
            ORDER BY timestamp DESC
            LIMIT {limit}
        """
        with self._connect() as conn:
            return pd.read_sql(sql, conn)

    def get_issues(self, run_id: Optional[str] = None, source: Optional[str] = None,
                   limit: int = 100) -> pd.DataFrame:
        """
        Return issue event records.

        Args:
            run_id : filter to a specific run (optional)
            source : 'validation' or 'anomaly' (optional)
            limit  : max rows to return
        """
        conditions = []
        if run_id:
            conditions.append(f"run_id = '{run_id}'")
        if source:
            conditions.append(f"source = '{source}'")
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

        sql = f"""
            SELECT run_id, source, column_name, issue_type, severity,
                   description, affected_count, timestamp
            FROM issue_events
            {where}
            ORDER BY timestamp DESC
            LIMIT {limit}
        """
        with self._connect() as conn:
            return pd.read_sql(sql, conn)

    def get_stats(self) -> dict:
        """
        Return aggregate statistics across all pipeline runs.

        Returns a dict with:
          - total_runs, runs_by_status
          - avg_duration_seconds
          - total_issues_detected, total_issues_healed
          - most_common_issue_types
          - heal_success_rate
        """
        with self._connect() as conn:
            # Overall run stats
            runs_df = pd.read_sql(
                "SELECT status, duration_seconds, validation_issues, "
                "anomalies_detected, issues_healed, heal_attempts "
                "FROM pipeline_runs",
                conn,
            )

            # Issue type frequency
            issues_df = pd.read_sql(
                "SELECT issue_type, COUNT(*) as count "
                "FROM issue_events GROUP BY issue_type ORDER BY count DESC LIMIT 10",
                conn,
            )

            # Repair outcome distribution
            repairs_df = pd.read_sql(
                "SELECT outcome, COUNT(*) as count "
                "FROM repair_attempts GROUP BY outcome",
                conn,
            )

        if runs_df.empty:
            return {"message": "No pipeline runs recorded yet."}

        total_runs     = len(runs_df)
        runs_by_status = runs_df["status"].value_counts().to_dict()
        avg_duration   = round(runs_df["duration_seconds"].mean(), 2)
        total_issues   = int(
            runs_df["validation_issues"].sum() + runs_df["anomalies_detected"].sum()
        )
        total_healed   = int(runs_df["issues_healed"].sum())

        heal_rate = 0.0
        if not repairs_df.empty:
            repair_counts  = repairs_df.set_index("outcome")["count"].to_dict()
            total_attempts = sum(repair_counts.values())
            successes      = repair_counts.get("success", 0)
            heal_rate      = round(successes / total_attempts * 100, 1) if total_attempts else 0.0

        return {
            "total_runs":             total_runs,
            "runs_by_status":         runs_by_status,
            "avg_duration_seconds":   avg_duration,
            "total_issues_detected":  total_issues,
            "total_issues_healed":    total_healed,
            "heal_success_rate_pct":  heal_rate,
            "most_common_issue_types": issues_df.to_dict("records"),
            "repair_outcomes":         repairs_df.to_dict("records") if not repairs_df.empty else [],
        }

    def print_stats(self) -> None:
        """Pretty-print aggregate stats to the console."""
        stats = self.get_stats()

        if "message" in stats:
            print(stats["message"])
            return

        print("\n" + "═" * 52)
        print("  📊  PIPELINE AUDIT STATISTICS")
        print("═" * 52)
        print(f"  Total runs            : {stats['total_runs']}")
        for status, count in stats["runs_by_status"].items():
            icon = {"success": "✅", "partial": "⚠️ ", "failed": "❌", "quarantined": "🚫"}.get(status, "  ")
            print(f"    {icon}  {status:<14}: {count}")
        print(f"  Avg duration          : {stats['avg_duration_seconds']}s")
        print(f"  Total issues detected : {stats['total_issues_detected']}")
        print(f"  Total issues healed   : {stats['total_issues_healed']}")
        print(f"  Heal success rate     : {stats['heal_success_rate_pct']}%")

        if stats["most_common_issue_types"]:
            print("\n  Most common issue types:")
            for row in stats["most_common_issue_types"]:
                print(f"    • {row['issue_type']:<22} {row['count']} occurrence(s)")

        if stats["repair_outcomes"]:
            print("\n  Repair attempt outcomes:")
            for row in stats["repair_outcomes"]:
                print(f"    • {row['outcome']:<22} {row['count']} attempt(s)")

        print("═" * 52 + "\n")

    def export_json(self, output_path: str = "audit_export.json") -> None:
        """Export the full audit log to a single JSON file."""
        try:
            runs    = self.get_runs(limit=10000)
            repairs = self.get_repairs(limit=10000)
            issues  = self.get_issues(limit=10000)

            export = {
                "exported_at":    datetime.now().isoformat(),
                "pipeline_runs":  runs.to_dict("records"),
                "repair_attempts": repairs.to_dict("records"),
                "issue_events":   issues.to_dict("records"),
            }

            with open(output_path, "w") as f:
                json.dump(export, f, indent=2, default=str)

            logger.info(f"Audit log exported → {output_path}")

        except Exception as e:
            logger.error(f"Export failed: {e}")

    # ── Internal ─────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn


# ─────────────────────────────────────────────
#  Quick test — run directly to see output
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile

    print("Testing AuditLogger with a temporary database...\n")

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        tmp_db = f.name

    audit = AuditLogger(tmp_db)

    # ── Simulate a pipeline run result ──────────
    from dataclasses import dataclass, field as dc_field

    @dataclass
    class MockRun:
        run_id:             str   = "test001"
        input_path:         str   = "data/sample.csv"
        output_path:        str   = "output/test001_clean.csv"
        status:             str   = "partial"
        total_rows:         int   = 20
        rows_written:       int   = 20
        validation_issues:  int   = 8
        anomalies_detected: int   = 3
        issues_healed:      int   = 7
        issues_remaining:   int   = 4
        heal_attempts:      int   = 2
        duration_seconds:   float = 12.4
        error_message:      str   = ""
        timestamp:          str   = dc_field(default_factory=lambda: datetime.now().isoformat())

        def to_dict(self):
            return self.__dict__

    @dataclass
    class MockAttempt:
        attempt_number: int   = 1
        diagnosis:      str   = "Null values in age column."
        fix_code:       str   = "def apply_fix(df):\n    df=df.copy()\n    df['age'].fillna(df['age'].median(), inplace=True)\n    return df"
        confidence:     float = 0.92
        outcome:        str   = "success"
        error_message:  str   = ""
        issues_before:  int   = 8
        issues_after:   int   = 4

        def to_dict(self):
            return self.__dict__

    @dataclass
    class MockHealResult:
        batch_id:            str  = "test001"
        success:             bool = True
        original_issue_count:int  = 8
        final_issue_count:   int  = 4
        quarantined:         bool = False
        timestamp:           str  = dc_field(default_factory=lambda: datetime.now().isoformat())
        attempts:            list = dc_field(default_factory=lambda: [MockAttempt()])

    mock_run   = MockRun()
    mock_heal  = MockHealResult()

    # Log run and heal
    audit.log_run(mock_run)
    audit.log_heal(mock_heal, run_id="test001")

    # Print history
    print("── Recent runs ─────────────────────────────")
    print(audit.get_runs().to_string(index=False))

    print("\n── Repair attempts ─────────────────────────")
    print(audit.get_repairs().to_string(index=False))

    # Print stats
    audit.print_stats()

    # Export
    export_path = "audit_test_export.json"
    audit.export_json(export_path)
    print(f"Exported to {export_path}")

    # Release SQLite connections before deleting temp file (required on Windows)
    del audit
    import gc
    gc.collect()
    try:
        os.unlink(tmp_db)
    except PermissionError:
        pass  # Windows may hold the handle briefly — safe to ignore in tests