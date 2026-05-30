"""
pipeline.py
-----------
Main orchestrator for the Self-Healing Data Pipeline.

Ties together ingestion, validation, anomaly detection, healing, and output
into a single end-to-end run. Every stage is logged and timed. Results are
written to the output directory and the audit log.

Usage (command line):
    python pipeline.py --input data/sample.csv
    python pipeline.py --input data/sample.csv --output output/clean.csv
    python pipeline.py --input data/sample.csv --config config.yaml --no-heal

Usage (as a module):
    from pipeline import Pipeline
    pipeline = Pipeline("config.yaml")
    result   = pipeline.run("data/sample.csv")
"""

import os
import sys
import json
import uuid
import argparse
import time
import sqlite3
import pandas as pd
import yaml
from datetime import datetime
from dataclasses import dataclass, field
from loguru import logger
from dotenv import load_dotenv
from typing import Optional

from validator import Validator
from anomaly   import AnomalyDetector
from healer    import Healer

load_dotenv()


# ─────────────────────────────────────────────
#  Configure logger
# ─────────────────────────────────────────────

def _setup_logger(log_dir: str = "logs") -> None:
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"pipeline_{datetime.now().strftime('%Y%m%d')}.log")
    logger.remove()
    logger.add(sys.stderr, level="INFO",
               format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")
    logger.add(log_file,   level="DEBUG",
               format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
               rotation="10 MB", retention="14 days")


# ─────────────────────────────────────────────
#  Result dataclass
# ─────────────────────────────────────────────

@dataclass
class PipelineResult:
    """End-to-end result of a single pipeline run."""
    run_id:               str
    input_path:           str
    output_path:          str  = ""
    status:               str  = "pending"   # success | partial | failed | quarantined
    total_rows:           int  = 0
    rows_written:         int  = 0
    validation_issues:    int  = 0
    anomalies_detected:   int  = 0
    issues_healed:        int  = 0
    issues_remaining:     int  = 0
    heal_attempts:        int  = 0
    duration_seconds:     float = 0.0
    error_message:        str  = ""
    timestamp:            str  = field(default_factory=lambda: datetime.now().isoformat())

    def print_summary(self) -> None:
        icon = {"success": "✅", "partial": "⚠️ ", "failed": "❌", "quarantined": "🚫"}.get(
            self.status, "❓"
        )
        print("\n" + "═" * 62)
        print(f"  {icon}  PIPELINE RUN COMPLETE")
        print("═" * 62)
        print(f"  Run ID          : {self.run_id}")
        print(f"  Status          : {self.status.upper()}")
        print(f"  Input           : {self.input_path}")
        print(f"  Output          : {self.output_path or 'N/A'}")
        print(f"  Total rows      : {self.total_rows}")
        print(f"  Rows written    : {self.rows_written}")
        print(f"  Validation issues : {self.validation_issues}")
        print(f"  Anomalies found : {self.anomalies_detected}")
        print(f"  Issues healed   : {self.issues_healed}")
        print(f"  Issues remaining: {self.issues_remaining}")
        print(f"  Heal attempts   : {self.heal_attempts}")
        print(f"  Duration        : {self.duration_seconds:.2f}s")
        if self.error_message:
            print(f"  Error           : {self.error_message}")
        print("═" * 62 + "\n")

    def to_dict(self) -> dict:
        return {
            "run_id":             self.run_id,
            "input_path":         self.input_path,
            "output_path":        self.output_path,
            "status":             self.status,
            "total_rows":         self.total_rows,
            "rows_written":       self.rows_written,
            "validation_issues":  self.validation_issues,
            "anomalies_detected": self.anomalies_detected,
            "issues_healed":      self.issues_healed,
            "issues_remaining":   self.issues_remaining,
            "heal_attempts":      self.heal_attempts,
            "duration_seconds":   round(self.duration_seconds, 3),
            "error_message":      self.error_message,
            "timestamp":          self.timestamp,
        }


# ─────────────────────────────────────────────
#  Pipeline
# ─────────────────────────────────────────────

class Pipeline:
    """
    End-to-end self-healing data pipeline.

    Stages:
        1. Ingest     — load CSV (or DataFrame) into pandas
        2. Validate   — run schema/rule checks (validator.py)
        3. Detect     — run statistical anomaly checks (anomaly.py)
        4. Heal       — call LLM to fix issues (healer.py)  [optional]
        5. Output     — write clean data to CSV
        6. Audit      — persist run metadata to SQLite
    """

    AUDIT_DB = "pipeline_audit.db"

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path

        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.pipeline_cfg  = self.config.get("pipeline", {})
        self.output_dir    = self.pipeline_cfg.get("output_dir", "output")
        os.makedirs(self.output_dir, exist_ok=True)

        self.validator = Validator(config_path)
        self.detector  = AnomalyDetector(config_path)
        self.healer    = Healer(config_path)

        self._init_audit_db()
        logger.info(f"Pipeline initialised. config={config_path}")

    # ── Public entry point ──────────────────────

    def run(
        self,
        input_source: str | pd.DataFrame,
        output_path: Optional[str] = None,
        heal: bool = True,
        run_id: Optional[str] = None,
    ) -> PipelineResult:
        """
        Execute a full pipeline run.

        Args:
            input_source : path to a CSV file, or a pandas DataFrame directly.
            output_path  : where to write the clean output CSV.
                           Defaults to output/<run_id>_clean.csv
            heal         : if False, skip the healing stage (validate + detect only).
            run_id       : optional custom run identifier.

        Returns:
            PipelineResult with full run metadata.
        """
        run_id  = run_id or str(uuid.uuid4())[:8]
        result  = PipelineResult(
            run_id=run_id,
            input_path=str(input_source) if isinstance(input_source, str) else "<DataFrame>",
        )
        start_time = time.time()

        logger.info(f"{'─'*50}")
        logger.info(f"Pipeline run started  run_id={run_id}")
        logger.info(f"{'─'*50}")

        try:
            # ── Stage 1: Ingest ──────────────────
            df = self._stage_ingest(input_source, result)
            if df is None:
                return self._finalise(result, start_time)

            # ── Stage 2: Validate ────────────────
            val_report = self._stage_validate(df, result)

            # ── Stage 3: Detect anomalies ────────
            anom_report = self._stage_detect(df, result)

            total_issues = result.validation_issues + result.anomalies_detected

            # ── Stage 4: Heal ────────────────────
            if heal and total_issues > 0:
                df = self._stage_heal(df, val_report, anom_report, result, run_id)
            elif total_issues == 0:
                logger.success("Data is clean — skipping healing stage.")
                result.status = "success"
            else:
                logger.info("Healing disabled — skipping healing stage.")
                result.status  = "partial"
                result.issues_remaining = total_issues

            # ── Stage 5: Output ──────────────────
            self._stage_output(df, result, output_path, run_id)

        except Exception as e:
            result.status        = "failed"
            result.error_message = str(e)
            logger.exception(f"Pipeline run failed: {e}")

        finally:
            self._finalise(result, start_time)

        return result

    # ── Stage implementations ───────────────────

    def _stage_ingest(
        self,
        input_source: str | pd.DataFrame,
        result: PipelineResult,
    ) -> Optional[pd.DataFrame]:
        logger.info("[ Stage 1 / 5 ]  Ingesting data...")

        try:
            if isinstance(input_source, pd.DataFrame):
                df = input_source.copy()
                logger.info(f"Loaded DataFrame directly — {len(df)} rows, {len(df.columns)} cols.")
            else:
                if not os.path.exists(input_source):
                    raise FileNotFoundError(f"Input file not found: {input_source}")
                ext = os.path.splitext(input_source)[1].lower()
                if ext == ".csv":
                    df = pd.read_csv(input_source)
                elif ext in (".xlsx", ".xls"):
                    df = pd.read_excel(input_source)
                elif ext == ".json":
                    df = pd.read_json(input_source)
                elif ext == ".parquet":
                    df = pd.read_parquet(input_source)
                else:
                    raise ValueError(f"Unsupported file format: {ext}")

                logger.info(
                    f"Loaded '{input_source}' — {len(df)} rows, {len(df.columns)} cols."
                )

            result.total_rows = len(df)
            return df

        except Exception as e:
            result.status        = "failed"
            result.error_message = f"Ingestion failed: {e}"
            logger.error(result.error_message)
            return None

    def _stage_validate(self, df: pd.DataFrame, result: PipelineResult):
        logger.info("[ Stage 2 / 5 ]  Validating data...")
        val_report = self.validator.validate(df)
        result.validation_issues = val_report.issue_count

        if val_report.is_clean:
            logger.success("Validation passed — no rule violations found.")
        else:
            logger.warning(
                f"Validation found {val_report.issue_count} issue(s):\n{val_report.summary()}"
            )
        return val_report

    def _stage_detect(self, df: pd.DataFrame, result: PipelineResult):
        logger.info("[ Stage 3 / 5 ]  Detecting anomalies...")
        anom_report = self.detector.detect(df)
        result.anomalies_detected = anom_report.anomaly_count

        if anom_report.is_clean:
            logger.success("Anomaly detection passed — no anomalies found.")
        else:
            logger.warning(
                f"Anomaly detection found {anom_report.anomaly_count} anomaly(s):\n"
                f"{anom_report.summary()}"
            )
        return anom_report

    def _stage_heal(
        self,
        df: pd.DataFrame,
        val_report,
        anom_report,
        result: PipelineResult,
        run_id: str,
    ) -> pd.DataFrame:
        total_issues = result.validation_issues + result.anomalies_detected
        logger.info(
            f"[ Stage 4 / 5 ]  Healing {total_issues} issue(s) "
            f"(max {self.healer.max_retries} attempt(s))..."
        )

        heal_result = self.healer.heal(
            df,
            validation_report=val_report,
            anomaly_report=anom_report,
            batch_id=run_id,
        )

        result.heal_attempts   = len(heal_result.attempts)
        result.issues_healed   = max(0, total_issues - heal_result.final_issue_count)
        result.issues_remaining = heal_result.final_issue_count

        if heal_result.quarantined:
            result.status = "quarantined"
            logger.warning("Batch quarantined — returning original DataFrame.")
            return df

        if heal_result.success:
            result.status = "success" if heal_result.final_issue_count == 0 else "partial"
            logger.success(
                f"Healing complete — {result.issues_healed} issue(s) fixed, "
                f"{result.issues_remaining} remaining."
            )
            return heal_result.fixed_df

        result.status = "failed"
        logger.error("Healing failed — returning original DataFrame.")
        return df

    def _stage_output(
        self,
        df: pd.DataFrame,
        result: PipelineResult,
        output_path: Optional[str],
        run_id: str,
    ) -> None:
        logger.info("[ Stage 5 / 5 ]  Writing output...")

        if output_path is None:
            output_path = os.path.join(self.output_dir, f"{run_id}_clean.csv")

        df.to_csv(output_path, index=False)
        result.output_path  = output_path
        result.rows_written = len(df)
        logger.success(f"Output written → {output_path}  ({len(df)} rows)")

    # ── Audit log ───────────────────────────────

    def _init_audit_db(self) -> None:
        """Create the audit table if it doesn't exist."""
        with sqlite3.connect(self.AUDIT_DB) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pipeline_runs (
                    run_id              TEXT PRIMARY KEY,
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
            """)
        logger.debug(f"Audit DB ready — {self.AUDIT_DB}")

    def _write_audit(self, result: PipelineResult) -> None:
        """Persist run metadata to SQLite audit log."""
        try:
            d = result.to_dict()
            with sqlite3.connect(self.AUDIT_DB) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO pipeline_runs VALUES
                    (:run_id, :input_path, :output_path, :status,
                     :total_rows, :rows_written, :validation_issues,
                     :anomalies_detected, :issues_healed, :issues_remaining,
                     :heal_attempts, :duration_seconds, :error_message, :timestamp)
                """, d)
            logger.debug(f"Audit record written for run_id={result.run_id}")
        except Exception as e:
            logger.warning(f"Failed to write audit record: {e}")

    def get_audit_history(self, limit: int = 20) -> pd.DataFrame:
        """Return the last N pipeline runs as a DataFrame."""
        with sqlite3.connect(self.AUDIT_DB) as conn:
            return pd.read_sql(
                f"SELECT * FROM pipeline_runs ORDER BY timestamp DESC LIMIT {limit}",
                conn,
            )

    # ── Helpers ─────────────────────────────────

    def _finalise(self, result: PipelineResult, start_time: float) -> PipelineResult:
        result.duration_seconds = round(time.time() - start_time, 3)
        self._write_audit(result)
        result.print_summary()
        return result


# ─────────────────────────────────────────────
#  CLI entry point
# ─────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Self-Healing Data Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py --input data/sample.csv
  python pipeline.py --input data/sample.csv --output output/clean.csv
  python pipeline.py --input data/sample.csv --no-heal
  python pipeline.py --input data/sample.csv --config config.yaml
  python pipeline.py --history
        """,
    )
    parser.add_argument("--input",   "-i", help="Path to input CSV/Excel/JSON/Parquet file")
    parser.add_argument("--output",  "-o", help="Path for the clean output CSV (optional)")
    parser.add_argument("--config",  "-c", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--no-heal",       action="store_true",   help="Skip healing stage")
    parser.add_argument("--history",       action="store_true",   help="Print last 20 audit runs")
    parser.add_argument("--run-id",        help="Custom run ID (optional)")
    return parser.parse_args()


def main() -> None:
    _setup_logger()
    args = _parse_args()

    pipeline = Pipeline(args.config)

    # Show audit history and exit
    if args.history:
        history = pipeline.get_audit_history(20)
        if history.empty:
            print("No pipeline runs recorded yet.")
        else:
            pd.set_option("display.max_columns", None)
            pd.set_option("display.width", 120)
            print("\nLast 20 pipeline runs:\n")
            print(history.to_string(index=False))
        return

    if not args.input:
        print("Error: --input is required. Use --help for usage.")
        sys.exit(1)

    result = pipeline.run(
        input_source=args.input,
        output_path=args.output,
        heal=not args.no_heal,
        run_id=args.run_id,
    )

    # Exit code reflects run status
    sys.exit(0 if result.status in ("success", "partial") else 1)


if __name__ == "__main__":
    main()