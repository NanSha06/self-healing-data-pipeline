"""
pipeline.py
-----------
Main orchestrator for the Self-Healing Data Pipeline.

Ties together ingestion, validation, anomaly detection, healing, and output
into a single end-to-end run. Every stage is logged and timed. Results are
written to the output directory and the full 3-table audit log (logger.py).

Usage (command line):
    python pipeline.py --input data/sample.csv
    python pipeline.py --input data/sample.csv --output output/clean.csv
    python pipeline.py --input data/sample.csv --config config.yaml --no-heal
    python pipeline.py --history
    python pipeline.py --history --status failed
    python pipeline.py --repairs <run_id>
    python pipeline.py --issues  <run_id>
    python pipeline.py --stats
    python pipeline.py --export-audit

Usage (as a module):
    from pipeline import Pipeline
    pipeline = Pipeline("config.yaml")
    result   = pipeline.run("data/sample.csv")
"""

import os
import sys
import uuid
import argparse
import time
import pandas as pd
import yaml
from datetime import datetime
from dataclasses import dataclass, field
from loguru import logger
from dotenv import load_dotenv
from typing import Optional

from ingestion import Ingestor
from validator import Validator
from anomaly   import AnomalyDetector
from healer    import Healer
from logger    import AuditLogger

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
        print(f"  Run ID            : {self.run_id}")
        print(f"  Status            : {self.status.upper()}")
        print(f"  Input             : {self.input_path}")
        print(f"  Output            : {self.output_path or 'N/A'}")
        print(f"  Total rows        : {self.total_rows}")
        print(f"  Rows written      : {self.rows_written}")
        print(f"  Validation issues : {self.validation_issues}")
        print(f"  Anomalies found   : {self.anomalies_detected}")
        print(f"  Issues healed     : {self.issues_healed}")
        print(f"  Issues remaining  : {self.issues_remaining}")
        print(f"  Heal attempts     : {self.heal_attempts}")
        print(f"  Duration          : {self.duration_seconds:.2f}s")
        if self.error_message:
            print(f"  Error             : {self.error_message}")
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
        1. Ingest   — load CSV / Excel / JSON / Parquet into pandas
        2. Validate — schema + rule checks        (validator.py)
        3. Detect   — statistical anomaly checks  (anomaly.py)
        4. Heal     — LLM-powered auto-fix        (healer.py)   [optional]
        5. Output   — write clean CSV to disk
        6. Audit    — persist full 3-table record  (logger.py)
    """

    AUDIT_DB = "pipeline_audit.db"

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path

        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.pipeline_cfg = self.config.get("pipeline", {})
        self.output_dir   = self.pipeline_cfg.get("output_dir", "output")
        os.makedirs(self.output_dir, exist_ok=True)

        # Core modules
        self.ingestor  = Ingestor()
        self.validator = Validator(config_path)
        self.detector  = AnomalyDetector(config_path)
        self.healer    = Healer(config_path)

        # Full 3-table audit logger (replaces inline SQLite from old version)
        self.audit = AuditLogger(self.AUDIT_DB)

        logger.info(f"Pipeline initialised. config={config_path}")

    # ── Public entry point ──────────────────────

    def run(
        self,
        input_source: "str | pd.DataFrame",
        output_path: Optional[str] = None,
        heal: bool = True,
        run_id: Optional[str] = None,
    ) -> PipelineResult:
        """
        Execute a full pipeline run.

        Args:
            input_source : path to a CSV/Excel/JSON/Parquet file, or a DataFrame.
            output_path  : destination CSV path.
                           Defaults to output/<run_id>_clean.csv
            heal         : set False to skip healing (validate + detect only).
            run_id       : optional custom run identifier.

        Returns:
            PipelineResult with full run metadata.
        """
        run_id     = run_id or str(uuid.uuid4())[:8]
        result     = PipelineResult(
            run_id=run_id,
            input_path=str(input_source) if isinstance(input_source, str) else "<DataFrame>",
        )
        start_time = time.time()

        logger.info(f"{'─'*50}")
        logger.info(f"Pipeline run started  run_id={run_id}")
        logger.info(f"{'─'*50}")

        # Keep reports in scope so _finalise can write all 3 audit tables
        val_report  = None
        anom_report = None
        heal_result = None

        try:
            # ── Stage 1: Ingest ──────────────────
            df = self._stage_ingest(input_source, result)
            if df is None:
                return self._finalise(result, start_time, val_report, anom_report, heal_result)

            # ── Stage 2: Validate ────────────────
            val_report = self._stage_validate(df, result)

            # ── Stage 3: Detect anomalies ────────
            anom_report = self._stage_detect(df, result)

            total_issues = result.validation_issues + result.anomalies_detected

            # ── Stage 4: Heal ────────────────────
            if heal and total_issues > 0:
                df, heal_result = self._stage_heal(
                    df, val_report, anom_report, result, run_id
                )
            elif total_issues == 0:
                logger.success("Data is clean — skipping healing stage.")
                result.status = "success"
            else:
                logger.info("Healing disabled — skipping healing stage.")
                result.status           = "partial"
                result.issues_remaining = total_issues

            # ── Stage 5: Output ──────────────────
            self._stage_output(df, result, output_path, run_id)

        except Exception as e:
            result.status        = "failed"
            result.error_message = str(e)
            logger.exception(f"Pipeline run failed: {e}")

        finally:
            self._finalise(result, start_time, val_report, anom_report, heal_result)

        return result

    # ── Stage implementations ───────────────────

    def _stage_ingest(
        self,
        input_source: "str | pd.DataFrame",
        result: PipelineResult,
    ) -> Optional[pd.DataFrame]:
        logger.info("[ Stage 1 / 5 ]  Ingesting data...")

        ingest_result = self.ingestor.load(input_source)

        if not ingest_result.success:
            result.status        = "failed"
            result.error_message = ingest_result.error_message
            return None

        result.total_rows = ingest_result.row_count
        return ingest_result.df

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
    ):
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

        result.heal_attempts    = len(heal_result.attempts)
        result.issues_healed    = max(0, total_issues - heal_result.final_issue_count)
        result.issues_remaining = heal_result.final_issue_count

        if heal_result.quarantined:
            result.status = "quarantined"
            logger.warning("Batch quarantined — returning original DataFrame.")
            return df, heal_result

        if heal_result.success:
            result.status = "success" if heal_result.final_issue_count == 0 else "partial"
            logger.success(
                f"Healing complete — {result.issues_healed} issue(s) fixed, "
                f"{result.issues_remaining} remaining."
            )
            return heal_result.fixed_df, heal_result

        result.status = "failed"
        logger.error("Healing failed — returning original DataFrame.")
        return df, heal_result

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

    # ── Finalise + full audit write ─────────────

    def _finalise(
        self,
        result: PipelineResult,
        start_time: float,
        val_report=None,
        anom_report=None,
        heal_result=None,
    ) -> PipelineResult:
        result.duration_seconds = round(time.time() - start_time, 3)

        # 1. Run summary row
        self.audit.log_run(result)

        # 2. Per-issue detail (validation + anomaly)
        if val_report is not None or anom_report is not None:
            self.audit.log_issues(
                run_id=result.run_id,
                validation_report=val_report,
                anomaly_report=anom_report,
            )

        # 3. LLM repair attempts
        if heal_result is not None:
            self.audit.log_heal(heal_result, run_id=result.run_id)

        result.print_summary()
        return result

    # ── Convenience query pass-throughs ─────────

    def get_history(self, limit: int = 20, status: Optional[str] = None) -> pd.DataFrame:
        """Return recent pipeline runs from the audit log."""
        return self.audit.get_runs(limit=limit, status=status)

    def get_repairs(self, run_id: Optional[str] = None, limit: int = 50) -> pd.DataFrame:
        """Return LLM repair attempt records from the audit log."""
        return self.audit.get_repairs(run_id=run_id, limit=limit)

    def get_issues(self, run_id: Optional[str] = None, limit: int = 100) -> pd.DataFrame:
        """Return per-issue event records from the audit log."""
        return self.audit.get_issues(run_id=run_id, limit=limit)

    def print_stats(self) -> None:
        """Print aggregate statistics across all pipeline runs."""
        self.audit.print_stats()

    def export_audit(self, output_path: str = "audit_export.json") -> None:
        """Export the full audit log to a JSON file."""
        self.audit.export_json(output_path)


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
  python pipeline.py --history --status failed
  python pipeline.py --repairs <run_id>
  python pipeline.py --issues  <run_id>
  python pipeline.py --stats
  python pipeline.py --export-audit
        """,
    )
    parser.add_argument("--input",        "-i", help="Path to input file (CSV/Excel/JSON/Parquet)")
    parser.add_argument("--output",       "-o", help="Path for clean output CSV (optional)")
    parser.add_argument("--config",       "-c", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--no-heal",            action="store_true",   help="Skip healing stage")
    parser.add_argument("--run-id",             help="Custom run ID (optional)")
    parser.add_argument("--history",            action="store_true",   help="Show last 20 runs")
    parser.add_argument("--status",             help="Filter --history by status (success/failed/partial/quarantined)")
    parser.add_argument("--repairs",            metavar="RUN_ID",      help="Show repair attempts for a run ID")
    parser.add_argument("--issues",             metavar="RUN_ID",      help="Show issue events for a run ID")
    parser.add_argument("--stats",              action="store_true",   help="Show aggregate stats")
    parser.add_argument("--export-audit",       action="store_true",   help="Export audit log to audit_export.json")
    return parser.parse_args()


def main() -> None:
    _setup_logger()
    args     = _parse_args()
    pipeline = Pipeline(args.config)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 140)
    pd.set_option("display.max_colwidth", 60)

    # ── Query / reporting commands ───────────────
    if args.history:
        df = pipeline.get_history(limit=20, status=args.status)
        if df.empty:
            print("No pipeline runs recorded yet.")
        else:
            print("\nRecent pipeline runs:\n")
            print(df.to_string(index=False))
        return

    if args.repairs:
        df = pipeline.get_repairs(run_id=args.repairs)
        if df.empty:
            print(f"No repair attempts found for run_id={args.repairs}")
        else:
            print(f"\nRepair attempts for run_id={args.repairs}:\n")
            print(df.to_string(index=False))
        return

    if args.issues:
        df = pipeline.get_issues(run_id=args.issues)
        if df.empty:
            print(f"No issue events found for run_id={args.issues}")
        else:
            print(f"\nIssue events for run_id={args.issues}:\n")
            print(df.to_string(index=False))
        return

    if args.stats:
        pipeline.print_stats()
        return

    if args.export_audit:
        pipeline.export_audit("audit_export.json")
        print("Exported → audit_export.json")
        return

    # ── Main run ─────────────────────────────────
    if not args.input:
        print("Error: --input is required. Use --help for usage.")
        sys.exit(1)

    result = pipeline.run(
        input_source=args.input,
        output_path=args.output,
        heal=not args.no_heal,
        run_id=args.run_id,
    )

    # Non-zero exit on hard failure or quarantine
    sys.exit(0 if result.status in ("success", "partial") else 1)


if __name__ == "__main__":
    main()