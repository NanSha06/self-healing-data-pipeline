"""
validator.py
------------
Rule-based validation engine for the Self-Healing Data Pipeline.

Validates a pandas DataFrame against a schema defined in config.yaml.
Returns a structured ValidationReport with all issues found, which is
passed to the LLM healer when problems are detected.

Usage:
    from validator import Validator
    v = Validator("config.yaml")
    report = v.validate(df)
    if not report.is_clean:
        print(report.summary())
"""

import re
import yaml
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger
from typing import Any


# ─────────────────────────────────────────────
#  Data classes for structured error reporting
# ─────────────────────────────────────────────

@dataclass
class ValidationIssue:
    """Represents a single validation failure on a column."""
    column: str
    issue_type: str          # null_values | type_mismatch | out_of_range | regex_mismatch | invalid_value | duplicate_id
    description: str
    affected_count: int
    affected_rows: list[int] # row indices with the problem
    sample_values: list[Any] # up to 5 bad sample values

    def to_dict(self) -> dict:
        return {
            "column": self.column,
            "issue_type": self.issue_type,
            "description": self.description,
            "affected_count": self.affected_count,
            "affected_rows": self.affected_rows[:10],  # cap at 10 for prompt size
            "sample_values": [str(v) for v in self.sample_values[:5]],
        }


@dataclass
class ValidationReport:
    """Full result of a validation run."""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    total_rows: int = 0
    total_columns: int = 0
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return len(self.issues) == 0

    @property
    def issue_count(self) -> int:
        return len(self.issues)

    def summary(self) -> str:
        if self.is_clean:
            return f"✅ Validation passed — {self.total_rows} rows, {self.total_columns} columns, no issues found."
        lines = [
            f"❌ Validation failed — {self.issue_count} issue(s) found across {self.total_rows} rows.",
        ]
        for i, issue in enumerate(self.issues, 1):
            lines.append(
                f"  [{i}] {issue.column!r} | {issue.issue_type} | {issue.description} "
                f"({issue.affected_count} row(s) affected)"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "total_rows": self.total_rows,
            "total_columns": self.total_columns,
            "is_clean": self.is_clean,
            "issue_count": self.issue_count,
            "issues": [issue.to_dict() for issue in self.issues],
        }


# ─────────────────────────────────────────────
#  Validator
# ─────────────────────────────────────────────

class Validator:
    """
    Validates a DataFrame against rules defined in config.yaml.

    Schema format (config.yaml):
        schema:
          columns:
            age:
              type: int
              nullable: false
              min: 0
              max: 120
            email:
              type: str
              nullable: false
              regex: "^[\\w.-]+@[\\w.-]+\\.\\w+$"
            country:
              type: str
              allowed_values: [IN, US, UK, DE, FR, CA]
            signup_date:
              type: date
              nullable: false
            id:
              type: int
              unique: true
    """

    # Map config type strings to Python types
    TYPE_MAP = {
        "int":   (int,),
        "float": (float, int),
        "str":   (str,),
        "bool":  (bool,),
    }

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        self.schema: dict = config.get("schema", {}).get("columns", {})
        logger.info(f"Validator loaded schema with {len(self.schema)} column rule(s).")

    # ── Public entry point ──────────────────────

    def validate(self, df: pd.DataFrame) -> ValidationReport:
        """
        Run all validation checks on the DataFrame.
        Returns a ValidationReport with every issue found.
        """
        report = ValidationReport(
            total_rows=len(df),
            total_columns=len(df.columns),
        )

        logger.info(f"Starting validation — {len(df)} rows, {len(df.columns)} columns.")

        for col_name, rules in self.schema.items():
            if col_name not in df.columns:
                logger.warning(f"Column '{col_name}' defined in schema but not found in DataFrame — skipping.")
                continue

            series = df[col_name]

            # Run each check; collect any issues returned
            checks = [
                self._check_nulls(series, col_name, rules),
                self._check_types(series, col_name, rules),
                self._check_range(series, col_name, rules),
                self._check_regex(series, col_name, rules),
                self._check_allowed_values(series, col_name, rules),
                self._check_date_format(series, col_name, rules),
                self._check_unique(series, col_name, rules),
            ]

            for issue in checks:
                if issue:
                    report.issues.append(issue)
                    logger.warning(
                        f"Issue in '{col_name}': {issue.issue_type} — {issue.affected_count} row(s) affected."
                    )

        if report.is_clean:
            logger.success("Validation passed — no issues found.")
        else:
            logger.error(f"Validation failed — {report.issue_count} issue(s) found.")

        return report

    # ── Individual checks ───────────────────────

    def _check_nulls(self, series: pd.Series, col: str, rules: dict) -> ValidationIssue | None:
        """Fail if nulls exist in a non-nullable column."""
        if rules.get("nullable", True):
            return None

        null_mask = series.isna()
        count = null_mask.sum()
        if count == 0:
            return None

        return ValidationIssue(
            column=col,
            issue_type="null_values",
            description=f"Column is non-nullable but contains {count} null/missing value(s).",
            affected_count=int(count),
            affected_rows=series.index[null_mask].tolist(),
            sample_values=[None] * min(count, 5),
        )

    def _check_types(self, series: pd.Series, col: str, rules: dict) -> ValidationIssue | None:
        """Fail if non-null values cannot be cast to the declared type."""
        declared_type = rules.get("type")
        if not declared_type or declared_type in ("date", "str"):
            # str accepts everything; date is checked separately
            return None

        expected_types = self.TYPE_MAP.get(declared_type)
        if not expected_types:
            return None

        non_null = series.dropna()
        bad_mask = ~non_null.apply(lambda v: isinstance(v, expected_types) or self._can_cast(v, declared_type))
        bad_rows = non_null.index[bad_mask].tolist()
        count = len(bad_rows)

        if count == 0:
            return None

        return ValidationIssue(
            column=col,
            issue_type="type_mismatch",
            description=f"Expected type '{declared_type}' but found {count} value(s) that cannot be converted.",
            affected_count=count,
            affected_rows=bad_rows,
            sample_values=non_null[bad_mask].tolist(),
        )

    def _check_range(self, series: pd.Series, col: str, rules: dict) -> ValidationIssue | None:
        """Fail if numeric values fall outside [min, max]."""
        has_min = "min" in rules
        has_max = "max" in rules
        if not has_min and not has_max:
            return None

        numeric = pd.to_numeric(series, errors="coerce").dropna()
        bad_mask = pd.Series([False] * len(numeric), index=numeric.index)

        if has_min:
            bad_mask |= numeric < rules["min"]
        if has_max:
            bad_mask |= numeric > rules["max"]

        count = bad_mask.sum()
        if count == 0:
            return None

        bound_desc = []
        if has_min:
            bound_desc.append(f"min={rules['min']}")
        if has_max:
            bound_desc.append(f"max={rules['max']}")

        return ValidationIssue(
            column=col,
            issue_type="out_of_range",
            description=f"Values outside allowed range ({', '.join(bound_desc)}): {count} violation(s).",
            affected_count=int(count),
            affected_rows=numeric.index[bad_mask].tolist(),
            sample_values=numeric[bad_mask].tolist(),
        )

    def _check_regex(self, series: pd.Series, col: str, rules: dict) -> ValidationIssue | None:
        """Fail if string values don't match the declared regex pattern."""
        pattern = rules.get("regex")
        if not pattern:
            return None

        non_null = series.dropna().astype(str)
        compiled = re.compile(pattern)
        bad_mask = ~non_null.apply(lambda v: bool(compiled.match(v)))
        bad_rows = non_null.index[bad_mask].tolist()
        count = len(bad_rows)

        if count == 0:
            return None

        return ValidationIssue(
            column=col,
            issue_type="regex_mismatch",
            description=f"Values do not match required pattern '{pattern}': {count} violation(s).",
            affected_count=count,
            affected_rows=bad_rows,
            sample_values=non_null[bad_mask].tolist(),
        )

    def _check_allowed_values(self, series: pd.Series, col: str, rules: dict) -> ValidationIssue | None:
        """Fail if values are not in the allowed_values list."""
        allowed = rules.get("allowed_values")
        if not allowed:
            return None

        non_null = series.dropna()
        bad_mask = ~non_null.isin(allowed)
        bad_rows = non_null.index[bad_mask].tolist()
        count = len(bad_rows)

        if count == 0:
            return None

        return ValidationIssue(
            column=col,
            issue_type="invalid_value",
            description=f"Values not in allowed list {allowed}: {count} violation(s).",
            affected_count=count,
            affected_rows=bad_rows,
            sample_values=non_null[bad_mask].tolist(),
        )

    def _check_date_format(self, series: pd.Series, col: str, rules: dict) -> ValidationIssue | None:
        """Fail if values declared as 'date' type cannot be parsed."""
        if rules.get("type") != "date":
            return None

        non_null = series.dropna().astype(str)
        bad_rows = []
        bad_values = []

        for idx, val in non_null.items():
            try:
                pd.to_datetime(val)
            except (ValueError, TypeError):
                bad_rows.append(idx)
                bad_values.append(val)

        count = len(bad_rows)
        if count == 0:
            return None

        return ValidationIssue(
            column=col,
            issue_type="type_mismatch",
            description=f"Values declared as 'date' but could not be parsed: {count} violation(s).",
            affected_count=count,
            affected_rows=bad_rows,
            sample_values=bad_values,
        )

    def _check_unique(self, series: pd.Series, col: str, rules: dict) -> ValidationIssue | None:
        """Fail if a column declared unique contains duplicate values."""
        if not rules.get("unique", False):
            return None

        duplicated = series[series.duplicated(keep=False)]
        count = len(duplicated)
        if count == 0:
            return None

        return ValidationIssue(
            column=col,
            issue_type="duplicate_id",
            description=f"Column declared unique but contains {count} duplicate value(s).",
            affected_count=count,
            affected_rows=duplicated.index.tolist(),
            sample_values=duplicated.tolist(),
        )

    # ── Helpers ─────────────────────────────────

    @staticmethod
    def _can_cast(value: Any, type_str: str) -> bool:
        """Try casting a value to the declared type. Returns True if successful."""
        try:
            if type_str == "int":
                int(value)
            elif type_str == "float":
                float(value)
            return True
        except (ValueError, TypeError):
            return False


# ─────────────────────────────────────────────
#  Quick test — run directly to see output
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import json

    # Inline config for standalone testing (no config.yaml needed)
    TEST_SCHEMA = {
        "schema": {
            "columns": {
                "id":          {"type": "int",   "nullable": False, "unique": True},
                "name":        {"type": "str",   "nullable": False},
                "age":         {"type": "int",   "nullable": False, "min": 0, "max": 120},
                "email":       {"type": "str",   "nullable": False, "regex": r"^[\w.-]+@[\w.-]+\.\w+$"},
                "country":     {"type": "str",   "allowed_values": ["IN", "US", "UK", "DE", "FR", "CA"]},
                "salary":      {"type": "float", "nullable": True,  "min": 0},
                "signup_date": {"type": "date",  "nullable": False},
            }
        }
    }

    # Write a temp config for the test
    import tempfile, os
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.dump(TEST_SCHEMA, f)
        tmp_config = f.name

    try:
        df = pd.read_csv("data/sample.csv")
        v = Validator(tmp_config)
        report = v.validate(df)

        print("\n" + "="*60)
        print(report.summary())
        print("="*60)
        print("\nFull report (JSON):")
        print(json.dumps(report.to_dict(), indent=2))
    finally:
        os.unlink(tmp_config)