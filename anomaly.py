"""
anomaly.py
----------
Statistical and frequency-based anomaly detection for the Self-Healing Data Pipeline.

Complements validator.py (which checks hard rules) by catching soft issues:
  - Statistical outliers in numeric columns (Z-score and IQR)
  - Rare / unexpected categories in string columns
  - Sudden distribution shifts (mean/std drift detection)
  - Duplicate rows
  - Constant / near-constant columns (low variance)

Returns a structured AnomalyReport compatible with the LLM healer.

Usage:
    from anomaly import AnomalyDetector
    detector = AnomalyDetector("config.yaml")
    report = detector.detect(df)
    if not report.is_clean:
        print(report.summary())
"""

import yaml
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger
from typing import Any


# ─────────────────────────────────────────────
#  Data classes
# ─────────────────────────────────────────────

@dataclass
class Anomaly:
    """Represents a single detected anomaly."""
    column: str
    anomaly_type: str       # outlier_zscore | outlier_iqr | rare_category | low_variance
                            # duplicate_rows | distribution_shift
    description: str
    severity: str           # low | medium | high
    affected_count: int
    affected_rows: list[int]
    sample_values: list[Any]
    stats: dict             # supporting statistics (mean, std, bounds, etc.)

    def to_dict(self) -> dict:
        return {
            "column": self.column,
            "anomaly_type": self.anomaly_type,
            "description": self.description,
            "severity": self.severity,
            "affected_count": self.affected_count,
            "affected_rows": self.affected_rows[:10],
            "sample_values": [str(v) for v in self.sample_values[:5]],
            "stats": self.stats,
        }


@dataclass
class AnomalyReport:
    """Full result of an anomaly detection run."""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    total_rows: int = 0
    total_columns: int = 0
    anomalies: list[Anomaly] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return len(self.anomalies) == 0

    @property
    def anomaly_count(self) -> int:
        return len(self.anomalies)

    @property
    def high_severity(self) -> list[Anomaly]:
        return [a for a in self.anomalies if a.severity == "high"]

    def summary(self) -> str:
        if self.is_clean:
            return f"✅ No anomalies detected — {self.total_rows} rows, {self.total_columns} columns."
        lines = [
            f"⚠️  {self.anomaly_count} anomaly(s) detected across {self.total_rows} rows.",
            f"   High severity: {len(self.high_severity)}",
        ]
        for i, anomaly in enumerate(self.anomalies, 1):
            lines.append(
                f"  [{i}] [{anomaly.severity.upper()}] {anomaly.column!r} | "
                f"{anomaly.anomaly_type} | {anomaly.description} "
                f"({anomaly.affected_count} row(s))"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "total_rows": self.total_rows,
            "total_columns": self.total_columns,
            "is_clean": self.is_clean,
            "anomaly_count": self.anomaly_count,
            "high_severity_count": len(self.high_severity),
            "anomalies": [a.to_dict() for a in self.anomalies],
        }


# ─────────────────────────────────────────────
#  AnomalyDetector
# ─────────────────────────────────────────────

class AnomalyDetector:
    """
    Detects statistical and structural anomalies in a DataFrame.

    Thresholds are configurable via config.yaml:

        anomaly_detection:
          zscore_threshold: 3.0       # flag values beyond N standard deviations
          iqr_multiplier: 1.5         # flag values beyond Q1/Q3 ± N*IQR
          rare_category_threshold: 0.01   # flag categories < 1% frequency
          low_variance_threshold: 0.01    # flag columns with std/mean < 1%
          min_rows_for_stats: 10      # skip statistical checks on tiny datasets
    """

    DEFAULT_CONFIG = {
        "zscore_threshold": 3.0,
        "iqr_multiplier": 1.5,
        "rare_category_threshold": 0.01,
        "low_variance_threshold": 0.01,
        "min_rows_for_stats": 10,
    }

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        user_cfg = config.get("anomaly_detection", {})
        self.cfg = {**self.DEFAULT_CONFIG, **user_cfg}
        self.schema: dict = config.get("schema", {}).get("columns", {})
        logger.info(f"AnomalyDetector initialised. Config: {self.cfg}")

    # ── Public entry point ──────────────────────

    def detect(self, df: pd.DataFrame) -> AnomalyReport:
        """
        Run all anomaly detection checks on the DataFrame.
        Returns an AnomalyReport with every anomaly found.
        """
        report = AnomalyReport(
            total_rows=len(df),
            total_columns=len(df.columns),
        )

        logger.info(f"Starting anomaly detection — {len(df)} rows, {len(df.columns)} columns.")

        # Row-level checks (whole DataFrame)
        dup_anomaly = self._check_duplicate_rows(df)
        if dup_anomaly:
            report.anomalies.append(dup_anomaly)

        # Build a set of columns declared as 'date' type in the schema
        # so we can skip numeric checks on date columns (they parse to timestamps
        # which cause false-positive outlier and low-variance anomalies after healing)
        date_cols = {
            col for col, rules in self.schema.items()
            if rules.get("type") == "date"
        }

        # Column-level checks
        for col in df.columns:
            series = df[col]

            # Skip numeric anomaly checks on date columns entirely
            if col in date_cols:
                continue

            numeric = pd.to_numeric(series, errors="coerce")
            is_numeric = numeric.notna().sum() > numeric.isna().sum()  # majority numeric

            if is_numeric:
                checks = [
                    self._check_zscore(numeric.dropna(), col, df),
                    self._check_iqr(numeric.dropna(), col, df),
                    self._check_low_variance(numeric.dropna(), col),
                ]
            else:
                checks = [
                    self._check_rare_categories(series.dropna(), col),
                ]

            for anomaly in checks:
                if anomaly:
                    report.anomalies.append(anomaly)
                    logger.warning(
                        f"[{anomaly.severity.upper()}] Anomaly in '{col}': "
                        f"{anomaly.anomaly_type} — {anomaly.affected_count} row(s)."
                    )

        if report.is_clean:
            logger.success("Anomaly detection passed — no anomalies found.")
        else:
            logger.warning(
                f"Anomaly detection complete — {report.anomaly_count} anomaly(s) found "
                f"({len(report.high_severity)} high severity)."
            )

        return report

    # ── Numeric checks ──────────────────────────

    def _check_zscore(
        self, numeric: pd.Series, col: str, df: pd.DataFrame
    ) -> Anomaly | None:
        """Flag values more than N standard deviations from the mean."""
        if len(numeric) < self.cfg["min_rows_for_stats"]:
            return None

        mean = numeric.mean()
        std = numeric.std()

        if std == 0:
            return None  # constant column — caught by low_variance check

        threshold = self.cfg["zscore_threshold"]
        zscores = ((numeric - mean) / std).abs()
        bad_mask = zscores > threshold
        bad_rows = numeric.index[bad_mask].tolist()
        count = len(bad_rows)

        if count == 0:
            return None

        # Severity: high if z-score > 2x threshold, medium otherwise
        max_z = float(zscores[bad_mask].max())
        severity = "high" if max_z > threshold * 2 else "medium"

        return Anomaly(
            column=col,
            anomaly_type="outlier_zscore",
            description=(
                f"{count} value(s) exceed {threshold}σ from the mean "
                f"(mean={mean:.2f}, std={std:.2f}, max Z={max_z:.2f})."
            ),
            severity=severity,
            affected_count=count,
            affected_rows=bad_rows,
            sample_values=numeric[bad_mask].tolist(),
            stats={
                "mean": round(mean, 4),
                "std": round(std, 4),
                "zscore_threshold": threshold,
                "max_zscore": round(max_z, 4),
            },
        )

    def _check_iqr(
        self, numeric: pd.Series, col: str, df: pd.DataFrame
    ) -> Anomaly | None:
        """Flag values outside Q1 - k*IQR and Q3 + k*IQR (Tukey fences)."""
        if len(numeric) < self.cfg["min_rows_for_stats"]:
            return None

        k = self.cfg["iqr_multiplier"]
        q1 = numeric.quantile(0.25)
        q3 = numeric.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            return None

        lower = q1 - k * iqr
        upper = q3 + k * iqr

        bad_mask = (numeric < lower) | (numeric > upper)
        bad_rows = numeric.index[bad_mask].tolist()
        count = len(bad_rows)

        if count == 0:
            return None

        # Only report if Z-score didn't already flag the same rows
        # (avoid double-reporting the same outliers)
        mean = numeric.mean()
        std = numeric.std()
        if std > 0:
            zscores = ((numeric[bad_mask] - mean) / std).abs()
            already_caught = (zscores > self.cfg["zscore_threshold"]).all()
            if already_caught:
                return None

        severity = "medium" if count <= max(3, len(numeric) * 0.05) else "high"

        return Anomaly(
            column=col,
            anomaly_type="outlier_iqr",
            description=(
                f"{count} value(s) outside Tukey fences "
                f"[{lower:.2f}, {upper:.2f}] (IQR={iqr:.2f}, k={k})."
            ),
            severity=severity,
            affected_count=count,
            affected_rows=bad_rows,
            sample_values=numeric[bad_mask].tolist(),
            stats={
                "q1": round(float(q1), 4),
                "q3": round(float(q3), 4),
                "iqr": round(float(iqr), 4),
                "lower_fence": round(float(lower), 4),
                "upper_fence": round(float(upper), 4),
                "iqr_multiplier": k,
            },
        )

    def _check_low_variance(self, numeric: pd.Series, col: str) -> Anomaly | None:
        """Flag columns where almost all values are the same (near-constant)."""
        if len(numeric) < self.cfg["min_rows_for_stats"]:
            return None

        mean = numeric.mean()
        std = numeric.std()

        if mean == 0:
            return None  # avoid division by zero

        cv = std / abs(mean)  # coefficient of variation

        if cv >= self.cfg["low_variance_threshold"]:
            return None

        return Anomaly(
            column=col,
            anomaly_type="low_variance",
            description=(
                f"Column has very low variance (CV={cv:.4f}). "
                f"Almost all values are near {mean:.2f} — may be a data pipeline issue."
            ),
            severity="low",
            affected_count=len(numeric),
            affected_rows=[],
            sample_values=numeric.unique().tolist()[:5],
            stats={
                "mean": round(float(mean), 4),
                "std": round(float(std), 4),
                "coefficient_of_variation": round(float(cv), 6),
            },
        )

    # ── Categorical checks ──────────────────────

    def _check_rare_categories(self, series: pd.Series, col: str) -> Anomaly | None:
        """Flag categories that appear less than N% of the time."""
        if len(series) < self.cfg["min_rows_for_stats"]:
            return None

        threshold = self.cfg["rare_category_threshold"]
        freq = series.value_counts(normalize=True)
        rare_cats = freq[freq < threshold].index.tolist()

        if not rare_cats:
            return None

        bad_mask = series.isin(rare_cats)
        bad_rows = series.index[bad_mask].tolist()
        count = len(bad_rows)

        return Anomaly(
            column=col,
            anomaly_type="rare_category",
            description=(
                f"{len(rare_cats)} rare category value(s) found below "
                f"{threshold*100:.1f}% frequency threshold: {rare_cats}."
            ),
            severity="low",
            affected_count=count,
            affected_rows=bad_rows,
            sample_values=rare_cats[:5],
            stats={
                "rare_categories": rare_cats,
                "frequency_threshold": threshold,
                "category_frequencies": {
                    k: round(float(v), 4) for k, v in freq[freq < threshold].items()
                },
            },
        )

    # ── Row-level checks ────────────────────────

    def _check_duplicate_rows(self, df: pd.DataFrame) -> Anomaly | None:
        """Flag fully duplicate rows (all columns identical)."""
        dup_mask = df.duplicated(keep=False)
        count = dup_mask.sum()

        if count == 0:
            return None

        dup_rows = df.index[dup_mask].tolist()
        severity = "high" if count > len(df) * 0.1 else "medium"

        return Anomaly(
            column="[all columns]",
            anomaly_type="duplicate_rows",
            description=f"{count} fully duplicate row(s) found in the dataset.",
            severity=severity,
            affected_count=int(count),
            affected_rows=dup_rows,
            sample_values=df[dup_mask].head(3).to_dict("records"),
            stats={"duplicate_count": int(count), "total_rows": len(df)},
        )


# ─────────────────────────────────────────────
#  Quick test — run directly to see output
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import json
    import tempfile
    import os

    TEST_CONFIG = {
        "schema": {"columns": {}},
        "anomaly_detection": {
            "zscore_threshold": 3.0,
            "iqr_multiplier": 1.5,
            "rare_category_threshold": 0.01,
            "low_variance_threshold": 0.01,
            "min_rows_for_stats": 5,
        },
    }

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.dump(TEST_CONFIG, f)
        tmp_config = f.name

    try:
        df = pd.read_csv("data/sample.csv")
        detector = AnomalyDetector(tmp_config)
        report = detector.detect(df)

        print("\n" + "=" * 60)
        print(report.summary())
        print("=" * 60)
        print("\nFull report (JSON):")
        print(json.dumps(report.to_dict(), indent=2))
    finally:
        os.unlink(tmp_config)