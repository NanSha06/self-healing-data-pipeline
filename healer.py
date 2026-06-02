"""
healer.py
---------
LLM-powered diagnosis and auto-fix engine for the Self-Healing Data Pipeline.

When validator.py or anomaly.py detect issues, the Healer:
  1. Builds a structured prompt with the error report + data sample
  2. Calls NVIDIA Llama 3.1 70B and requests a JSON response containing:
       - diagnosis   : plain-English explanation of the root cause
       - fix_code    : a Python function apply_fix(df) -> pd.DataFrame
       - confidence  : 0.0–1.0 score for how certain the model is
  3. Executes the fix safely against a DataFrame COPY
  4. Re-runs validation/anomaly checks to confirm the fix worked
  5. Accepts the fix only if it improves the data quality score
  6. Retries with a refined prompt if the fix fails (up to max_retries)
  7. Quarantines the batch if all retries are exhausted

Usage:
    from healer import Healer
    from validator import Validator, ValidationReport
    from anomaly import AnomalyDetector, AnomalyReport

    healer  = Healer("config.yaml")
    result  = healer.heal(df, validation_report=report)
    if result.success:
        clean_df = result.fixed_df
"""

import os
import json
import textwrap
import traceback
import yaml
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger
from openai import OpenAI
from dotenv import load_dotenv
from typing import Optional

from validator import Validator, ValidationReport
from anomaly import AnomalyDetector, AnomalyReport

load_dotenv()


# ─────────────────────────────────────────────
#  Data classes
# ─────────────────────────────────────────────

@dataclass
class HealAttempt:
    """Records a single healing attempt."""
    attempt_number: int
    diagnosis: str
    fix_code: str
    confidence: float
    outcome: str               # success | failed_execution | failed_validation | low_confidence
    error_message: str = ""
    issues_before: int = 0
    issues_after: int = 0

    def to_dict(self) -> dict:
        return {
            "attempt_number": self.attempt_number,
            "diagnosis": self.diagnosis,
            "fix_code": self.fix_code,
            "confidence": self.confidence,
            "outcome": self.outcome,
            "error_message": self.error_message,
            "issues_before": self.issues_before,
            "issues_after": self.issues_after,
        }


@dataclass
class HealResult:
    """Final result returned to the pipeline after all healing attempts."""
    batch_id: str
    success: bool
    fixed_df: Optional[pd.DataFrame]
    original_issue_count: int
    final_issue_count: int
    attempts: list[HealAttempt] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    quarantined: bool = False

    def summary(self) -> str:
        status = "✅ Healed" if self.success else ("🚫 Quarantined" if self.quarantined else "❌ Failed")
        return (
            f"{status} | batch={self.batch_id} | "
            f"issues {self.original_issue_count} → {self.final_issue_count} | "
            f"{len(self.attempts)} attempt(s)"
        )

    def to_dict(self) -> dict:
        return {
            "batch_id": self.batch_id,
            "success": self.success,
            "quarantined": self.quarantined,
            "original_issue_count": self.original_issue_count,
            "final_issue_count": self.final_issue_count,
            "attempts": [a.to_dict() for a in self.attempts],
            "timestamp": self.timestamp,
        }


# ─────────────────────────────────────────────
#  Healer
# ─────────────────────────────────────────────

class Healer:
    """
    LLM-powered self-healing engine.

    Calls NVIDIA's Llama 3.1 70B to diagnose data issues and generate
    Python fix code, then safely executes and validates the fix.
    """

    NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
    MODEL           = "meta/llama-3.1-70b-instruct"

    SYSTEM_PROMPT = textwrap.dedent("""
        You are an expert data engineer specialising in data quality and pipeline repair.
        You will be given a report of data quality issues found in a pandas DataFrame,
        along with a sample of the problematic rows.

        Your job is to diagnose the root cause and write a Python fix.

        You MUST respond with ONLY a valid JSON object — no markdown, no explanation outside the JSON.
        The JSON must have exactly these three keys:

        {
          "diagnosis": "<one or two sentences explaining the root cause>",
          "fix_code": "<complete Python function as a single string>",
          "confidence": <float between 0.0 and 1.0>
        }

        Rules for fix_code:
        - Define a function named apply_fix(df) that accepts a pandas DataFrame and returns a fixed DataFrame.
        - Always start with: import pandas as pd; import numpy as np
        - For datetime: use "from datetime import datetime" (NOT "import datetime") to avoid AttributeError.
        - Do NOT modify the DataFrame in-place — always work on a copy: df = df.copy()
        - Check column existence before operating: if 'col' in df.columns:
        - For null numeric values: fill with median — df[col].fillna(df[col].median())
        - For null string values: fill with mode — df[col].fillna(df[col].mode()[0]) if not df[col].mode().empty else df[col]
        - For out-of-range values: clip to valid bounds — df[col] = df[col].clip(lower=min_val, upper=max_val)
        - For invalid emails: set to None — df[col] = df[col].apply(lambda x: x if pd.notnull(x) and "@" in str(x) and "." in str(x).split("@")[-1] else None)
        - For invalid category codes: replace with mode of valid values only — valid = df[col][df[col].isin(valid_list)]; df[col] = df[col].apply(lambda x: x if x in valid_list else (valid.mode()[0] if not valid.mode().empty else valid_list[0]))
        - For unparseable dates: use pd.to_datetime(df[col], errors="coerce") — this sets bad values to NaT automatically. Do NOT fill NaT with datetime.now() as it creates low-variance anomalies.
        - Keep the fix targeted — only fix the columns mentioned in the issues report.
        - The function must be syntactically valid Python that runs without errors.
    """).strip()

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        pipeline_cfg = config.get("pipeline", {})
        self.max_retries: int        = pipeline_cfg.get("max_heal_retries", 3)
        self.confidence_threshold: float = pipeline_cfg.get("confidence_threshold", 0.7)
        self.quarantine_on_failure: bool = pipeline_cfg.get("quarantine_on_failure", True)
        self.quarantine_dir: str     = "quarantine"
        self.config_path             = config_path

        api_key = os.getenv("NVIDIA_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "NVIDIA_API_KEY not found. Set it in your .env file or environment variables."
            )

        self.client = OpenAI(
            base_url=self.NVIDIA_BASE_URL,
            api_key=api_key,
        )

        # Re-use the same validator/detector for post-fix checks
        self.validator = Validator(config_path)
        self.detector  = AnomalyDetector(config_path)

        logger.info(
            f"Healer initialised. model={self.MODEL} | "
            f"max_retries={self.max_retries} | confidence_threshold={self.confidence_threshold}"
        )

    # ── Public entry point ──────────────────────

    def heal(
        self,
        df: pd.DataFrame,
        validation_report: Optional[ValidationReport] = None,
        anomaly_report: Optional[AnomalyReport] = None,
        batch_id: Optional[str] = None,
    ) -> HealResult:
        """
        Attempt to heal a DataFrame based on validation / anomaly reports.

        At least one of validation_report or anomaly_report must be provided.
        If neither is provided, the healer runs both checks itself.

        Returns a HealResult with the fixed DataFrame (or original if healing failed).
        """
        if batch_id is None:
            batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Run checks if not provided
        if validation_report is None:
            validation_report = self.validator.validate(df)
        if anomaly_report is None:
            anomaly_report = self.detector.detect(df)

        all_issues = (
            [i.to_dict() for i in validation_report.issues] +
            [a.to_dict() for a in anomaly_report.anomalies]
        )
        original_issue_count = len(all_issues)

        if original_issue_count == 0:
            logger.info("Healer called but no issues found — returning DataFrame unchanged.")
            return HealResult(
                batch_id=batch_id,
                success=True,
                fixed_df=df,
                original_issue_count=0,
                final_issue_count=0,
            )

        logger.info(
            f"Healer starting — batch={batch_id} | {original_issue_count} issue(s) to fix."
        )

        result = HealResult(
            batch_id=batch_id,
            success=False,
            fixed_df=df,
            original_issue_count=original_issue_count,
            final_issue_count=original_issue_count,
        )

        current_df    = df.copy()
        current_issues = all_issues
        previous_error = ""

        for attempt_num in range(1, self.max_retries + 1):
            logger.info(f"Heal attempt {attempt_num}/{self.max_retries}...")

            # ── Step 1: Call LLM ─────────────────
            llm_response = self._call_llm(current_df, current_issues, previous_error)

            if llm_response is None:
                attempt = HealAttempt(
                    attempt_number=attempt_num,
                    diagnosis="LLM call failed or returned invalid JSON.",
                    fix_code="",
                    confidence=0.0,
                    outcome="failed_execution",
                    error_message="LLM returned None or unparseable response.",
                    issues_before=len(current_issues),
                    issues_after=len(current_issues),
                )
                result.attempts.append(attempt)
                previous_error = attempt.error_message
                continue

            diagnosis  = llm_response.get("diagnosis", "No diagnosis provided.")
            fix_code   = llm_response.get("fix_code", "")
            confidence = float(llm_response.get("confidence", 0.0))

            logger.info(f"LLM diagnosis: {diagnosis}")
            logger.info(f"LLM confidence: {confidence:.2f}")

            # ── Step 2: Confidence gate ──────────
            if confidence < self.confidence_threshold:
                logger.warning(
                    f"Confidence {confidence:.2f} below threshold {self.confidence_threshold} "
                    f"— skipping this fix, flagging for human review."
                )
                attempt = HealAttempt(
                    attempt_number=attempt_num,
                    diagnosis=diagnosis,
                    fix_code=fix_code,
                    confidence=confidence,
                    outcome="low_confidence",
                    error_message=(
                        f"Confidence {confidence:.2f} < threshold {self.confidence_threshold}."
                    ),
                    issues_before=len(current_issues),
                    issues_after=len(current_issues),
                )
                result.attempts.append(attempt)
                previous_error = attempt.error_message
                continue

            # ── Step 3: Execute fix safely ───────
            fixed_df, exec_error = self._execute_fix(fix_code, current_df)

            if exec_error:
                logger.error(f"Fix execution failed: {exec_error}")
                attempt = HealAttempt(
                    attempt_number=attempt_num,
                    diagnosis=diagnosis,
                    fix_code=fix_code,
                    confidence=confidence,
                    outcome="failed_execution",
                    error_message=exec_error,
                    issues_before=len(current_issues),
                    issues_after=len(current_issues),
                )
                result.attempts.append(attempt)
                previous_error = f"The previous fix_code raised this error: {exec_error}"
                continue

            # ── Step 4: Re-validate ──────────────
            new_val_report  = self.validator.validate(fixed_df)
            new_anom_report = self.detector.detect(fixed_df)
            new_issues = (
                [i.to_dict() for i in new_val_report.issues] +
                [a.to_dict() for a in new_anom_report.anomalies]
            )
            issues_after = len(new_issues)

            logger.info(
                f"After fix: {len(current_issues)} issues → {issues_after} issues."
            )

            # ── Step 5: Accept or reject fix ────
            if issues_after < len(current_issues):
                # Fix improved the data — accept it
                attempt = HealAttempt(
                    attempt_number=attempt_num,
                    diagnosis=diagnosis,
                    fix_code=fix_code,
                    confidence=confidence,
                    outcome="success",
                    issues_before=len(current_issues),
                    issues_after=issues_after,
                )
                result.attempts.append(attempt)
                current_df     = fixed_df
                current_issues = new_issues

                if issues_after == 0:
                    logger.success(f"All issues resolved on attempt {attempt_num}.")
                    break
                else:
                    logger.info(
                        f"Partial fix — {issues_after} issue(s) remain. "
                        f"Continuing to next attempt."
                    )
                    previous_error = ""
            else:
                # Fix didn't help
                logger.warning("Fix did not reduce issue count — rejecting and retrying.")
                attempt = HealAttempt(
                    attempt_number=attempt_num,
                    diagnosis=diagnosis,
                    fix_code=fix_code,
                    confidence=confidence,
                    outcome="failed_validation",
                    error_message="Issue count did not decrease after applying fix.",
                    issues_before=len(current_issues),
                    issues_after=issues_after,
                )
                result.attempts.append(attempt)
                previous_error = (
                    "The previous fix did not reduce the issue count. "
                    "Try a different approach."
                )

        # ── Final outcome ────────────────────────
        result.final_issue_count = len(current_issues)

        if result.final_issue_count < original_issue_count:
            result.success  = True
            result.fixed_df = current_df
            logger.success(result.summary())
        else:
            result.success = False
            result.fixed_df = df  # return original, unmodified
            logger.error(f"Healing failed after {self.max_retries} attempt(s).")

            if self.quarantine_on_failure:
                self._quarantine(df, result)

        return result

    # ── LLM interaction ─────────────────────────

    def _call_llm(
        self,
        df: pd.DataFrame,
        issues: list[dict],
        previous_error: str = "",
    ) -> Optional[dict]:
        """
        Build the prompt and call NVIDIA Llama 3.1 70B.
        Returns parsed JSON dict or None on failure.
        """
        prompt = self._build_prompt(df, issues, previous_error)

        try:
            response = self.client.chat.completions.create(
                model=self.MODEL,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.2,      # low temp for deterministic code generation
                max_tokens=2048,
            )

            raw = response.choices[0].message.content.strip()
            logger.debug(f"Raw LLM response:\n{raw}")

            # Strip markdown fences if the model added them despite instructions
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            parsed = json.loads(raw)
            return parsed

        except json.JSONDecodeError as e:
            logger.error(f"LLM returned invalid JSON: {e}")
            return None
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            return None

    def _build_prompt(
        self,
        df: pd.DataFrame,
        issues: list[dict],
        previous_error: str = "",
    ) -> str:
        """
        Build a structured prompt including issue report, schema info,
        and a sample of bad rows for each issue.
        """
        # Collect affected row indices across all issues
        all_bad_rows = set()
        for issue in issues:
            all_bad_rows.update(issue.get("affected_rows", [])[:5])

        bad_sample = df.loc[list(all_bad_rows)].head(10) if all_bad_rows else df.head(5)

        prompt_parts = [
            "## Data Quality Issues Report",
            f"Total issues found: {len(issues)}",
            "",
            "### Issues:",
            json.dumps(issues, indent=2),
            "",
            "### DataFrame Schema:",
            str(df.dtypes.to_dict()),
            "",
            f"### Total rows in DataFrame: {len(df)}",
            "",
            "### Sample of affected rows (JSON):",
            bad_sample.to_json(orient="records", indent=2),
        ]

        if previous_error:
            prompt_parts += [
                "",
                "### Previous attempt feedback:",
                previous_error,
                "Please correct the approach and try again.",
            ]

        prompt_parts += [
            "",
            "Now respond with the JSON object as instructed.",
        ]

        return "\n".join(prompt_parts)

    # ── Safe code execution ─────────────────────

    def _execute_fix(
        self, fix_code: str, df: pd.DataFrame
    ) -> tuple[Optional[pd.DataFrame], str]:
        """
        Safely execute the LLM-generated fix_code against a copy of the DataFrame.

        Returns (fixed_df, error_message). If execution succeeds, error_message is "".
        The fix runs in a restricted namespace — no filesystem or network access.
        """
        if not fix_code or not fix_code.strip():
            return None, "fix_code is empty."

        # Allow full builtins so the LLM can use re, import, etc.
        # Security: we block filesystem and network modules post-import
        import builtins, re, math

        BLOCKED_MODULES = {"os", "sys", "subprocess", "socket", "shutil",
                           "pathlib", "http", "urllib", "requests", "open"}

        import datetime as _datetime_module

        safe_globals = {
            "__builtins__": builtins,   # full builtins — needed for imports inside fn
            "pd": pd,
            "np": __import__("numpy"),
            "re": re,
            "math": math,
            # Expose the datetime module so both styles work in LLM-generated code:
            #   import datetime              -> datetime.datetime.now()
            #   from datetime import datetime -> datetime.now()
            "datetime": _datetime_module,
        }
        local_ns: dict = {}

        # Patch __import__ to block dangerous modules
        original_import = builtins.__import__
        def safe_import(name, *args, **kwargs):
            if name.split(".")[0] in BLOCKED_MODULES:
                raise ImportError(f"Module '{name}' is not allowed in fix_code.")
            return original_import(name, *args, **kwargs)
        safe_globals["__builtins__"] = dict(vars(builtins))
        safe_globals["__builtins__"]["__import__"] = safe_import

        try:
            # Define the function in the safe namespace
            exec(fix_code, safe_globals, local_ns)  # noqa: S102

            if "apply_fix" not in local_ns:
                return None, "fix_code did not define a function named 'apply_fix'."

            apply_fix = local_ns["apply_fix"]

            # Run on a copy — never touch the original
            df_copy   = df.copy()
            fixed_df  = apply_fix(df_copy)

            if not isinstance(fixed_df, pd.DataFrame):
                return None, "apply_fix did not return a pandas DataFrame."

            if fixed_df.shape[0] != df.shape[0]:
                return None, (
                    f"apply_fix changed row count ({df.shape[0]} → {fixed_df.shape[0]}). "
                    "Row count must stay the same."
                )

            return fixed_df, ""

        except Exception:
            error = traceback.format_exc()
            logger.debug(f"Fix code that failed:\n{fix_code}")
            logger.debug(f"Full traceback:\n{error}")
            return None, error

    # ── Quarantine ──────────────────────────────

    def _quarantine(self, df: pd.DataFrame, result: HealResult) -> None:
        """Write unresolvable batches to the quarantine folder with full context."""
        os.makedirs(self.quarantine_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save the bad data
        data_path = os.path.join(self.quarantine_dir, f"{result.batch_id}_{ts}_data.csv")
        df.to_csv(data_path, index=False)

        # Save the heal result report
        report_path = os.path.join(self.quarantine_dir, f"{result.batch_id}_{ts}_report.json")
        with open(report_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        result.quarantined = True
        logger.warning(
            f"Batch quarantined → {data_path} | Report → {report_path}"
        )


# ─────────────────────────────────────────────
#  Quick test — run directly to see output
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uuid

    logger.info("Running Healer standalone test...")

    df      = pd.read_csv("data/sample.csv")
    healer  = Healer("config.yaml")

    result  = healer.heal(df, batch_id=str(uuid.uuid4())[:8])

    print("\n" + "=" * 60)
    print(result.summary())
    print("=" * 60)

    print("\nAttempt log:")
    for attempt in result.attempts:
        print(f"\n  Attempt {attempt.attempt_number}:")
        print(f"    Outcome    : {attempt.outcome}")
        print(f"    Confidence : {attempt.confidence:.2f}")
        print(f"    Diagnosis  : {attempt.diagnosis}")
        print(f"    Issues     : {attempt.issues_before} → {attempt.issues_after}")
        if attempt.error_message:
            print(f"    Error      : {attempt.error_message[:120]}")

    if result.success and result.fixed_df is not None:
        print(f"\n✅ Fixed DataFrame ({len(result.fixed_df)} rows):")
        print(result.fixed_df.to_string(index=False))