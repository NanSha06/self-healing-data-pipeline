"""
producer.py
-----------
Pushes pipeline jobs into the Redis Stream for the consumer to pick up.

Can be used as:
  - A CLI tool to submit a CSV file manually
  - An importable module called from api.py or any other trigger

Usage (CLI):
    python producer.py --file data/sample.csv
    python producer.py --file data/sample.csv --no-heal
    python producer.py --file data/sample.csv --run-id my-run-001
    python producer.py --status                    # check recent results
    python producer.py --pending                   # show unprocessed jobs
"""

import os
import sys
import uuid
import time
import argparse
import pandas as pd
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

try:
    import redis
except ImportError:
    print("redis-py not installed. Run: pip install redis")
    sys.exit(1)


# ─────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────

REDIS_HOST     = os.getenv("REDIS_HOST",     "localhost")
REDIS_PORT     = int(os.getenv("REDIS_PORT", 6379))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
REDIS_DB       = int(os.getenv("REDIS_DB",   0))

JOBS_STREAM    = os.getenv("JOBS_STREAM",    "pipeline:jobs")
RESULTS_STREAM = os.getenv("RESULTS_STREAM", "pipeline:results")
CONSUMER_GROUP = os.getenv("CONSUMER_GROUP", "pipeline-workers")


# ─────────────────────────────────────────────
#  Producer
# ─────────────────────────────────────────────

class StreamProducer:
    """
    Publishes pipeline job messages to a Redis Stream.
    """

    def __init__(self):
        self.redis = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD,
            db=REDIS_DB,
            decode_responses=True,
        )
        try:
            self.redis.ping()
            logger.success(f"Producer connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
        except redis.exceptions.ConnectionError as e:
            logger.error(f"Cannot connect to Redis: {e}")
            sys.exit(1)

    # ── Publish a job ────────────────────────────

    def submit(
        self,
        file_path: str,
        heal:      bool = True,
        run_id:    str  = None,
    ) -> str:
        """
        Submit a CSV file for pipeline processing.

        Args:
            file_path : path to the CSV file (must be accessible by the consumer)
            heal      : whether to run LLM healing (default True)
            run_id    : optional custom run ID

        Returns:
            job_id — the Redis stream message ID
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: '{file_path}'")

        job_id = run_id or str(uuid.uuid4())[:8]

        message = {
            "job_id":    job_id,
            "file_path": file_path,
            "heal":      "true" if heal else "false",
            "run_id":    job_id,
            "submitted": str(time.time()),
        }

        msg_id = self.redis.xadd(JOBS_STREAM, message)

        logger.success(
            f"Job submitted — job_id={job_id} file={file_path} "
            f"heal={heal} msg_id={msg_id}"
        )
        return job_id

    # ── Wait for result ──────────────────────────

    def wait_for_result(
        self,
        job_id:      str,
        timeout_sec: int = 300,
        poll_ms:     int = 2000,
    ) -> dict | None:
        """
        Poll the results stream until the job completes or times out.

        Args:
            job_id      : the job ID returned by submit()
            timeout_sec : max seconds to wait (default 5 minutes)
            poll_ms     : polling interval in milliseconds

        Returns:
            Result dict, or None if timed out.
        """
        deadline = time.time() + timeout_sec
        last_id  = "0-0"

        logger.info(f"Waiting for result of job {job_id} (timeout={timeout_sec}s)...")

        while time.time() < deadline:
            messages = self.redis.xread(
                {RESULTS_STREAM: last_id},
                count=100,
                block=poll_ms,
            )
            if messages:
                for _, entries in messages:
                    for msg_id, fields in entries:
                        last_id = msg_id
                        if fields.get("job_id") == job_id:
                            logger.success(
                                f"Result received for job {job_id}: "
                                f"status={fields.get('status')}"
                            )
                            return fields

        logger.warning(f"Timeout waiting for job {job_id} after {timeout_sec}s")
        return None

    # ── Info helpers ─────────────────────────────

    def get_recent_results(self, count: int = 10) -> list[dict]:
        """Return the last N results from the results stream."""
        messages = self.redis.xrevrange(RESULTS_STREAM, count=count)
        return [fields for _, fields in messages]

    def get_stream_info(self) -> dict:
        """Return length and group info for the jobs stream."""
        try:
            length = self.redis.xlen(JOBS_STREAM)
            groups = self.redis.xinfo_groups(JOBS_STREAM)
            pending = sum(g.get("pending", 0) for g in groups)
            return {
                "stream":        JOBS_STREAM,
                "total_messages": length,
                "consumer_groups": len(groups),
                "pending_messages": pending,
                "groups": [
                    {
                        "name":       g.get("name"),
                        "consumers":  g.get("consumers"),
                        "pending":    g.get("pending"),
                        "last_delivered": g.get("last-delivered-id"),
                    }
                    for g in groups
                ],
            }
        except redis.exceptions.ResponseError:
            return {"stream": JOBS_STREAM, "status": "stream does not exist yet"}

    def print_status(self) -> None:
        """Print stream status and recent results to the console."""
        info = self.get_stream_info()
        print("\n── Stream Info ─────────────────────────────")
        for k, v in info.items():
            if k != "groups":
                print(f"  {k:<22}: {v}")
        if info.get("groups"):
            print("  Consumer groups:")
            for g in info["groups"]:
                print(
                    f"    • {g['name']} | consumers={g['consumers']} "
                    f"| pending={g['pending']}"
                )

        results = self.get_recent_results(10)
        if results:
            print("\n── Recent results ──────────────────────────")
            for r in results:
                status_icon = {
                    "success":     "✅",
                    "partial":     "⚠️ ",
                    "failed":      "❌",
                    "quarantined": "🚫",
                }.get(r.get("status", ""), "❓")
                print(
                    f"  {status_icon} job={r.get('job_id','?'):<10} "
                    f"status={r.get('status','?'):<12} "
                    f"healed={r.get('issues_healed','?'):<4} "
                    f"remaining={r.get('issues_remaining','?')}"
                )
        else:
            print("\n  No results yet.")
        print()


# ─────────────────────────────────────────────
#  CLI entry point
# ─────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Self-Healing Pipeline — Redis Stream Producer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python producer.py --file data/sample.csv
  python producer.py --file data/sample.csv --no-heal
  python producer.py --file data/sample.csv --wait
  python producer.py --status
        """,
    )
    parser.add_argument("--file",    "-f", help="Path to CSV file to process")
    parser.add_argument("--no-heal",       action="store_true", help="Skip LLM healing")
    parser.add_argument("--run-id",        help="Custom run ID (optional)")
    parser.add_argument("--wait",          action="store_true",
                        help="Block until result is received (default: fire and forget)")
    parser.add_argument("--timeout",       type=int, default=300,
                        help="Timeout in seconds when using --wait (default: 300)")
    parser.add_argument("--status",        action="store_true",
                        help="Show stream status and recent results then exit")
    return parser.parse_args()


def main() -> None:
    args     = _parse_args()
    producer = StreamProducer()

    if args.status:
        producer.print_status()
        return

    if not args.file:
        print("Error: --file is required. Use --help for usage.")
        sys.exit(1)

    job_id = producer.submit(
        file_path=args.file,
        heal=not args.no_heal,
        run_id=args.run_id,
    )

    print(f"\nJob submitted: {job_id}")
    print(f"Stream:        {JOBS_STREAM}")

    if args.wait:
        result = producer.wait_for_result(job_id, timeout_sec=args.timeout)
        if result:
            print("\n── Result ──────────────────────────────────")
            for k, v in result.items():
                print(f"  {k:<22}: {v}")
        else:
            print(f"\nNo result after {args.timeout}s — the consumer may still be running.")
            sys.exit(1)
    else:
        print("The consumer will process this job asynchronously.")
        print(f"Check results with: python producer.py --status")


if __name__ == "__main__":
    main()