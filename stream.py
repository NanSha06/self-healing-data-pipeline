"""
stream.py
---------
Redis Streams consumer for the Self-Healing Data Pipeline.

Continuously reads pipeline jobs from a Redis Stream, runs the full
self-healing pipeline for each job, and publishes results back to a
results stream.

Message format (produced by producer.py or api.py):
    {
        "job_id"   : "abc123",
        "file_path": "data/upload_abc123.csv",
        "heal"     : "true",
        "run_id"   : "abc123"          # optional
    }

Result format (published to pipeline:results):
    {
        "job_id"          : "abc123",
        "run_id"          : "abc123",
        "status"          : "success",
        "total_rows"      : "20",
        "issues_healed"   : "7",
        "issues_remaining": "0",
        "output_path"     : "output/abc123_clean.csv",
        "duration_seconds": "12.4",
        "error_message"   : ""
    }

Run:
    python stream.py
    python stream.py --stream pipeline:jobs --group pipeline-workers
"""

import os
import sys
import time
import signal
import argparse
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

# ── Redis connection ─────────────────────────
try:
    import redis
except ImportError:
    print("redis-py not installed. Run: pip install redis")
    sys.exit(1)

# ── Pipeline ─────────────────────────────────
from pipeline import Pipeline


# ─────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────

REDIS_HOST     = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT     = int(os.getenv("REDIS_PORT", 6379))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
REDIS_DB       = int(os.getenv("REDIS_DB", 0))

JOBS_STREAM    = os.getenv("JOBS_STREAM",    "pipeline:jobs")
RESULTS_STREAM = os.getenv("RESULTS_STREAM", "pipeline:results")
CONSUMER_GROUP = os.getenv("CONSUMER_GROUP", "pipeline-workers")
CONSUMER_NAME  = os.getenv("CONSUMER_NAME",  f"worker-{os.getpid()}")

CONFIG_PATH    = os.getenv("CONFIG_PATH", "config.yaml")

# How long to block waiting for new messages (ms). 0 = block forever.
BLOCK_MS       = int(os.getenv("BLOCK_MS", 0))   # 0 = non-blocking poll

# Max messages to fetch per poll
BATCH_SIZE     = int(os.getenv("BATCH_SIZE", 1))


# ─────────────────────────────────────────────
#  Consumer
# ─────────────────────────────────────────────

class StreamConsumer:
    """
    Reads jobs from a Redis Stream using consumer groups (at-least-once delivery).

    Consumer groups ensure:
      - Each job is processed by exactly one worker (even with multiple consumers)
      - If a worker crashes mid-job, the message stays pending and can be reclaimed
      - Jobs are acknowledged (XACK) only after successful processing
    """

    def __init__(
        self,
        jobs_stream:    str = JOBS_STREAM,
        results_stream: str = RESULTS_STREAM,
        consumer_group: str = CONSUMER_GROUP,
        consumer_name:  str = CONSUMER_NAME,
        config_path:    str = CONFIG_PATH,
    ):
        self.jobs_stream    = jobs_stream
        self.results_stream = results_stream
        self.consumer_group = consumer_group
        self.consumer_name  = consumer_name
        self.config_path    = config_path
        self.running        = False

        # Connect to Redis
        self.redis = self._connect()

        # Ensure stream and consumer group exist
        self._ensure_group()

        # Load pipeline (reused across all jobs — no re-init overhead)
        logger.info(f"Loading pipeline from {config_path}...")
        self.pipeline = Pipeline(config_path)

        logger.info(
            f"Consumer ready — stream={jobs_stream} group={consumer_group} "
            f"name={consumer_name}"
        )

    # ── Setup ───────────────────────────────────

    def _connect(self) -> redis.Redis:
        """Create and verify a Redis connection with keepalive to prevent socket timeouts."""
        client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD,
            db=REDIS_DB,
            decode_responses=True,
            # Keep the connection alive — prevents TimeoutError on idle blocking reads
            socket_keepalive=True,
            socket_keepalive_options={},
            # No hard socket timeout — the BLOCK parameter in xreadgroup handles timing
            socket_timeout=None,
            socket_connect_timeout=10,
            # Health check keeps connection warm between polls
            health_check_interval=30,
        )
        try:
            client.ping()
            logger.success(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
        except redis.exceptions.ConnectionError as e:
            logger.error(
                f"Cannot connect to Redis at {REDIS_HOST}:{REDIS_PORT}. "
                f"Is Redis running? Error: {e}"
            )
            sys.exit(1)
        return client

    def _ensure_group(self) -> None:
        """Create the stream and consumer group if they don't exist."""
        try:
            # MKSTREAM creates the stream if it doesn't exist
            self.redis.xgroup_create(
                self.jobs_stream,
                self.consumer_group,
                id="0",          # start from beginning
                mkstream=True,
            )
            logger.info(
                f"Consumer group '{self.consumer_group}' created "
                f"on stream '{self.jobs_stream}'"
            )
        except redis.exceptions.ResponseError as e:
            if "BUSYGROUP" in str(e):
                logger.debug(f"Consumer group '{self.consumer_group}' already exists.")
            else:
                raise

    # ── Main loop ───────────────────────────────

    def start(self) -> None:
        """
        Start the consumer loop. Blocks until stop() is called or
        a shutdown signal (SIGINT / SIGTERM) is received.
        """
        self.running = True

        # Graceful shutdown on Ctrl+C or Docker stop
        signal.signal(signal.SIGINT,  self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        logger.info(f"Consumer started. Waiting for jobs on '{self.jobs_stream}'...")
        logger.info("Press Ctrl+C to stop.")

        # First: process any pending messages from a previous crashed worker
        self._process_pending()

        # Then: listen for new messages — pure poll loop, no blocking reads
        while self.running:
            try:
                # BLOCK=0 means non-blocking: returns immediately with whatever is
                # in the stream right now. No socket held open between polls.
                messages = self.redis.xreadgroup(
                    groupname=self.consumer_group,
                    consumername=self.consumer_name,
                    streams={self.jobs_stream: ">"},
                    count=BATCH_SIZE,
                    block=0,                           # always 0 — never block
                )

                if not messages:
                    time.sleep(1)                      # idle — wait 1s before next poll
                    continue

                for stream_name, entries in messages:
                    for message_id, fields in entries:
                        self._process_message(message_id, fields)

            except (
                redis.exceptions.ConnectionError,
                redis.exceptions.TimeoutError,
                redis.exceptions.ResponseError,
                TimeoutError,                          # built-in Python socket timeout
                OSError,                               # underlying socket errors
            ):
                logger.warning("Redis connection interrupted — reconnecting in 5s...")
                time.sleep(5)
                try:
                    self.redis = self._connect()
                    self._ensure_group()
                except Exception:
                    pass

            except Exception as e:
                logger.warning(f"Consumer loop error ({type(e).__name__}): {e} — retrying in 2s...")
                time.sleep(2)

        logger.info("Consumer stopped.")

    def stop(self) -> None:
        self.running = False

    def _handle_shutdown(self, signum, frame) -> None:
        logger.info(f"Shutdown signal received ({signum}). Stopping after current job...")
        self.stop()

    # ── Message processing ───────────────────────

    def _process_message(self, message_id: str, fields: dict) -> None:
        """
        Process a single job message from the stream.

        Acknowledges the message only if processing succeeds or produces
        a deterministic result (even a failed pipeline run is acknowledged,
        since retrying won't fix a bad input file).
        """
        job_id    = fields.get("job_id",    message_id)
        file_path = fields.get("file_path", "")
        heal      = fields.get("heal", "true").lower() == "true"
        run_id    = fields.get("run_id") or job_id[:8]

        logger.info(f"Processing job — id={job_id} file={file_path} heal={heal}")

        if not file_path:
            logger.error(f"Job {job_id} has no file_path — skipping.")
            self._ack(message_id)
            return

        if not os.path.exists(file_path):
            logger.error(f"File not found for job {job_id}: '{file_path}'")
            self._publish_result(job_id, run_id, {
                "status":        "failed",
                "error_message": f"File not found: {file_path}",
            })
            self._ack(message_id)
            return

        # ── Run the pipeline ─────────────────────
        start = time.time()
        try:
            result = self.pipeline.run(
                input_source=file_path,
                heal=heal,
                run_id=run_id,
            )

            self._publish_result(job_id, run_id, {
                "status":           result.status,
                "total_rows":       str(result.total_rows),
                "validation_issues":str(result.validation_issues),
                "anomalies_found":  str(result.anomalies_detected),
                "issues_healed":    str(result.issues_healed),
                "issues_remaining": str(result.issues_remaining),
                "heal_attempts":    str(result.heal_attempts),
                "output_path":      result.output_path,
                "duration_seconds": str(result.duration_seconds),
                "error_message":    result.error_message,
            })

            logger.success(
                f"Job {job_id} complete — status={result.status} "
                f"duration={result.duration_seconds}s"
            )

        except Exception as e:
            logger.exception(f"Pipeline raised an exception for job {job_id}: {e}")
            self._publish_result(job_id, run_id, {
                "status":        "failed",
                "error_message": str(e),
            })

        finally:
            # Always ACK — even failed jobs, so they don't loop forever
            self._ack(message_id)

            # Clean up temp upload files after processing
            if file_path.startswith("data/_upload_") and os.path.exists(file_path):
                os.remove(file_path)
                logger.debug(f"Cleaned up temp file: {file_path}")

    def _process_pending(self) -> None:
        """
        Re-process any messages that were delivered to a previous (crashed)
        worker but never acknowledged. Uses XAUTOCLAIM to reclaim them.
        """
        try:
            # Reclaim messages idle for more than 60 seconds
            result = self.redis.xautoclaim(
                self.jobs_stream,
                self.consumer_group,
                self.consumer_name,
                min_idle_time=60000,    # 60 seconds in ms
                start_id="0-0",
                count=10,
            )
            # xautoclaim returns (next_id, messages, deleted_ids)
            messages = result[1] if isinstance(result, (list, tuple)) else []

            if messages:
                logger.info(f"Reclaimed {len(messages)} pending message(s) from crashed workers.")
                for message_id, fields in messages:
                    self._process_message(message_id, fields)
            else:
                logger.debug("No pending messages to reclaim.")

        except Exception as e:
            logger.debug(f"Could not check pending messages: {e}")

    # ── Redis helpers ────────────────────────────

    def _ack(self, message_id: str) -> None:
        """Acknowledge a message so it won't be redelivered."""
        self.redis.xack(self.jobs_stream, self.consumer_group, message_id)
        logger.debug(f"ACK'd message {message_id}")

    def _publish_result(self, job_id: str, run_id: str, data: dict) -> None:
        """Publish a job result to the results stream."""
        payload = {"job_id": job_id, "run_id": run_id, **data}
        self.redis.xadd(
            self.results_stream,
            payload,
            maxlen=1000,    # keep last 1000 results
        )
        logger.debug(f"Published result for job {job_id} to '{self.results_stream}'")


# ─────────────────────────────────────────────
#  CLI entry point
# ─────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-Healing Pipeline — Redis Stream Consumer")
    parser.add_argument("--stream",  default=JOBS_STREAM,    help="Redis stream name for jobs")
    parser.add_argument("--group",   default=CONSUMER_GROUP, help="Consumer group name")
    parser.add_argument("--name",    default=CONSUMER_NAME,  help="This consumer's name")
    parser.add_argument("--config",  default=CONFIG_PATH,    help="Path to config.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args     = _parse_args()
    consumer = StreamConsumer(
        jobs_stream=args.stream,
        consumer_group=args.group,
        consumer_name=args.name,
        config_path=args.config,
    )
    consumer.start()