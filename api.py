"""
api.py
------
FastAPI REST API for the Self-Healing Data Pipeline.

Accepts CSV file uploads via HTTP, pushes them to the Redis Stream,
and returns a job ID the client can poll for results.

Endpoints:
    POST /run          Upload a CSV and queue a pipeline job
    GET  /status/{id}  Poll for job result by job ID
    GET  /results      List recent pipeline results
    GET  /health       Health check (Redis + pipeline ready)

Run locally:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Run via Docker:
    (handled by docker-compose.yml)
"""

import os
import uuid
import time
import shutil
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

try:
    import redis as redis_lib
except ImportError:
    raise ImportError("redis-py not installed. Run: pip install redis")

from producer import StreamProducer


# ─────────────────────────────────────────────
#  App setup
# ─────────────────────────────────────────────

app = FastAPI(
    title="Self-Healing Data Pipeline API",
    description=(
        "Upload a CSV file to automatically detect and fix data quality issues "
        "using LLM-powered diagnosis via NVIDIA Llama 3.1 70B."
    ),
    version="1.0.0",
)

# Allow Streamlit / browser requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "data")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Shared producer instance
_producer: Optional[StreamProducer] = None

def get_producer() -> StreamProducer:
    global _producer
    if _producer is None:
        _producer = StreamProducer()
    return _producer


# ─────────────────────────────────────────────
#  Response models
# ─────────────────────────────────────────────

class JobSubmittedResponse(BaseModel):
    job_id:   str
    message:  str
    stream:   str
    poll_url: str

class JobResultResponse(BaseModel):
    job_id:           str
    run_id:           str
    status:           str
    total_rows:       Optional[int]   = None
    validation_issues:Optional[int]   = None
    anomalies_found:  Optional[int]   = None
    issues_healed:    Optional[int]   = None
    issues_remaining: Optional[int]   = None
    heal_attempts:    Optional[int]   = None
    output_path:      Optional[str]   = None
    duration_seconds: Optional[float] = None
    error_message:    Optional[str]   = None

class HealthResponse(BaseModel):
    status: str
    redis:  str
    uptime_seconds: float

_start_time = time.time()


# ─────────────────────────────────────────────
#  Endpoints
# ─────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Check if the API and Redis are reachable."""
    try:
        get_producer().redis.ping()
        redis_status = "ok"
    except Exception as e:
        redis_status = f"error: {e}"

    overall = "ok" if redis_status == "ok" else "degraded"

    return HealthResponse(
        status=overall,
        redis=redis_status,
        uptime_seconds=round(time.time() - _start_time, 1),
    )


@app.post("/run", response_model=JobSubmittedResponse, tags=["Pipeline"])
async def run_pipeline(
    file:     UploadFile = File(..., description="CSV file to process"),
    heal:     bool       = Query(True,  description="Enable LLM healing"),
    run_id:   Optional[str] = Query(None, description="Custom run ID (optional)"),
):
    """
    Upload a CSV file and queue it for self-healing pipeline processing.

    Returns a job_id you can use to poll /status/{job_id} for the result.
    """
    # Validate file type
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail=f"Only CSV files are supported. Got: '{file.filename}'"
        )

    # Save upload to disk so the consumer can access it
    job_id    = run_id or str(uuid.uuid4())[:8]
    file_name = f"_upload_{job_id}.csv"
    file_path = os.path.join(UPLOAD_DIR, file_name)

    try:
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        logger.info(f"Uploaded file saved → {file_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")
    finally:
        file.file.close()

    # Push to Redis Stream
    try:
        submitted_id = get_producer().submit(
            file_path=file_path,
            heal=heal,
            run_id=job_id,
        )
    except Exception as e:
        # Clean up if we can't queue the job
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Failed to queue job: {e}")

    return JobSubmittedResponse(
        job_id=submitted_id,
        message="Job queued successfully. Poll /status/{job_id} for the result.",
        stream=os.getenv("JOBS_STREAM", "pipeline:jobs"),
        poll_url=f"/status/{submitted_id}",
    )


@app.get("/status/{job_id}", response_model=JobResultResponse, tags=["Pipeline"])
def get_status(
    job_id:  str,
    timeout: int = Query(0, description="Seconds to wait for result (0 = non-blocking)"),
):
    """
    Check the status of a submitted pipeline job.

    - Pass timeout=0 (default) for an immediate response (returns 202 if still processing).
    - Pass timeout=N to wait up to N seconds for the result (max 300).
    """
    timeout = min(max(timeout, 0), 300)

    producer = get_producer()
    result   = None

    if timeout > 0:
        result = producer.wait_for_result(job_id, timeout_sec=timeout, poll_ms=1000)
    else:
        # Non-blocking: scan the last 500 results
        messages = producer.redis.xrevrange(
            os.getenv("RESULTS_STREAM", "pipeline:results"),
            count=500,
        )
        for _, fields in messages:
            if fields.get("job_id") == job_id:
                result = fields
                break

    if result is None:
        # Job is still queued or being processed
        return JSONResponse(
            status_code=202,
            content={
                "job_id":  job_id,
                "status":  "pending",
                "message": "Job is queued or currently processing. Try again shortly.",
            },
        )

    def _int(v):
        try: return int(v)
        except: return None

    def _float(v):
        try: return float(v)
        except: return None

    return JobResultResponse(
        job_id=result.get("job_id", job_id),
        run_id=result.get("run_id", job_id),
        status=result.get("status", "unknown"),
        total_rows=_int(result.get("total_rows")),
        validation_issues=_int(result.get("validation_issues")),
        anomalies_found=_int(result.get("anomalies_found")),
        issues_healed=_int(result.get("issues_healed")),
        issues_remaining=_int(result.get("issues_remaining")),
        heal_attempts=_int(result.get("heal_attempts")),
        output_path=result.get("output_path"),
        duration_seconds=_float(result.get("duration_seconds")),
        error_message=result.get("error_message") or None,
    )


@app.get("/results", tags=["Pipeline"])
def list_results(limit: int = Query(10, ge=1, le=100)):
    """Return the last N pipeline results from the results stream."""
    producer = get_producer()
    messages = producer.redis.xrevrange(
        os.getenv("RESULTS_STREAM", "pipeline:results"),
        count=limit,
    )
    results = []
    for msg_id, fields in messages:
        results.append({"message_id": msg_id, **fields})

    return {"count": len(results), "results": results}


@app.get("/download/{run_id}", tags=["Pipeline"])
def download_result(run_id: str):
    """Download the cleaned output CSV for a completed run."""
    output_dir  = os.getenv("OUTPUT_DIR", "output")
    output_path = os.path.join(output_dir, f"{run_id}_clean.csv")

    if not os.path.exists(output_path):
        raise HTTPException(
            status_code=404,
            detail=f"Output file not found for run_id='{run_id}'. "
                   f"The job may still be processing or may have failed."
        )

    return FileResponse(
        path=output_path,
        media_type="text/csv",
        filename=f"{run_id}_clean.csv",
    )


# ─────────────────────────────────────────────
#  Run directly
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=int(os.getenv("API_PORT", 8000)),
        reload=False,
    )