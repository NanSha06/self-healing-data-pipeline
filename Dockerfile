# ─────────────────────────────────────────────
# Self-Healing Data Pipeline — Dockerfile
# ─────────────────────────────────────────────
# Builds a single image used by both services:
#   - api      : FastAPI REST endpoint  (port 8000)
#   - consumer : Redis Stream consumer  (no port)
#   - streamlit: Streamlit dashboard    (port 8501)
#
# Build:
#   docker build -t self-healing-pipeline .
#
# Run (via docker-compose — recommended):
#   docker-compose up
# ─────────────────────────────────────────────

FROM python:3.11-slim

# Metadata
LABEL maintainer="self-healing-pipeline"
LABEL description="Self-Healing Data Pipeline with NVIDIA LLM"

# ── System dependencies ──────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ────────────────────────
WORKDIR /app

# ── Install Python dependencies ──────────────
# Copy requirements first for Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    redis \
    streamlit \
    python-multipart

# ── Copy application code ────────────────────
COPY pipeline.py   .
COPY ingestion.py  .
COPY validator.py  .
COPY anomaly.py    .
COPY healer.py     .
COPY logger.py     .
COPY stream.py     .
COPY producer.py   .
COPY api.py        .
COPY app.py        .
COPY config.yaml   .

# ── Create runtime directories ───────────────
RUN mkdir -p data output quarantine logs

# ── Non-root user for security ───────────────
# Create home directory explicitly so Streamlit can write its machine ID file
RUN groupadd -r pipeline && \
    useradd -r -g pipeline -m -d /home/pipeline pipeline && \
    mkdir -p /home/pipeline/.streamlit && \
    chown -R pipeline:pipeline /home/pipeline && \
    chown -R pipeline:pipeline /app
USER pipeline

# ── Streamlit config — disable telemetry, set headless mode ──
RUN echo '[general]' > /home/pipeline/.streamlit/credentials.toml && \
    echo 'email = ""' >> /home/pipeline/.streamlit/credentials.toml && \
    echo '[browser]' >> /home/pipeline/.streamlit/config.toml && \
    echo 'gatherUsageStats = false' >> /home/pipeline/.streamlit/config.toml && \
    echo 'serverAddress = "0.0.0.0"' >> /home/pipeline/.streamlit/config.toml

# ── Expose ports ─────────────────────────────
EXPOSE 8000
EXPOSE 8501

# ── Health check ─────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Default command (overridden by docker-compose) ──
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]