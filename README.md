# 🔧 Self-Healing Data Pipeline

> A Python pipeline that detects broken data and fixes itself — automatically — using LLM-powered diagnosis via NVIDIA's free-tier API.

---

## 📌 Overview

The Self-Healing Data Pipeline ingests raw data from various sources (CSV, REST APIs, databases), validates it against defined rules, detects anomalies, and — when issues are found — calls a large language model to diagnose the problem and generate a Python fix on the fly. Fixed data is re-validated before being passed downstream. Every repair attempt is logged for full auditability.

```
Raw Data → Ingest → Validate → Detect Anomalies → Transform → Output
                       ↓               ↓
                    [issues]        [anomaly]
                       └──────┬──────┘
                              ↓
                       LLM Healer (NVIDIA Llama 3.1 70B)
                              ↓
                       Auto-Fix → Re-Validate → Retry (max 3×)
                              ↓
                       Quarantine if unresolved
```

---

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| Language | Python 3.10+ |
| LLM | `meta/llama-3.1-70b-instruct` via NVIDIA API |
| LLM Client | `openai` SDK (NVIDIA-compatible) |
| Data | `pandas` |
| Validation | Custom rule engine (`great_expectations` optional) |
| Storage | SQLite / PostgreSQL |
| Logging | `loguru` |
| Config | `PyYAML` |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/self-healing-pipeline.git
cd self-healing-pipeline
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Get your NVIDIA API key

Sign up for free at [build.nvidia.com](https://build.nvidia.com) — no credit card required.

### 4. Set environment variables

```bash
export NVIDIA_API_KEY="your_api_key_here"
```

Or create a `.env` file:

```
NVIDIA_API_KEY=your_api_key_here
```

### 5. Run the pipeline

```bash
python pipeline.py --input data/sample.csv
```

---

## 📁 Project Structure

```
self-healing-pipeline/
├── pipeline.py          # Main orchestrator — run this
├── ingestion.py         # Load from CSV, API, or DB
├── validator.py         # Schema + rule-based validation
├── anomaly.py           # Statistical & frequency anomaly detection
├── healer.py            # LLM diagnosis and fix-code generation
├── logger.py            # Audit log writer
├── config.yaml          # Validation rules and thresholds
├── data/
│   └── sample.csv       # Sample dataset for testing
├── quarantine/          # Batches that could not be auto-healed
├── tests/
│   ├── test_validator.py
│   ├── test_anomaly.py
│   └── test_healer.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Configuration

Define your validation schema in `config.yaml`:

```yaml
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
      allowed_values: [IN, US, UK, DE, FR]

pipeline:
  max_heal_retries: 3
  quarantine_on_failure: true
  confidence_threshold: 0.7   # LLM fixes below this score go to human review
```

---

## 🧠 How the LLM Healer Works

When validation or anomaly detection flags an issue, the healer module:

1. **Builds a structured prompt** with the error report, column metadata, and a sample of bad rows
2. **Calls Llama 3.1 70B** on the NVIDIA API and requests a JSON response containing:
   - `diagnosis` — one-sentence explanation of the issue
   - `fix_code` — a Python function `apply_fix(df)` that returns a corrected DataFrame
   - `confidence` — a 0–1 score for how certain the model is
3. **Executes the fix** against a copy of the data (original is never mutated until the fix is validated)
4. **Re-runs validation** — if the score improves, the fix is accepted; otherwise it retries with a refined prompt
5. **Quarantines** the batch after 3 failed attempts and sends an alert

```python
# healer.py — simplified
def heal(df, error_report):
    response = client.chat.completions.create(
        model="meta/llama-3.1-70b-instruct",
        messages=[{"role": "user", "content": build_prompt(df, error_report)}],
        response_format={"type": "json_object"}
    )
    result = json.loads(response.choices[0].message.content)
    return result  # { diagnosis, fix_code, confidence }
```

---

## 📊 Audit Log

Every repair attempt is written to a `repairs` table:

| Column | Description |
|---|---|
| `timestamp` | When the repair ran |
| `batch_id` | Unique ID for the data batch |
| `issue_type` | `null_values`, `type_mismatch`, `outlier`, etc. |
| `diagnosis` | LLM's explanation |
| `fix_applied` | The generated Python fix |
| `confidence` | LLM confidence score |
| `outcome` | `success`, `failed`, `quarantined` |

---

## 🔒 Safety Considerations

- LLM-generated fix code runs against a **DataFrame copy only** — the original is never touched until the fix passes re-validation
- Fixes with `confidence < threshold` (configurable) are **flagged for human review** rather than auto-applied
- All `exec` calls are scoped to a restricted namespace — no access to filesystem or network
- Quarantined batches are **never silently dropped** — they are preserved with their full error context

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 🗺️ Roadmap

- [ ] Streamlit dashboard for pipeline health and repair history
- [ ] Kafka integration for streaming / real-time healing
- [ ] Human-in-the-loop approval UI for low-confidence fixes
- [ ] Domain-specific prompt templates (finance, IoT, e-commerce)
- [ ] Slack / email alerting for quarantined batches

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙌 Acknowledgements

- [NVIDIA NIM](https://build.nvidia.com) for free-tier LLM API access
- [Meta Llama 3.1](https://ai.meta.com/blog/meta-llama-3-1/) for the underlying model
- Inspired by the growing field of self-healing data systems