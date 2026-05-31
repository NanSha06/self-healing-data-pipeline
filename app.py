"""
app.py
------
Streamlit web application for the Self-Healing Data Pipeline.

Run with:
    streamlit run app.py
"""

import os
import sys
import uuid
import sqlite3
import pandas as pd
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ── Page config (must be first Streamlit call) ──
st.set_page_config(
    page_title="Self-Healing Data Pipeline",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@400;600;700;800&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}
code, pre, .stCode {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Background */
.stApp {
    background: #0a0e1a;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0d1220 !important;
    border-right: 1px solid #1e2a45;
}
[data-testid="stSidebar"] * {
    color: #c8d6f0 !important;
}

/* Main content text */
h1, h2, h3, h4, h5, h6 { color: #e8f0ff !important; }
p, li, span, label       { color: #a0b4d0 !important; }

/* Metric cards */
[data-testid="stMetric"] {
    background: #111827;
    border: 1px solid #1e2a45;
    border-radius: 12px;
    padding: 1rem 1.25rem !important;
}
[data-testid="stMetricLabel"]  { color: #6b7fa3 !important; font-size: 0.75rem !important; letter-spacing: 0.08em; text-transform: uppercase; }
[data-testid="stMetricValue"]  { color: #e8f0ff !important; font-size: 1.8rem !important; font-weight: 800; }
[data-testid="stMetricDelta"]  { font-size: 0.8rem !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    color: #fff !important;
    border: none;
    border-radius: 8px;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    letter-spacing: 0.04em;
    padding: 0.55rem 1.4rem;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #3b82f6, #2563eb);
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(37,99,235,0.4);
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #111827;
    border: 2px dashed #1e3a5f;
    border-radius: 12px;
    padding: 1rem;
}
[data-testid="stFileUploader"]:hover {
    border-color: #2563eb;
}

/* DataFrames */
[data-testid="stDataFrame"] {
    border: 1px solid #1e2a45 !important;
    border-radius: 10px;
    overflow: hidden;
}

/* Expander */
[data-testid="stExpander"] {
    background: #111827;
    border: 1px solid #1e2a45 !important;
    border-radius: 10px;
}

/* Alerts */
.stSuccess { background: #0d2818 !important; border-left: 4px solid #22c55e !important; color: #86efac !important; border-radius: 8px; }
.stWarning { background: #1a1500 !important; border-left: 4px solid #eab308 !important; color: #fde047 !important; border-radius: 8px; }
.stError   { background: #1a0a0a !important; border-left: 4px solid #ef4444 !important; color: #fca5a5 !important; border-radius: 8px; }
.stInfo    { background: #0a1628 !important; border-left: 4px solid #3b82f6 !important; color: #93c5fd !important; border-radius: 8px; }

/* Tabs */
[data-testid="stTab"] button {
    color: #6b7fa3 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
}
[data-testid="stTab"] button[aria-selected="true"] {
    color: #3b82f6 !important;
    border-bottom: 2px solid #3b82f6 !important;
}

/* Divider */
hr { border-color: #1e2a45 !important; }

/* Selectbox / text input */
[data-testid="stSelectbox"] > div,
[data-testid="stTextInput"] > div > div > input {
    background: #111827 !important;
    border: 1px solid #1e2a45 !important;
    color: #e8f0ff !important;
    border-radius: 8px;
}

/* Status badge helper classes */
.badge-success  { background:#0d2818; color:#22c55e; border:1px solid #16a34a; padding:2px 10px; border-radius:20px; font-size:0.75rem; font-weight:700; }
.badge-partial  { background:#1a1500; color:#eab308; border:1px solid #ca8a04; padding:2px 10px; border-radius:20px; font-size:0.75rem; font-weight:700; }
.badge-failed   { background:#1a0a0a; color:#ef4444; border:1px solid #dc2626; padding:2px 10px; border-radius:20px; font-size:0.75rem; font-weight:700; }
.badge-quarantined { background:#1a0a1a; color:#a855f7; border:1px solid #9333ea; padding:2px 10px; border-radius:20px; font-size:0.75rem; font-weight:700; }

/* Scrollbar */
::-webkit-scrollbar       { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0a0e1a; }
::-webkit-scrollbar-thumb { background: #1e2a45; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Import pipeline modules (with error guard)
# ─────────────────────────────────────────────

@st.cache_resource
def load_pipeline(config_path: str = "config.yaml"):
    from pipeline import Pipeline
    return Pipeline(config_path)


def check_env() -> bool:
    return bool(os.getenv("NVIDIA_API_KEY"))


# ─────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🔧 Pipeline Control")
    st.markdown("---")

    # API key status
    if check_env():
        st.success("NVIDIA API key ✓")
    else:
        st.error("NVIDIA API key missing")
        st.caption("Add NVIDIA_API_KEY to your .env file")

    st.markdown("---")

    # Config path
    config_path = st.text_input("Config path", value="config.yaml", label_visibility="visible")

    # Healing toggle
    enable_heal = st.toggle("Enable LLM healing", value=True)

    # Custom run ID
    custom_run_id = st.text_input("Custom run ID (optional)", placeholder="auto-generated")

    st.markdown("---")
    st.markdown("#### Navigation")
    page = st.radio(
        "Go to",
        ["🚀 Run Pipeline", "📊 Dashboard", "🔬 Issue Explorer", "🛠️ Repair Log", "📁 Audit Export"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("Self-Healing Data Pipeline")
    st.caption("Powered by NVIDIA Llama 3.1 70B")


# ─────────────────────────────────────────────
#  Header
# ─────────────────────────────────────────────

st.markdown("""
<div style="padding: 2rem 0 1rem 0;">
  <div style="font-size:0.75rem; letter-spacing:0.15em; color:#3b82f6; font-weight:700; text-transform:uppercase; margin-bottom:0.4rem;">
    PROJECT 01
  </div>
  <h1 style="font-family:'Syne',sans-serif; font-size:2.4rem; font-weight:800; color:#e8f0ff; margin:0; line-height:1.1;">
    Self-Healing Data Pipeline
  </h1>
  <p style="color:#6b7fa3; margin-top:0.5rem; font-size:1rem;">
    Detects broken data and fixes it automatically using LLM diagnosis.
  </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")


# ─────────────────────────────────────────────
#  Helper: load audit DB directly
# ─────────────────────────────────────────────

AUDIT_DB = "pipeline_audit.db"

def get_runs(limit: int = 50, status: str = None) -> pd.DataFrame:
    if not os.path.exists(AUDIT_DB):
        return pd.DataFrame()
    where = f"WHERE status = '{status}'" if status else ""
    with sqlite3.connect(AUDIT_DB) as conn:
        try:
            return pd.read_sql(
                f"SELECT * FROM pipeline_runs {where} ORDER BY timestamp DESC LIMIT {limit}",
                conn
            )
        except Exception:
            return pd.DataFrame()

def get_issues(run_id: str = None, limit: int = 200) -> pd.DataFrame:
    if not os.path.exists(AUDIT_DB):
        return pd.DataFrame()
    where = f"WHERE run_id = '{run_id}'" if run_id else ""
    with sqlite3.connect(AUDIT_DB) as conn:
        try:
            return pd.read_sql(
                f"SELECT * FROM issue_events {where} ORDER BY timestamp DESC LIMIT {limit}",
                conn
            )
        except Exception:
            return pd.DataFrame()

def get_repairs(run_id: str = None, limit: int = 100) -> pd.DataFrame:
    if not os.path.exists(AUDIT_DB):
        return pd.DataFrame()
    where = f"WHERE run_id = '{run_id}'" if run_id else ""
    with sqlite3.connect(AUDIT_DB) as conn:
        try:
            return pd.read_sql(
                f"SELECT * FROM repair_attempts {where} ORDER BY timestamp DESC LIMIT {limit}",
                conn
            )
        except Exception:
            return pd.DataFrame()

def status_badge(status: str) -> str:
    icons = {"success": "✅", "partial": "⚠️", "failed": "❌", "quarantined": "🚫"}
    return f"{icons.get(status, '❓')} {status.upper()}"


# ─────────────────────────────────────────────
#  Page: Run Pipeline
# ─────────────────────────────────────────────

if page == "🚀 Run Pipeline":
    st.markdown("### Upload & Run")

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Drop your CSV file here",
            type=["csv"],
            help="CSV files supported. Max size depends on your system memory.",
        )

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        use_sample = st.button("📂 Use sample.csv", use_container_width=True)
        if use_sample:
            st.session_state["use_sample"] = True

    # Preview uploaded file
    if uploaded_file:
        try:
            preview_df = pd.read_csv(uploaded_file)
            uploaded_file.seek(0)
            st.markdown(f"**Preview** — {len(preview_df)} rows × {len(preview_df.columns)} columns")
            st.dataframe(preview_df.head(8), use_container_width=True, height=240)
        except Exception as e:
            st.error(f"Could not read file: {e}")

    st.markdown("---")

    # Run button
    run_clicked = st.button("▶ Run Pipeline", type="primary", use_container_width=False)

    if run_clicked or st.session_state.get("run_triggered"):
        st.session_state.pop("run_triggered", None)
        st.session_state.pop("use_sample", None)

        # Determine input source
        input_path = None

        if uploaded_file:
            # Save uploaded file temporarily
            tmp_path = f"data/_upload_{uuid.uuid4().hex[:8]}.csv"
            os.makedirs("data", exist_ok=True)
            with open(tmp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            input_path = tmp_path
        elif os.path.exists("data/sample.csv"):
            input_path = "data/sample.csv"
        else:
            st.error("No input file found. Upload a CSV or make sure data/sample.csv exists.")
            st.stop()

        if not check_env() and enable_heal:
            st.error("NVIDIA_API_KEY is missing. Add it to your .env file to enable healing.")
            st.stop()

        run_id = custom_run_id.strip() or str(uuid.uuid4())[:8]

        # ── Run the pipeline ─────────────────────
        st.markdown("---")
        st.markdown("### 🔄 Pipeline Running...")

        progress_bar = st.progress(0)
        status_area  = st.empty()

        def update(msg: str, pct: int):
            status_area.info(msg)
            progress_bar.progress(pct)

        try:
            update("Stage 1 / 5 — Ingesting data...", 10)
            pipeline = load_pipeline(config_path)

            update("Stage 2 / 5 — Validating schema and rules...", 30)
            update("Stage 3 / 5 — Detecting anomalies...", 50)

            if enable_heal:
                update("Stage 4 / 5 — Calling LLM healer (this may take ~30s)...", 65)
            else:
                update("Stage 4 / 5 — Healing skipped.", 65)

            result = pipeline.run(
                input_source=input_path,
                heal=enable_heal,
                run_id=run_id,
            )

            update("Stage 5 / 5 — Writing output...", 90)
            progress_bar.progress(100)
            status_area.empty()

            # ── Result summary ───────────────────
            st.markdown("---")
            st.markdown("### ✅ Run Complete")

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Status",            result.status.upper())
            c2.metric("Total Rows",        result.total_rows)
            c3.metric("Issues Found",      result.validation_issues + result.anomalies_detected)
            c4.metric("Issues Healed",     result.issues_healed)
            c5.metric("Duration",          f"{result.duration_seconds:.1f}s")

            if result.status == "success":
                st.success(f"All issues resolved! Clean file → `{result.output_path}`")
            elif result.status == "partial":
                st.warning(f"{result.issues_remaining} issue(s) remain. Output → `{result.output_path}`")
            elif result.status == "quarantined":
                st.error("Batch quarantined after max retries. Check the `quarantine/` folder.")
            elif result.status == "failed":
                st.error(f"Pipeline failed: {result.error_message}")

            # Show clean output preview
            if result.output_path and os.path.exists(result.output_path):
                with st.expander("📄 View cleaned output", expanded=False):
                    clean_df = pd.read_csv(result.output_path)
                    st.dataframe(clean_df, use_container_width=True, height=300)
                    st.download_button(
                        "⬇ Download clean CSV",
                        data=clean_df.to_csv(index=False),
                        file_name=f"{run_id}_clean.csv",
                        mime="text/csv",
                    )

            # Store run_id for cross-page navigation
            st.session_state["last_run_id"] = run_id

        except Exception as e:
            progress_bar.empty()
            status_area.empty()
            st.error(f"Pipeline error: {e}")
            import traceback
            with st.expander("Full traceback"):
                st.code(traceback.format_exc())

        finally:
            # Clean up temp upload file
            if uploaded_file and "tmp_path" in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)

    # Trigger from "Use sample.csv" button
    if st.session_state.get("use_sample"):
        st.session_state["run_triggered"] = True
        st.rerun()


# ─────────────────────────────────────────────
#  Page: Dashboard
# ─────────────────────────────────────────────

elif page == "📊 Dashboard":
    st.markdown("### Pipeline Dashboard")

    runs_df = get_runs(limit=100)

    if runs_df.empty:
        st.info("No pipeline runs yet. Go to **Run Pipeline** to get started.")
    else:
        # ── KPI row ──────────────────────────────
        total        = len(runs_df)
        successes    = len(runs_df[runs_df["status"] == "success"])
        partials     = len(runs_df[runs_df["status"] == "partial"])
        failures     = len(runs_df[runs_df["status"].isin(["failed", "quarantined"])])
        total_healed = int(runs_df["issues_healed"].sum())
        avg_duration = runs_df["duration_seconds"].mean()

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Runs",      total)
        c2.metric("✅ Success",       successes)
        c3.metric("⚠️ Partial",      partials)
        c4.metric("❌ Failed",        failures)
        c5.metric("🔧 Total Healed",  total_healed)

        st.markdown("---")

        # ── Charts row ───────────────────────────
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Issues per run")
            chart_df = runs_df[["run_id", "validation_issues", "anomalies_detected", "issues_healed"]].head(15)
            chart_df = chart_df.set_index("run_id")
            st.bar_chart(chart_df, height=260)

        with col2:
            st.markdown("#### Duration per run (seconds)")
            dur_df = runs_df[["run_id", "duration_seconds"]].head(15).set_index("run_id")
            st.line_chart(dur_df, height=260)

        st.markdown("---")

        # ── Run history table ────────────────────
        st.markdown("#### Recent runs")

        display_df = runs_df[[
            "run_id", "timestamp", "status", "total_rows",
            "validation_issues", "anomalies_detected",
            "issues_healed", "issues_remaining",
            "heal_attempts", "duration_seconds"
        ]].copy()

        display_df["timestamp"] = pd.to_datetime(
            display_df["timestamp"]
        ).dt.strftime("%Y-%m-%d %H:%M:%S")

        st.dataframe(display_df, use_container_width=True, height=350)

        # ── Status filter ────────────────────────
        st.markdown("---")
        col1, col2 = st.columns([1, 3])
        with col1:
            filter_status = st.selectbox(
                "Filter by status",
                ["All", "success", "partial", "failed", "quarantined"]
            )

        filtered = (
            runs_df if filter_status == "All"
            else runs_df[runs_df["status"] == filter_status]
        )
        st.caption(f"{len(filtered)} run(s) matching filter")
        st.dataframe(
            filtered[["run_id", "timestamp", "status", "total_rows",
                       "issues_healed", "issues_remaining", "duration_seconds"]],
            use_container_width=True, height=250
        )


# ─────────────────────────────────────────────
#  Page: Issue Explorer
# ─────────────────────────────────────────────

elif page == "🔬 Issue Explorer":
    st.markdown("### Issue Explorer")
    st.caption("Browse every validation and anomaly issue detected across runs.")

    runs_df = get_runs(limit=50)

    if runs_df.empty:
        st.info("No pipeline runs yet.")
    else:
        run_ids  = ["All runs"] + runs_df["run_id"].tolist()
        selected = st.selectbox("Select run", run_ids)

        chosen_id  = None if selected == "All runs" else selected
        issues_df  = get_issues(run_id=chosen_id)

        if issues_df.empty:
            st.info("No issues recorded for this selection.")
        else:
            # Summary metrics
            val_issues  = issues_df[issues_df["source"] == "validation"]
            anom_issues = issues_df[issues_df["source"] == "anomaly"]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total issues",      len(issues_df))
            c2.metric("Validation",        len(val_issues))
            c3.metric("Anomalies",         len(anom_issues))
            c4.metric("Columns affected",  issues_df["column_name"].nunique())

            st.markdown("---")

            # Source filter
            source_filter = st.radio(
                "Show",
                ["All", "Validation only", "Anomaly only"],
                horizontal=True,
            )
            if source_filter == "Validation only":
                issues_df = val_issues
            elif source_filter == "Anomaly only":
                issues_df = anom_issues

            # Issue type breakdown
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### By issue type")
                type_counts = issues_df["issue_type"].value_counts().reset_index()
                type_counts.columns = ["issue_type", "count"]
                st.dataframe(type_counts, use_container_width=True, height=220)

            with col2:
                st.markdown("#### By column")
                col_counts = issues_df["column_name"].value_counts().reset_index()
                col_counts.columns = ["column", "count"]
                st.dataframe(col_counts, use_container_width=True, height=220)

            st.markdown("---")
            st.markdown("#### Full issue log")
            display_cols = [
                "run_id", "source", "column_name", "issue_type",
                "severity", "description", "affected_count", "timestamp"
            ]
            st.dataframe(
                issues_df[[c for c in display_cols if c in issues_df.columns]],
                use_container_width=True, height=360
            )


# ─────────────────────────────────────────────
#  Page: Repair Log
# ─────────────────────────────────────────────

elif page == "🛠️ Repair Log":
    st.markdown("### LLM Repair Log")
    st.caption("Every healing attempt made by the NVIDIA Llama 3.1 70B model.")

    runs_df = get_runs(limit=50)

    if runs_df.empty:
        st.info("No pipeline runs yet.")
    else:
        run_ids  = ["All runs"] + runs_df["run_id"].tolist()
        selected = st.selectbox("Select run", run_ids)

        chosen_id   = None if selected == "All runs" else selected
        repairs_df  = get_repairs(run_id=chosen_id)

        if repairs_df.empty:
            st.info("No repair attempts recorded. This could mean the data was clean or healing was disabled.")
        else:
            # Summary
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total attempts",   len(repairs_df))
            c2.metric("✅ Successful",     len(repairs_df[repairs_df["outcome"] == "success"]))
            c3.metric("❌ Failed",         len(repairs_df[repairs_df["outcome"] != "success"]))
            avg_conf = repairs_df["confidence"].mean()
            c4.metric("Avg confidence",   f"{avg_conf:.0%}" if not pd.isna(avg_conf) else "—")

            st.markdown("---")

            # Attempt cards
            st.markdown("#### Attempt details")
            for _, row in repairs_df.iterrows():
                outcome_icon = {
                    "success":            "✅",
                    "failed_execution":   "💥",
                    "failed_validation":  "⚠️",
                    "low_confidence":     "🤔",
                }.get(row["outcome"], "❓")

                with st.expander(
                    f"{outcome_icon}  Run `{row['run_id']}` — Attempt {int(row['attempt_number'])} "
                    f"| {row['outcome']} | confidence {row['confidence']:.0%}",
                    expanded=False,
                ):
                    col1, col2 = st.columns([1, 1])

                    with col1:
                        st.markdown("**Diagnosis**")
                        st.info(row["diagnosis"] or "—")
                        st.markdown(
                            f"**Issues:** {int(row['issues_before'])} → {int(row['issues_after'])}"
                        )

                    with col2:
                        if row.get("error_message"):
                            st.markdown("**Error**")
                            st.error(str(row["error_message"])[:400])

                    if row.get("fix_code"):
                        st.markdown("**Generated fix code**")
                        st.code(row["fix_code"], language="python")


# ─────────────────────────────────────────────
#  Page: Audit Export
# ─────────────────────────────────────────────

elif page == "📁 Audit Export":
    st.markdown("### Audit Export")
    st.caption("Download the full audit log for external analysis or archiving.")

    runs_df    = get_runs(limit=10000)
    issues_df  = get_issues(limit=10000)
    repairs_df = get_repairs(limit=10000)

    if runs_df.empty:
        st.info("No data to export yet.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Pipeline runs",   len(runs_df))
        c2.metric("Issue events",    len(issues_df))
        c3.metric("Repair attempts", len(repairs_df))

        st.markdown("---")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### Pipeline Runs")
            st.download_button(
                "⬇ Download runs.csv",
                data=runs_df.to_csv(index=False),
                file_name="pipeline_runs.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with col2:
            st.markdown("#### Issue Events")
            if not issues_df.empty:
                st.download_button(
                    "⬇ Download issues.csv",
                    data=issues_df.to_csv(index=False),
                    file_name="issue_events.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            else:
                st.caption("No issue data yet.")

        with col3:
            st.markdown("#### Repair Attempts")
            if not repairs_df.empty:
                st.download_button(
                    "⬇ Download repairs.csv",
                    data=repairs_df.to_csv(index=False),
                    file_name="repair_attempts.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            else:
                st.caption("No repair data yet.")

        st.markdown("---")
        st.markdown("#### Full JSON export")

        import json
        export_data = {
            "exported_at":     datetime.now().isoformat(),
            "pipeline_runs":   runs_df.to_dict("records"),
            "issue_events":    issues_df.to_dict("records") if not issues_df.empty else [],
            "repair_attempts": repairs_df.to_dict("records") if not repairs_df.empty else [],
        }
        st.download_button(
            "⬇ Download full audit_export.json",
            data=json.dumps(export_data, indent=2, default=str),
            file_name="audit_export.json",
            mime="application/json",
            use_container_width=False,
        )