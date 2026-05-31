"""
ingestion.py
------------
Data ingestion module for the Self-Healing Data Pipeline.

Handles loading data from multiple sources into a pandas DataFrame:
  - Local files  : CSV, Excel (.xlsx/.xls), JSON, Parquet
  - REST API     : GET endpoint returning JSON or CSV
  - SQL database : any SQLAlchemy-compatible DB (SQLite, PostgreSQL, MySQL)
  - Raw DataFrame: pass one in directly (useful for testing / programmatic use)

Every load is wrapped in an IngestionResult that carries the DataFrame,
metadata (source, rows, columns, load time), and any error that occurred.

Usage:
    from ingestion import Ingestor

    ingestor = Ingestor()

    # From a file
    result = ingestor.load("data/sample.csv")

    # From a REST API
    result = ingestor.load("https://api.example.com/data.json")

    # From a database
    result = ingestor.load(
        "sqlite:///my.db",
        query="SELECT * FROM users WHERE active = 1"
    )

    if result.success:
        df = result.df
        print(result.summary())
"""

import os
import time
import json
import requests
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime
from io import StringIO
from loguru import logger
from typing import Optional


# ─────────────────────────────────────────────
#  Result dataclass
# ─────────────────────────────────────────────

@dataclass
class IngestionResult:
    """Metadata and outcome of a single ingestion attempt."""
    source:          str
    source_type:     str         # file_csv | file_excel | file_json | file_parquet
                                 # api_json | api_csv | database | dataframe
    success:         bool        = False
    df:              Optional[pd.DataFrame] = None
    row_count:       int         = 0
    column_count:    int         = 0
    columns:         list[str]   = field(default_factory=list)
    load_time_ms:    float       = 0.0
    error_message:   str         = ""
    timestamp:       str         = field(default_factory=lambda: datetime.now().isoformat())

    def summary(self) -> str:
        if self.success:
            return (
                f"✅ Ingested [{self.source_type}] '{self.source}' — "
                f"{self.row_count} rows × {self.column_count} cols "
                f"in {self.load_time_ms:.0f}ms"
            )
        return f"❌ Ingestion failed [{self.source_type}] '{self.source}' — {self.error_message}"

    def to_dict(self) -> dict:
        return {
            "source":        self.source,
            "source_type":   self.source_type,
            "success":       self.success,
            "row_count":     self.row_count,
            "column_count":  self.column_count,
            "columns":       self.columns,
            "load_time_ms":  round(self.load_time_ms, 2),
            "error_message": self.error_message,
            "timestamp":     self.timestamp,
        }


# ─────────────────────────────────────────────
#  Ingestor
# ─────────────────────────────────────────────

class Ingestor:
    """
    Unified data loader that detects the source type automatically
    and returns a consistent IngestionResult.

    Source detection logic:
      - Starts with 'http://' or 'https://'  → API
      - Starts with a DB dialect prefix       → Database
        (sqlite:///, postgresql://, mysql://, etc.)
      - Is a pd.DataFrame                     → Direct
      - Otherwise                             → File (by extension)
    """

    # Database URI prefixes recognised as SQL sources
    DB_PREFIXES = (
        "sqlite:///", "sqlite://",
        "postgresql://", "postgres://",
        "mysql://", "mysql+pymysql://",
        "mssql://", "oracle://",
    )

    # Mapping of file extension → (source_type label, loader function)
    FILE_LOADERS = {
        ".csv":     ("file_csv",     pd.read_csv),
        ".xlsx":    ("file_excel",   pd.read_excel),
        ".xls":     ("file_excel",   pd.read_excel),
        ".json":    ("file_json",    pd.read_json),
        ".parquet": ("file_parquet", pd.read_parquet),
        ".tsv":     ("file_csv",     lambda p: pd.read_csv(p, sep="\t")),
    }

    def __init__(self, request_timeout: int = 30):
        """
        Args:
            request_timeout : seconds before an API request times out (default 30).
        """
        self.request_timeout = request_timeout

    # ── Public entry point ──────────────────────

    def load(
        self,
        source: "str | pd.DataFrame",
        query: Optional[str] = None,
        api_headers: Optional[dict] = None,
        csv_kwargs: Optional[dict] = None,
        excel_kwargs: Optional[dict] = None,
    ) -> IngestionResult:
        """
        Load data from any supported source into a DataFrame.

        Args:
            source       : file path, URL, DB connection string, or DataFrame.
            query        : SQL query string (required for database sources).
            api_headers  : extra HTTP headers for API sources (e.g. auth tokens).
            csv_kwargs   : extra kwargs forwarded to pd.read_csv (e.g. encoding).
            excel_kwargs : extra kwargs forwarded to pd.read_excel (e.g. sheet_name).

        Returns:
            IngestionResult — always returns one, never raises.
        """
        start = time.time()

        # ── Detect source type and dispatch ─────
        if isinstance(source, pd.DataFrame):
            result = self._load_dataframe(source)

        elif isinstance(source, str) and source.startswith(("http://", "https://")):
            result = self._load_api(source, headers=api_headers or {})

        elif isinstance(source, str) and source.startswith(self.DB_PREFIXES):
            result = self._load_database(source, query=query)

        elif isinstance(source, str):
            result = self._load_file(
                source,
                csv_kwargs=csv_kwargs or {},
                excel_kwargs=excel_kwargs or {},
            )

        else:
            result = IngestionResult(
                source=str(source),
                source_type="unknown",
                success=False,
                error_message=f"Unrecognised source type: {type(source).__name__}",
            )

        result.load_time_ms = (time.time() - start) * 1000

        if result.success:
            logger.success(result.summary())
        else:
            logger.error(result.summary())

        return result

    # ── File loader ─────────────────────────────

    def _load_file(
        self,
        path: str,
        csv_kwargs: dict,
        excel_kwargs: dict,
    ) -> IngestionResult:
        """Load a local file by extension."""
        ext = os.path.splitext(path)[1].lower()

        if ext not in self.FILE_LOADERS:
            return IngestionResult(
                source=path,
                source_type="file_unknown",
                success=False,
                error_message=(
                    f"Unsupported file extension '{ext}'. "
                    f"Supported: {', '.join(self.FILE_LOADERS)}"
                ),
            )

        source_type, loader = self.FILE_LOADERS[ext]

        if not os.path.exists(path):
            return IngestionResult(
                source=path,
                source_type=source_type,
                success=False,
                error_message=f"File not found: '{path}'",
            )

        try:
            # Forward format-specific kwargs
            if ext == ".csv" or ext == ".tsv":
                df = loader(path, **csv_kwargs)
            elif ext in (".xlsx", ".xls"):
                df = loader(path, **excel_kwargs)
            else:
                df = loader(path)

            return self._build_success(df, path, source_type)

        except Exception as e:
            return IngestionResult(
                source=path,
                source_type=source_type,
                success=False,
                error_message=f"Failed to read file: {e}",
            )

    # ── API loader ──────────────────────────────

    def _load_api(self, url: str, headers: dict) -> IngestionResult:
        """
        Fetch data from a REST API endpoint.

        Supports:
          - JSON response  → pd.DataFrame via pd.json_normalize or pd.read_json
          - CSV response   → pd.read_csv via StringIO
        """
        logger.info(f"Fetching API: {url}")
        try:
            response = requests.get(url, headers=headers, timeout=self.request_timeout)
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "")

            # ── Parse response body ──────────────
            if "text/csv" in content_type or url.endswith(".csv"):
                df = pd.read_csv(StringIO(response.text))
                source_type = "api_csv"

            elif "application/json" in content_type or url.endswith(".json"):
                data = response.json()

                # Handle top-level list vs dict with a data key
                if isinstance(data, list):
                    df = pd.json_normalize(data)
                elif isinstance(data, dict):
                    # Try common wrapper keys: data, results, items, records
                    for key in ("data", "results", "items", "records"):
                        if key in data and isinstance(data[key], list):
                            df = pd.json_normalize(data[key])
                            break
                    else:
                        df = pd.json_normalize([data])
                else:
                    raise ValueError(f"Unexpected JSON root type: {type(data).__name__}")

                source_type = "api_json"

            else:
                # Try JSON first, fall back to CSV
                try:
                    data = response.json()
                    df   = pd.json_normalize(data if isinstance(data, list) else [data])
                    source_type = "api_json"
                except Exception:
                    df = pd.read_csv(StringIO(response.text))
                    source_type = "api_csv"

            return self._build_success(df, url, source_type)

        except requests.exceptions.Timeout:
            return IngestionResult(
                source=url, source_type="api_json", success=False,
                error_message=f"Request timed out after {self.request_timeout}s",
            )
        except requests.exceptions.ConnectionError:
            return IngestionResult(
                source=url, source_type="api_json", success=False,
                error_message="Connection error — check the URL and your network.",
            )
        except requests.exceptions.HTTPError as e:
            return IngestionResult(
                source=url, source_type="api_json", success=False,
                error_message=f"HTTP {response.status_code}: {e}",
            )
        except Exception as e:
            return IngestionResult(
                source=url, source_type="api_json", success=False,
                error_message=f"API ingestion failed: {e}",
            )

    # ── Database loader ─────────────────────────

    def _load_database(self, connection_string: str, query: Optional[str]) -> IngestionResult:
        """
        Load data from a SQL database via SQLAlchemy.

        Args:
            connection_string : SQLAlchemy URL e.g. 'sqlite:///mydb.db'
            query             : SQL SELECT statement. If None, lists available tables.
        """
        try:
            from sqlalchemy import create_engine, text
        except ImportError:
            return IngestionResult(
                source=connection_string,
                source_type="database",
                success=False,
                error_message="sqlalchemy is not installed. Run: pip install sqlalchemy",
            )

        if not query:
            return IngestionResult(
                source=connection_string,
                source_type="database",
                success=False,
                error_message=(
                    "A SQL query is required for database sources. "
                    "Pass it via: ingestor.load(conn_str, query='SELECT ...')"
                ),
            )

        logger.info(f"Connecting to database: {connection_string}")
        try:
            engine = create_engine(connection_string)
            with engine.connect() as conn:
                df = pd.read_sql(text(query), conn)
            engine.dispose()
            return self._build_success(df, connection_string, "database")

        except Exception as e:
            return IngestionResult(
                source=connection_string,
                source_type="database",
                success=False,
                error_message=f"Database ingestion failed: {e}",
            )

    # ── Direct DataFrame ────────────────────────

    def _load_dataframe(self, df: pd.DataFrame) -> IngestionResult:
        """Accept a DataFrame passed directly."""
        return self._build_success(df.copy(), "<DataFrame>", "dataframe")

    # ── Helper ──────────────────────────────────

    def _build_success(
        self, df: pd.DataFrame, source: str, source_type: str
    ) -> IngestionResult:
        """Build a successful IngestionResult from a loaded DataFrame."""
        return IngestionResult(
            source=source,
            source_type=source_type,
            success=True,
            df=df,
            row_count=len(df),
            column_count=len(df.columns),
            columns=df.columns.tolist(),
        )


# ─────────────────────────────────────────────
#  Quick test — run directly to see output
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import json as _json

    ingestor = Ingestor()

    print("\n" + "=" * 60)
    print("  INGESTION TEST SUITE")
    print("=" * 60)

    # ── Test 1: CSV file ─────────────────────────
    print("\n[1] Loading CSV file...")
    result = ingestor.load("data/sample.csv")
    print(result.summary())
    if result.success:
        print(f"    Columns : {result.columns}")
        print(f"    Preview :\n{result.df.head(3).to_string(index=False)}")

    # ── Test 2: Non-existent file ────────────────
    print("\n[2] Loading non-existent file...")
    result = ingestor.load("data/does_not_exist.csv")
    print(result.summary())

    # ── Test 3: Unsupported extension ────────────
    print("\n[3] Loading unsupported file type...")
    result = ingestor.load("data/file.xyz")
    print(result.summary())

    # ── Test 4: Direct DataFrame ─────────────────
    print("\n[4] Loading from a DataFrame directly...")
    sample_df = pd.DataFrame({
        "id":    [1, 2, 3],
        "name":  ["Alice", "Bob", "Charlie"],
        "score": [95, 87, 72],
    })
    result = ingestor.load(sample_df)
    print(result.summary())
    if result.success:
        print(f"    Preview :\n{result.df.to_string(index=False)}")

    # ── Test 5: Public REST API ──────────────────
    print("\n[5] Loading from a public REST API...")
    result = ingestor.load("https://jsonplaceholder.typicode.com/users")
    print(result.summary())
    if result.success:
        print(f"    Columns : {result.columns}")
        print(f"    Rows    : {result.row_count}")

    # ── Test 6: SQLite database ──────────────────
    print("\n[6] Loading from an in-memory SQLite database...")
    try:
        import sqlalchemy
        from sqlalchemy import create_engine, text as sa_text

        engine = create_engine("sqlite:///:memory:")
        sample_df.to_sql("users", engine, index=False, if_exists="replace")
        engine.dispose()

        result = ingestor.load(
            "sqlite:///:memory:",
            query="SELECT * FROM users",
        )
        # Note: in-memory DB is gone after engine.dispose() — just testing the path
        print(result.summary())
    except ImportError:
        print("    ⚠️  sqlalchemy not installed — skipping DB test.")

    print("\n" + "=" * 60)
    print("  Test suite complete.")
    print("=" * 60)