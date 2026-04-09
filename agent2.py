import os
import time
import json
import hashlib
import re
from datetime import datetime
from typing import TypedDict, Optional, Dict
from langgraph.graph import Graph
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from pathlib import Path
from functools import lru_cache
from sqlalchemy.types import UserDefinedType
from sqlalchemy import text

from close_query_select import SQLQueryRetriever

os.environ["LANG"] = "C.UTF-8"
os.environ["PGCLIENTENCODING"] = "utf-8"  # harmless no-op for MSSQL; left as-is to keep code shape

# Treating geocoord types (harmless placeholder for MSSQL)
class GeoCoord(UserDefinedType):
    def __init__(self, *args, **kwargs):
        pass
    def get_col_spec(self, **kw):
        return "geometry"
    def bind_processor(self, dialect):
        return None
    def result_processor(self, dialect, coltype):
        return None

# ---- State ----
class State(TypedDict, total=False):
    question: str
    query: Optional[str]
    result: Optional[str]
    error: Optional[str]
    answer: Optional[str]
    db_schema: Dict
    execution_time_seconds: Optional[float]
    explain_plan: Optional[str]
    index_suggestions: Optional[list]
    index_notes: Optional[str]
    created_indexes: Optional[list]
    execution_time_seconds_after_index: Optional[float]

# ---- Load env / Connect (MSSQL) ----
load_dotenv(override=True)
db_host = os.getenv("DB_HOST")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_database = os.getenv("DB_DATABASE")
db_port = os.getenv("DB_PORT")

DATABASE_URI = (
    f"mssql+pyodbc://{db_user}:{db_password}@{db_host}:{db_port}/{db_database}"
    "?driver=ODBC+Driver+18+for+SQL+Server&TrustServerCertificate=yes"
)
print(f"[DEBUG] Connecting to DB: {DATABASE_URI}")

db = SQLDatabase.from_uri(
    DATABASE_URI,
    include_tables=None,
    sample_rows_in_table_info=2,
    view_support=True
)

dialect_name = db._engine.dialect.name  # 'mssql'

def _run_ddl_autocommit(sql: str):
    """Run DDL in autocommit context (works with MSSQL/pyodbc)."""
    engine = db._engine
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.exec_driver_sql(sql)

# ---- LLM ----
llm = ChatOpenAI(
    openai_api_base="https://api.deepseek.com/v1",
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    model="deepseek-chat",
    temperature=0.0
)

# ---- Logging ----
def log_query(question: str, query: str, execution_time: float, log_path: str = "query_log.json"):
    timestamp = datetime.now().isoformat()
    log_entry = {
        "timestamp": timestamp,
        "question": question,
        "query": query,
        "execution_time_seconds": round(execution_time, 4)
    }
    if Path(log_path).exists():
        with open(log_path, "r", encoding="utf-8") as f:
            try:
                logs = json.load(f)
            except json.JSONDecodeError:
                logs = []
    else:
        logs = []
    logs.append(log_entry)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2, ensure_ascii=False)

# ---- Schema (MSSQL) ----
@lru_cache(maxsize=1)
def get_db_schema() -> Dict:
    """
    Get SQL Server schema with:
      - columns (name/type/null/len)
      - primary_keys
      - foreign_keys {col: {foreign_table, foreign_column}}
      - indices: [{index_name, is_unique, is_primary, columns:[ordered]}]
      - row_estimate (from sys.dm_db_partition_stats)
    """
    print("Loading schema from database...")
    # reflect table list from langchain SQLDatabase
    tables = db.get_usable_table_names()
    schema: Dict = {}

    # helper to split [schema].[table] or plain table
    def split_two(part: str):
        if not part:
            return None, None
        if '.' in part:
            s, t = part.split('.', 1)
            return s.strip('[]'), t.strip('[]')
        return None, part.strip('[]')

    for table in tables:
        try:
            tbl_schema, tbl_name = split_two(table)
            # if no schema, let SQL Server resolve via default schema
            qualified = f"{('['+tbl_schema+'].') if tbl_schema else ''}[{tbl_name}]"
            qualified_unbraced = f"{(tbl_schema + '.') if tbl_schema else ''}{tbl_name}"

            # --- Columns ---
            columns = db.run(f"""
                SELECT 
                    c.name AS column_name,
                    t.name AS data_type,
                    c.is_nullable,
                    c.max_length
                FROM sys.columns c
                JOIN sys.types t ON c.user_type_id = t.user_type_id
                JOIN sys.objects o ON o.object_id = c.object_id
                WHERE o.object_id = OBJECT_ID(N'{qualified_unbraced}')
                ORDER BY c.column_id;
            """)

            # --- Primary Keys ---
            pk_result = db.run(f"""
                SELECT c.name AS column_name
                FROM sys.key_constraints kc
                JOIN sys.index_columns ic ON kc.parent_object_id = ic.object_id AND kc.unique_index_id = ic.index_id
                JOIN sys.columns c ON ic.object_id = c.object_id AND ic.column_id = c.column_id
                WHERE kc.parent_object_id = OBJECT_ID(N'{qualified_unbraced}')
                  AND kc.type = 'PK'
                ORDER BY ic.key_ordinal;
            """)
            primary_keys = [row[0] for row in (pk_result or []) if isinstance(row, (list, tuple)) and row] or []

            # --- Foreign Keys ---
            fk_result = db.run(f"""
                SELECT 
                    pc.name AS column_name,
                    OBJECT_SCHEMA_NAME(fk.referenced_object_id) AS foreign_schema,
                    OBJECT_NAME(fk.referenced_object_id) AS foreign_table,
                    rc.name AS foreign_column
                FROM sys.foreign_keys fk
                JOIN sys.foreign_key_columns fkc 
                     ON fk.object_id = fkc.constraint_object_id
                JOIN sys.columns pc 
                     ON pc.object_id = fkc.parent_object_id AND pc.column_id = fkc.parent_column_id
                JOIN sys.columns rc 
                     ON rc.object_id = fkc.referenced_object_id AND rc.column_id = fkc.referenced_column_id
                WHERE fk.parent_object_id = OBJECT_ID(N'{qualified_unbraced}');
            """)
            foreign_keys = {}
            for row in (fk_result or []):
                if isinstance(row, (list, tuple)) and len(row) >= 4:
                    col = row[0]
                    f_schema = row[1]
                    f_table  = row[2]
                    f_col    = row[3]
                    fq = f"{f_schema}.{f_table}" if f_schema else f_table
                    foreign_keys[col] = {"foreign_table": fq, "foreign_column": f_col}

            # --- Row estimate (approx) ---
            row_estimate_result = db.run(f"""
                SELECT SUM(row_count)
                FROM sys.dm_db_partition_stats
                WHERE object_id = OBJECT_ID(N'{qualified_unbraced}')
                  AND index_id IN (0,1);
            """)
            row_estimate = 0
            try:
                if row_estimate_result:
                    r = row_estimate_result[0]
                    if isinstance(r, (list, tuple)) and r and r[0] is not None:
                        row_estimate = int(r[0])
                    else:
                        # Fallback to sys.partitions if DMV not available
                        row_estimate_result2 = db.run(f"""
                            SELECT SUM(p.rows)
                            FROM sys.partitions p
                            WHERE p.object_id = OBJECT_ID(N'{qualified_unbraced}')
                              AND p.index_id IN (0,1);
                        """)
                        if row_estimate_result2:
                            r2 = row_estimate_result2[0]
                            if isinstance(r2, (list, tuple)) and r2 and r2[0] is not None:
                                row_estimate = int(r2[0])
            except Exception:
                row_estimate = 0

            # --- Indexes with ORDERED columns ---
            index_result = db.run(f"""
                SELECT 
                  i.name AS index_name,
                  i.is_unique,
                  CASE WHEN kc.type = 'PK' THEN 1 ELSE 0 END AS is_primary,
                  col.name AS col_name,
                  ic.key_ordinal AS col_position
                FROM sys.indexes i
                JOIN sys.index_columns ic 
                  ON i.object_id = ic.object_id AND i.index_id = ic.index_id
                JOIN sys.columns col
                  ON ic.object_id = col.object_id AND ic.column_id = col.column_id
                LEFT JOIN sys.key_constraints kc
                  ON kc.parent_object_id = i.object_id AND kc.unique_index_id = i.index_id
                WHERE i.object_id = OBJECT_ID(N'{qualified_unbraced}')
                  AND i.is_hypothetical = 0
                ORDER BY i.name, ic.key_ordinal;
            """)

            idx_by_name = {}
            for row in (index_result or []):
                if not (isinstance(row, (list, tuple)) and len(row) >= 5):
                    continue
                index_name, is_unique, is_primary, col_name, col_pos = row
                meta = idx_by_name.setdefault(index_name, {
                    "index_name": index_name,
                    "is_unique": bool(is_unique),
                    "is_primary": bool(is_primary),
                    "columns": []
                })
                meta["columns"].append(col_name)

            indexes = list(idx_by_name.values())

            # --- Compose entry ---
            if columns:
                cols_map = {}
                for col in columns:
                    if not (isinstance(col, (list, tuple)) and len(col) >= 1):
                        continue
                    name  = col[0]
                    dtype = col[1] if len(col) > 1 else None
                    isnul = col[2] if len(col) > 2 else None
                    maxlen = col[3] if len(col) > 3 else None
                    cols_map[name] = {
                        "type": dtype,
                        "nullable": bool(isnul),
                        "max_length": maxlen,
                        "primary_key": name in primary_keys,
                        "foreign_key": foreign_keys.get(name)
                    }

                schema[table] = {
                    "columns": cols_map,
                    "description": f"Table {table}",
                    "primary_keys": primary_keys,
                    "foreign_keys": foreign_keys,
                    "indices": indexes,            # ordered
                    "row_estimate": row_estimate,  # approx
                }

        except Exception as e:
            print(f"[ERROR] Failed to fetch schema for table '{table}': {e}")
            continue

    return schema

# ---- Fingerprint ----
def generate_schema_fingerprint(schema: Dict) -> str:
    if not schema:
        return ""
    fingerprint_parts = []
    for table, details in sorted(schema.items()):
        columns_part = ",".join(sorted(details["columns"].keys()))
        pks_part = ",".join(sorted(details.get("primary_keys", [])))
        fingerprint_parts.append(f"{table}:{columns_part}:{pks_part}")
    return "|".join(fingerprint_parts)

# ---- Prompt ----
query_prompt_template = PromptTemplate(
    input_variables=["dialect", "table_info", "input", "schema_info", "few_args"],
    template=(
        "You are a SQL Server expert. Given this schema and question, generate an accurate T-SQL query.\n\n"
        "Database Schema:\n{table_info}\n\n"
        "Detailed Structure:\n{schema_info}\n\n"
        "Follow these rules:\n"
        "{few_args}\n"
        # "- Rewrite the given sql query so that it remains semantically identical but is as efficient as possible in terms of execution time. Avoid changing the result — only optimize the structure and form of the query. \n"
        
        # "- After the initial thought above, apply the following heuristics: \n"
        # "- When a WHERE clause filters on R.A, and there's a join R.A = S.B, you must also add the same filter to S.B in the WHERE clause. Even if redundant. Always apply this. For example, before: SELECT * FROM R JOIN S ON R.id = S.id WHERE R.id = 123; and after: SELECT * FROM R JOIN S ON R.id = S.id WHERE R.id = 123 AND S.id = 123; \n"
        # "- If the SQL contains col IN (val1, val2, ...), rewrite as multiple col = valX conditions joined by OR. \n"
        # "- Avoid correlated subqueries. Prefer Common Table Expressions (CTEs) or JOINs for filtering based on per-group logic (e.g., MAX per group), especially when computing additional fields.\n"
        
        # "- Do correlated subqueries. Avoid Common table expressions (CTEs) or JOINs for filtering based on per-group logic (e.g., MAX per group), expecially when computing additional fields.\n"
        
        # "- Remove unnecessary GROUP BY clauses. If there is no HAVING clause and the SELECT contains only one aggregate function or no aggregates and the GROUP BY attribute is a primary key (from a single table or not used as a foreign key), the GROUP BY can be eliminated to reduce query cost.\n"
        # "- Change query with disjunction in the WHERE to a union of query results.\n"
        # "- Remove ALL operation with greater/less-than comparison operators by including a MAX or MIN aggregate function in the subquery.\n"
        # "- Remove SOME/ANY operation with greater/less-than comparison operators by including a MAX or MIN aggregate function in the subquery. Do not change ANYTHING ELSE. DO NOT REMOVE ORDER BY!\n"
        # "- Replace IN set operation by a join operation.\n"
        # "- Eliminate DISTINCT if the SELECTed columns come from the primary table and the JOIN key is a primary or unique key.\n"
        # "- Move function applied to a column index to another position in the expression. DO NOT DO THIS: SELECT * FROM h_lineitem WHERE CAST(l_quantity AS VARCHAR(20)) = '28' AND DO NOT DO THIS: SELECT * FROM H_Lineitem WHERE l_quantity = 28.0; INSTEAD DO THIS: SELECT * FROM h_lineitem WHERE l_quantity = CAST('28' AS NUMERIC); \n\n"
        # "- Move arithmetic expression applied to a column index to another position in the expression.\n"
        "\nQuestion: {input}\n\n"
        "Return ONLY the SQL query in ```sql``` blocks."
    ),
)

# ---- SQL extract ----
def extract_sql_query(text):
    if not text:
        return None
    matches = re.findall(r"```(?:sql)?\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if matches:
        for block in reversed(matches):
            query = block.strip()
            if query.lower().startswith(('select', 'insert', 'update', 'delete', 'with')):
                return query
    match = re.search(r"\b(?:SELECT|INSERT|UPDATE|DELETE|WITH).*?(?:;|$)", text, re.DOTALL | re.IGNORECASE)
    return match.group(0).strip() if match else None

# ---- Query generation ----
def sql_query(state: State) -> State:
    print("\nGenerating SQL query...")
    schema = state.get("db_schema", {})

    schema_info = []
    for table, details in schema.items():
        columns = []
        for col_name, col_info in details["columns"].items():
            col_desc = f"{col_name} ({col_info['type']}"
            if col_info["primary_key"]:
                col_desc += ", PK"
            if col_info["foreign_key"]:
                fk = col_info["foreign_key"]
                col_desc += f", FK->{fk['foreign_table']}.{fk['foreign_column']}"
            columns.append(col_desc)
        schema_info.append(f"Table {table}:\n  - " + "\n  - ".join(columns))

    try:
        prompt = query_prompt_template.invoke({
            "dialect": "mssql",
            "table_info": db.get_table_info(),
            "input": state["question"],
            "schema_info": "\n\n".join(schema_info),
            "few_args": state["few_args"]
        })
        response = llm.invoke(prompt)
        query = extract_sql_query(response.content)
        if not query:
            return {**state, "error": "Failed to extract valid SQL query"}
        if ";" not in query:
            query += ";"
        return {**state, "query": query}
    except Exception as e:
        print(f"Query generation failed: {e}")
        return {**state, "error": f"Query generation failed: {str(e)}"}

SMALL_TABLE_ROW_THRESHOLD = 50_000  # unified threshold

def _resolve_table_key(schema: dict, table: str) -> str:
    """Resolve a suggested table name to the actual key in db_schema (handles dbo., brackets, case)."""
    if table in schema:
        return table
    t = table.strip('[]').lower()
    # Try exact, schema-qualified, or last-part match
    candidates = []
    for k in schema.keys():
        kk = k.strip('[]').lower()
        if kk == t or kk.endswith('.' + t) or kk.split('.')[-1] == t:
            candidates.append(k)
    if candidates:
        candidates.sort(key=len)  # deterministic: shortest key first
        return candidates[0]
    return table  # fallback (may still miss; preserves current behavior)

def _has_leading_index(schema: dict, table: str, column: str) -> bool:
    try:
        table_key = _resolve_table_key(schema, table)
        for idx in (schema.get(table_key, {}).get("indices") or []):
            cols = idx.get("columns") or []
            if cols and cols[0].lower() == column.lower():
                return True
    except Exception:
        pass
    return False

def _is_small_table(schema: dict, table: str) -> bool:
    try:
        table_key = _resolve_table_key(schema, table)
        return (schema.get(table_key, {}).get("row_estimate") or 0) < SMALL_TABLE_ROW_THRESHOLD
    except Exception:
        return False

def _filter_index_suggestions_with_guardrails(state: State, suggestions: list) -> tuple[list, list]:
    schema = state.get("db_schema", {}) or {}
    plan = (state.get("explain_plan") or "")
    # (SHOWPLAN_TEXT parsing stays as you wrote)
    seqscan_tables = set(m.group(1) for m in re.finditer(r"Table\s*=\s*([^\s,]+)", plan))

    filtered = []
    notes = []
    for s in suggestions or []:
        table = (s.get("table") or "").strip()
        cols = (s.get("columns") or [])[:]
        if not table or not cols:
            notes.append(f"skip: malformed suggestion {s}")
            continue

        table_key = _resolve_table_key(schema, table)
        rows_est = (schema.get(table_key, {}) or {}).get("row_estimate", None)

        # 1) small table guard — skip ONLY if we have a positive estimate below the threshold
        if isinstance(rows_est, (int, float)) and rows_est > 0 and rows_est < SMALL_TABLE_ROW_THRESHOLD:
            notes.append(f"skip: small table {table} (rows≈{rows_est})")
            continue

        # 2) leading-index guard
        first_col = cols[0]
        if _has_leading_index(schema, table, first_col):
            notes.append(f"skip: existing leading index on {table}({first_col})")
            continue

        # Optional: require that the plan references this table – you had this as optional, keep your choice

        # IMPORTANT: normalize the table name in the suggestion so CREATE uses the resolved key
        s["table"] = table_key
        filtered.append(s)

    return filtered, notes

# ---- EXPLAIN for MSSQL ----
def explain_for(query: str) -> str:
    try:
        q = (query or "").strip()
        if q.endswith(";"):
            q = q[:-1]
        # Use estimated plan; SHOWPLAN_TEXT returns textual plan without executing the query
        engine = db._engine
        with engine.connect() as conn:
            conn.exec_driver_sql("SET SHOWPLAN_TEXT ON;")
            rows = conn.exec_driver_sql(q).fetchall()
            conn.exec_driver_sql("SET SHOWPLAN_TEXT OFF;")
        # rows often look like tuples (StmtText,)
        lines = []
        for r in rows:
            if isinstance(r, (list, tuple)) and r:
                lines.append(str(r[0]))
            else:
                lines.append(str(r))
        return "\n".join(lines)
    except Exception as e:
        return f"(explain error: {e})"
    
    
def _drain_select_streaming(sql: str, chunk_size: int = 50_000) -> int:
    """
    Execute a SELECT and drain all rows in chunks to avoid OOM.
    Returns total rows drained. Does NOT keep rows in memory.
    """
    engine = db._engine
    total = 0
    # stream_results tells SQLAlchemy/pyodbc to avoid preloading everything
    with engine.connect().execution_options(stream_results=True) as conn:
        result = conn.exec_driver_sql(sql)
        # (Optional) hint the DBAPI fetch buffer
        try:
            if hasattr(result, "cursor") and hasattr(result.cursor, "arraysize"):
                result.cursor.arraysize = chunk_size
        except Exception:
            pass
        while True:
            rows = result.fetchmany(chunk_size)
            if not rows:
                break
            total += len(rows)
    return total

# ---- Execute ----
def sql_execute(state: State) -> State:
    """Execute query with real timing, draining results in chunks (no printing, no OOM)."""
    if not state or "query" not in state or not state["query"]:
        error_msg = "No query to execute"
        print(error_msg)
        return {**state, "error": error_msg}

    q = state["query"]
    print(f"\nExecuting SQL Query:\n{q}")

    try:
        if hasattr(db, '_cursor'):
            db._cursor = None

        # estimated plan for the original query (no execution)
        plan_text = explain_for(q)

        start_time = time.perf_counter()

        # Real execution:
        # - For SELECT: stream & drain in chunks so we measure true runtime without storing rows
        # - For non-SELECT: just execute (no resultset to drain)
        upper = q.lstrip().upper()
        row_count = None
        if upper.startswith("SELECT"):
            row_count = _drain_select_streaming(q)
        else:
            # use SQLAlchemy to execute without fetching a large result set
            engine = db._engine
            with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
                conn.exec_driver_sql(q)

        elapsed = time.perf_counter() - start_time
        print(f"\n⏱️ Query executed in {elapsed:.4f} seconds.")
        log_query(state["question"], q, elapsed)

        # Give the LLM a minimal, non-disclosing preview so it knows there were rows
        if row_count is not None:
            result_preview = f"Returned rows (preview omitted): {row_count}"
        else:
            result_preview = "Statement executed."

        return {
            **state,
            "result": result_preview,          # no actual rows stored/printed
            "formatted_result": None,
            "friendly_summary": None,
            "execution_time_seconds": elapsed, # REAL wall-clock time
            "explain_plan": plan_text,
            "error": None
        }

    except Exception as e:
        error_msg = f"Query execution failed: {str(e)}"
        print(error_msg)
        return {**state, "error": error_msg}

# ---- Result formatting (unchanged) ----
def format_sql_result(raw_result, query: str = "") -> str:
    if not raw_result or raw_result == "No results":
        return "No results found"
    if isinstance(raw_result, str):
        try:
            parsed = eval(raw_result)
            if isinstance(parsed, (list, tuple)):
                raw_result = parsed
        except:
            return raw_result
    if isinstance(raw_result, (list, tuple)):
        if not raw_result:
            return "No data found"
        if len(raw_result) == 1 and isinstance(raw_result[0], str) and raw_result[0].startswith('['):
            try:
                raw_result = eval(raw_result[0])
            except:
                pass
        column_names = []
        if hasattr(db, '_cursor') and db._cursor and hasattr(db._cursor, 'description') and db._cursor.description:
            column_names = [desc[0] for desc in db._cursor.description]
        output_lines = []
        if column_names:
            header = " | ".join(f"{name[:20]:<20}" for name in column_names)
            separator = "-" * len(header)
            output_lines.extend([header, separator])
        max_rows = 20
        for i, row in enumerate(raw_result[:max_rows]):
            if isinstance(row, (tuple, list)):
                formatted_cells = []
                for cell in row:
                    if cell is None:
                        cell_str = "NULL"
                    elif hasattr(cell, 'strftime'):
                        cell_str = cell.strftime("%Y-%m-%d %H:%M:%S")
                    else:
                        cell_str = str(cell)
                        if '[dados privados]' in cell_str.lower():
                            cell_str = '[private]'
                        elif len(cell_str) > 30:
                            cell_str = cell_str[:27] + "..."
                    formatted_cells.append(f"{cell_str[:30]:<30}")
                output_lines.append(" | ".join(formatted_cells))
            else:
                output_lines.append(str(row))
        summary = []
        if len(raw_result) > max_rows:
            summary.append(f"Showing first {max_rows} of {len(raw_result)} rows")
        elif len(raw_result) > 1:
            summary.append(f"Total rows: {len(raw_result)}")
        if query and "COUNT" in query.upper() and raw_result:
            try:
                summary.append(f"Total count: {raw_result[0][0]}")
            except:
                pass
        if summary:
            output_lines.append("\n" + "\n".join(summary))
        return "\n".join(output_lines)
    return str(raw_result)

MIN_IMPROVEMENT_RATIO = 0.10  # 10%

# ---- Index helpers (MSSQL) ----
CI_WITH_OPT_NAME_RE = re.compile(
    r"^\s*CREATE\s+INDEX\s+(?:IF\s+NOT\s+EXISTS\s+)?(?:([^\s(]+)\s+)?ON\s",
    re.IGNORECASE,
)
INDEX_NAME_RE = re.compile(
    r"CREATE\s+INDEX\s+(?:IF\s+NOT\s+EXISTS\s+)?([^\s(]+)",
    re.IGNORECASE,
)

def _extract_index_name(create_sql: str) -> Optional[str]:
    if not create_sql:
        return None
    m = INDEX_NAME_RE.search(create_sql)
    return m.group(1) if m else None

def _is_managed_index_name(name: str) -> bool:
    return bool(re.fullmatch(r"idx_[0-9a-f]{10}", (name or "")))

def _run_drop_index(index_name: str):
    """
    SQL Server requires table name to drop an index.
    We resolve it from sys.indexes, then build DROP INDEX <idx> ON <schema.table>.
    """
    tsql = f"""
    DECLARE @tbl nvarchar(300), @sql nvarchar(max);
    SELECT TOP(1)
        @tbl = QUOTENAME(OBJECT_SCHEMA_NAME(object_id)) + '.' + QUOTENAME(OBJECT_NAME(object_id))
    FROM sys.indexes
    WHERE name = N'{index_name}';
    IF @tbl IS NOT NULL
    BEGIN
        SET @sql = N'DROP INDEX ' + QUOTENAME(N'{index_name}') + N' ON ' + @tbl + N';';
        EXEC sp_executesql @sql;
    END
    """
    _run_ddl_autocommit(tsql)

def _ensure_named_index_sql(raw_sql: str) -> tuple[str, str, bool]:
    """
    MSSQL: ensure explicit index name and wrap with IF NOT EXISTS checking BOTH name and target table.
    Returns (tsql_batch, index_name, managed_flag).
    """
    s = (raw_sql or "").strip().rstrip(";")
    m = re.search(r"^\s*CREATE\s+INDEX\s+(?:IF\s+NOT\s+EXISTS\s+)?(?:([^\s(]+)\s+)?ON\s+(.*)$",
                  s, re.IGNORECASE)
    if not m:
        raise ValueError("Not a valid CREATE INDEX statement")

    existing_name = m.group(1)
    tail = m.group(2).strip()  # e.g. [dbo].[H_Customer] (c_nationkey) [INCLUDE ...] [WHERE ...]

    # Extract the table expression right after ON ... up to the first '('
    tm = re.match(r"([^\s(]+)\s*\(", tail)
    if not tm:
        raise ValueError("Could not parse table name after ON")
    table_expr = tm.group(1)  # could be H_Customer, [dbo].[H_Customer], dbo.H_Customer, etc.

    def _wrap(create_stmt: str, idx_name: str) -> str:
        return f"""
IF NOT EXISTS (
  SELECT 1
  FROM sys.indexes i
  WHERE i.name = N'{idx_name}'
    AND i.object_id = OBJECT_ID(N'{table_expr}')
)
BEGIN
  {create_stmt};
END
""".strip()

    if existing_name:
        name = existing_name
        create_stmt = f"CREATE INDEX {name} ON {tail}"
        return _wrap(create_stmt, name), name, False

    # synthesize stable managed name
    name = "idx_" + hashlib.md5(s.encode("utf-8")).hexdigest()[:10]
    create_stmt = f"CREATE INDEX {name} ON {tail}"
    return _wrap(create_stmt, name), name, True

def _safe_build_create_if_missing(sugg: dict) -> Optional[str]:
    table = (sugg.get("table") or "").strip()
    cols = sugg.get("columns") or []
    where = sugg.get("where")
    if not table or not cols:
        return None

    # Resolve/normalize table name from cached schema
    try:
        resolved = _resolve_table_key(get_db_schema(), table)
    except Exception:
        resolved = table

    # Bracket and qualify
    if "." in resolved:
        s, t = [p.strip('[]') for p in resolved.split('.', 1)]
        qualified = f"[{s}].[{t}]"
    else:
        qualified = f"[{resolved.strip('[]')}]"

    seed = f"{resolved}:{','.join(cols)}:{where or ''}"
    name = "idx_" + hashlib.md5(seed.encode("utf-8")).hexdigest()[:10]
    cols_sql = ", ".join(f"[{c}]" for c in cols)

    create_stmt = f"CREATE INDEX {name} ON {qualified} ({cols_sql})"
    if where:
        create_stmt += f" WHERE {where}"

    return f"""
IF NOT EXISTS (
  SELECT 1
  FROM sys.indexes i
  WHERE i.name = N'{name}'
    AND i.object_id = OBJECT_ID(N'{qualified}')
)
BEGIN
  {create_stmt};
END
""".strip()

def indexes_by_table(schema: dict) -> str:
    if not schema:
        return "(none)"
    lines = []
    for table, det in sorted(schema.items()):
        pk = det.get("primary_keys", []) or []
        idxs = det.get("indices", []) or []
        idx_desc = []
        for i in idxs:
            cols = i.get("columns") or []
            name = i.get("index_name")
            uniq = i.get("is_unique")
            prim = i.get("is_primary")
            flags = []
            if uniq: flags.append("unique")
            if prim: flags.append("pk")
            flag_str = f" [{', '.join(flags)}]" if flags else ""
            idx_desc.append(f"{name}({', '.join(cols)}){flag_str}")
        pk_str = f"PK({', '.join(pk)})" if pk else "PK(-)"
        idx_str = ", ".join(idx_desc) if idx_desc else "-"
        lines.append(f"- {table}: {pk_str}; Indexes: {idx_str}")
    return "\n".join(lines) if lines else "(none)"

FENCED_JSON_RE = re.compile(r"```json\s*(\{[\s\S]*?\})\s*```", re.IGNORECASE)
BARE_JSON_RE   = re.compile(r"(\{[\s\S]*\"index_suggestions\"[\s\S]*\})", re.IGNORECASE)

def extract_index_info(text: str):
    if not text:
        return [], ""
    m = FENCED_JSON_RE.search(text) or BARE_JSON_RE.search(text)
    if not m:
        return [], ""
    try:
        payload = json.loads(m.group(1))
        sugs = payload.get("index_suggestions", []) or []
        notes = payload.get("notes", "") or ""
        return sugs if isinstance(sugs, list) else [], str(notes)
    except Exception:
        return [], ""

SEQSCAN_RE = re.compile(r"Table\s*=\s*([^\s,]+)", re.IGNORECASE)  # SHOWPLAN_TEXT-ish
FILTER_COL_RE = re.compile(r"(\w+)\.(\w+)\s*=\s*'[^']*'|\b(\w+)\.(\w+)\s*IN\s*\(", re.IGNORECASE)

def heuristic_suggest_if_seqscan_big_equality(state: State) -> list:
    plan = state.get("explain_plan") or ""
    q = state.get("query") or ""
    if not plan or not q:
        return []
    tables = set(m.group(1) for m in SEQSCAN_RE.finditer(plan))
    if not tables:
        return []
    cols = []
    for m in FILTER_COL_RE.finditer(q):
        if m.group(1) and m.group(2):
            cols.append((m.group(1), m.group(2)))
        elif m.group(3) and m.group(4):
            cols.append((m.group(3), m.group(4)))
    sugs = []
    schema = state.get("db_schema", {})
    for t, c in cols:
        t_norm = t
        if t_norm in tables and t_norm in schema:
            has_any = any(
                (i.get("columns") or []) and c in i.get("columns")
                for i in (schema[t_norm].get("indices") or [])
            )
            if not has_any:
                sugs.append({
                    "table": t_norm,
                    "columns": [c],
                    "where": None,
                    "create_sql": f"CREATE INDEX ON {t_norm} ({c})",  # will be normalized
                    "rationale": "Fallback: table scan on equality and no index detected"
                })
    return sugs

# ---- Answer + index suggestion (unchanged logic, MSSQL-safe) ----
def generate_answer(state: State) -> State:
    if "error" in state and state["error"]:
        return {**state, "answer": state["error"]}
    try:
        raw_result = state.get("result", "No results")
        formatted = format_sql_result(raw_result, state.get("query", ""))

        idx_text = indexes_by_table(state.get("db_schema", {}))
        exec_time = state.get("execution_time_seconds", "NA")

        prompt = (
           "You are a senior database performance engineer and data analyst.\n"
           "First, write a concise 1–2 sentence natural-language answer to the user based on the query results.\n\n"
           f"Original Question: {state['question']}\n"
           f"SQL Query Executed:\n{state.get('query', 'No query')}\n"
           f"Execution Time (seconds): {exec_time}\n"
           "- Current Indexes (by table):\n"
           f"{idx_text}\n"
           "- (Optional) EXPLAIN/plan:\n"
           f"{state.get('explain_plan', '(not collected)')}\n"
           "- Query Results (formatted preview):\n"
           f"{formatted}\n\n"
           "POLICY FOR INDEX SUGGESTIONS:\n"
           "- If the plan shows a scan on a filtered equality column AND the table is large (≥ 50k rows), suggest an index even if runtime < 0.2s.\n"
           "- Suggest index creation ONLY if it would likely improve this query's performance.\n"
           "- Prefer equality/IN columns first, then at most one range column.\n"
           "- Avoid duplicates of existing indexes; skip tiny tables.\n"
           "- Up to 2 suggestions. If none are warranted, return an empty list.\n\n"
           "OUTPUT:\n"
           "1) First, the natural-language answer paragraph.\n"
           "2) Then, on a new line, output a JSON code block:\n"
           "```json\n"
           "{\n"
           '  "index_suggestions": [\n'
           '    {\n'
           '      "table": "string",\n'
           '      "columns": ["col1", "col2"],\n'
           '      "where": "optional partial predicate or null",\n'
           '      "create_sql": "CREATE INDEX ...",\n'
           '      "rationale": "short reason tied to evidence"\n'
           "    }\n"
           "  ],\n"
           '  "notes": "If index_suggestions is empty, briefly explain why (e.g., small table, existing index, low runtime, or plan already uses index)."\n'
           "}\n"
           "```\n"
           'If no suggestions, still return a valid JSON object exactly in this format:\n'
           '{ "index_suggestions": [], "notes": "your reason here" }\n'
        )

        response = llm.invoke(prompt)
        final_answer = response.content if response else formatted

        suggestions, notes = extract_index_info(final_answer)
        if suggestions:
            print("\n[INDEX SUGGESTIONS]")
            for s in suggestions:
                print(f"- {s.get('create_sql') or ''}  # {s.get('rationale') or ''}")
        print(f"[DEBUG/generate_answer] returning suggestions: {len(suggestions)}")
        return {**state, "answer": final_answer, "index_suggestions": suggestions, "index_notes": notes}

    except Exception as e:
        error_msg = f"Failed to generate answer: {str(e)}"
        print(error_msg)
        return {**state, "answer": error_msg, "index_suggestions": []}

# Reusable regex already defined above
# _extract_index_name, _is_managed_index_name, _run_drop_index present above

def apply_indexes_and_reexecute(state: State) -> State:
    before_elapsed = state.get("execution_time_seconds")
    improvement_pct = None
    dropped = []
    cleanup_reason = None
    
    try:
        suggestions = state.get("index_suggestions") or []
        if not suggestions:
            return state

        created = []
        for s in suggestions:
            raw = (s.get("create_sql") or "").strip()
            if raw:
                try:
                    sql, idx_name, managed = _ensure_named_index_sql(raw)
                except Exception:
                    sql = _safe_build_create_if_missing(s)
                    idx_name = _extract_index_name(sql) if sql else None
                    managed = bool(idx_name and _is_managed_index_name(idx_name))
            else:
                sql = _safe_build_create_if_missing(s)
                idx_name = _extract_index_name(sql) if sql else None
                managed = bool(idx_name and _is_managed_index_name(idx_name))

            if not sql or not idx_name:
                created.append({
                    "sql": sql, "index_name": idx_name, "status": "skipped",
                    "error": "missing/invalid CREATE INDEX SQL"
                })
                continue

            try:
                print(f"\n[INDEX CREATE] {sql}")
                _run_ddl_autocommit(sql)
                created.append({
                    "sql": sql,
                    "index_name": idx_name,
                    "managed": managed,
                    "status": "created",
                    "rationale": s.get("rationale")
                })
            except Exception as e:
                created.append({
                    "sql": sql,
                    "index_name": idx_name,
                    "managed": managed,
                    "status": "error",
                    "error": str(e),
                    "rationale": s.get("rationale")
                })

        # Refresh schema
        get_db_schema.cache_clear()
        new_schema = get_db_schema()

        # Re-run query to measure impact (usar o MESMO método do "antes": streaming)
        after_elapsed = None
        if state.get("query"):
            q = state["query"]
            if hasattr(db, "_cursor"):
                db._cursor = None

            start = time.perf_counter()
            upper = q.lstrip().upper()

            if upper.startswith("SELECT"):
                # drena em chunks, sem materializar resultado em memória
                _ = _drain_select_streaming(q)
            else:
                # comandos não-SELECT (DDL/DML): executa em autocommit
                engine = db._engine
                with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
                    conn.exec_driver_sql(q)

            after_elapsed = time.perf_counter() - start
            print(f"\n⏱️ Re-executed with indexes in {after_elapsed:.4f} seconds.")

        if (
            isinstance(before_elapsed, (int, float)) and before_elapsed > 0 and
            isinstance(after_elapsed, (int, float)) and after_elapsed > 0
        ):
            improvement = (before_elapsed - after_elapsed) / before_elapsed
            improvement_pct = improvement * 100.0

            if improvement < MIN_IMPROVEMENT_RATIO:
                cleanup_reason = (
                    f"Speedup {improvement_pct:.2f}% < {MIN_IMPROVEMENT_RATIO*100:.0f}% "
                    f"→ dropping indexes created in this run."
                )
                for c in created:
                    if c.get("status") == "created":
                        name = c.get("index_name")
                        if not name:
                            name = _extract_index_name(c.get("sql") or "")
                        if name:
                            try:
                                _run_drop_index(name)
                                dropped.append({"index_name": name, "status": "dropped"})
                            except Exception as e:
                                dropped.append({"index_name": name, "status": "error", "error": str(e)})

                # Refresh schema again after drop
                get_db_schema.cache_clear()
                new_schema = get_db_schema()

        return {
            **state,
            "db_schema": new_schema,
            "created_indexes": created,
            "execution_time_seconds_after_index": after_elapsed,
            "runtime_improvement_pct": improvement_pct,
            "dropped_indexes": dropped,
            "cleanup_reason": cleanup_reason,
        }

    except Exception as e:
        print(f"[WARN] apply_indexes_and_reexecute failed: {e}")
        return {**state, "created_indexes": [{"status": "error", "error": str(e)}]}

# ---- Workflow (unchanged) ----
workflow = Graph()
workflow.add_node("generate_query", sql_query)
workflow.add_node("execute_query", sql_execute)
workflow.add_node("generate_answer", generate_answer)

workflow.add_edge("generate_query", "execute_query")
workflow.add_edge("execute_query", "generate_answer")

workflow.set_entry_point("generate_query")
workflow.set_finish_point("generate_answer")
app = workflow.compile()

def run_agent(question, few_args, state=None):
    try:
        if not state:
            state = {
                "question": question,
                "few_args": few_args,
                "db_schema": get_db_schema()
            }
        else:
            state["question"] = question
            state["few_args"] = few_args
            state["db_schema"] = get_db_schema()  # Refresh schema

        result = app.invoke(state) or state
        print(f"[DEBUG/run_agent] got suggestions: {len((result or {}).get('index_suggestions') or [])}")

        print(f"[DEBUG] Suggestions count: {len(result.get('index_suggestions') or [])}")
        if result.get("index_suggestions"):
            filtered, guardrail_notes = _filter_index_suggestions_with_guardrails(result, result["index_suggestions"])
            if guardrail_notes:
                extra = (result.get("index_notes") or "").strip()
                joined = ("; ".join(guardrail_notes))
                result["index_notes"] = f"{extra + ' | ' if extra else ''}{joined}"
            result["index_suggestions"] = filtered

            if not filtered:
                return result

            before = result.get("execution_time_seconds")
            result = apply_indexes_and_reexecute(result)

            after = result.get("execution_time_seconds_after_index")
            if isinstance(before, (int, float)) and isinstance(after, (int, float)) and after > 0:
                result["runtime_delta_seconds"] = after - before
                result["runtime_speedup_x"] = before / after

        return result

    except Exception as e:
        print(f"[ERROR] Workflow execution failed: {e}")
        return state or {"question": question, "error": str(e)}

# ---- Main (unchanged shape) ----
if __name__ == "__main__":
    print("📊 Database SQL Agent (type 'exit' to quit, 'refresh schema' to reload schema)")
    print(f"✅ Connected to database: {db_database} in {dialect_name}")
    retriever = SQLQueryRetriever(model_name='all-MiniLM-L6-v2')
    retriever.carregar_json('dados.json')
    retriever.construir_index()

    agent_state = {
        "db_schema": get_db_schema()
    }

    while True:
        question = input("\nEnter your question: ").strip()
        few_args = retriever.buscar_filtrado(question, top_k=2, threshold=0.5)

        if question.lower() in ['exit', 'quit']:
            print("👋 Goodbye!")
            break

        if question.lower() == 'refresh schema':
            get_db_schema.cache_clear()
            agent_state["db_schema"] = get_db_schema()
            print("[INFO]  Schema cache cleared and reloaded.")
            continue

        try:
            agent_state = run_agent(question, few_args, agent_state)

            if not isinstance(agent_state, dict):
                print("[WARNING] Agent returned invalid state. Resetting.")
                agent_state = {"db_schema": get_db_schema()}
                continue

            if agent_state.get("answer"):
                print("\n Answer:\n" + agent_state["answer"])

            created = agent_state.get("created_indexes") or []
            if created:
                print("\n[INDEX CREATE RESULTS]")
                for c in created:
                    status = c.get("status")
                    sql    = (c.get("sql") or "").strip()
                    info   = c.get("error") or c.get("rationale") or ""
                    print(f"- {status}: {sql}{(' // ' + info) if info else ''}")

                before = agent_state.get("execution_time_seconds")
                after  = agent_state.get("execution_time_seconds_after_index")
                impr   = agent_state.get("runtime_improvement_pct")

                if isinstance(before, (int, float)) and isinstance(after, (int, float)):
                    delta   = after - before
                    speedup = (before / after) if after > 0 else None
                    if impr is None and before > 0:
                        impr = (before - after) / before * 100.0
                        agent_state["runtime_improvement_pct"] = impr

                    print("\n Runtime:")
                    print(f"- Before: {before:.4f} s")
                    print(f"- After : {after:.4f} s")
                    print(f"- Delta : {delta:+.4f} s")
                    if speedup:
                        print(f"- Speedup: {speedup:.2f}×")
                    if impr is not None:
                        print(f"- Improvement: {impr:.2f}%")

                dropped = agent_state.get("dropped_indexes") or []
                if dropped:
                    print("\n[INDEX CLEANUP]")
                    if agent_state.get("cleanup_reason"):
                        print(f"- Reason: {agent_state['cleanup_reason']}")
                    for d in dropped:
                        if d.get("status") == "dropped":
                            print(f"- dropped: {d['index_name']}")
                        else:
                            print(f"- drop error: {d.get('index_name')} // {d.get('error')}")

            else:
                print("\n[INDEX SUGGESTIONS] none — ask another question.")
                if agent_state.get("index_notes"):
                    print(f"[WHY NONE] {agent_state['index_notes']}")

        except Exception as e:
            print(f"[FATAL ERROR] Unexpected failure: {e}")
            agent_state = {"db_schema": get_db_schema()}
