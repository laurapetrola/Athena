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
from sqlalchemy.dialects.postgresql.base import ischema_names
from sqlalchemy.types import UserDefinedType
from sqlalchemy import text

from close_query_select import SQLQueryRetriever


os.environ["LANG"] = "C.UTF-8"
os.environ["PGCLIENTENCODING"] = "utf-8"

# Treating geocoord types
class GeoCoord(UserDefinedType):
    def __init__(self, *args, **kwargs):
        pass 

    def get_col_spec(self, **kw):
        return "geometry"

    def bind_processor(self, dialect):
        return None

    def result_processor(self, dialect, coltype):
        return None
    
ischema_names["geometry"] = GeoCoord
ischema_names["geography"] = GeoCoord
ischema_names["point"] = GeoCoord

#State
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

# Loading .env variables for database initialization
load_dotenv(override=True)
db_host = os.getenv("DB_HOST")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_database = os.getenv("DB_DATABASE")
db_port = os.getenv("DB_PORT")
DATABASE_URI = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_database}"
print(f"[DEBUG] Connecting to DB: {DATABASE_URI}")

# Initializing Database
db = SQLDatabase.from_uri(
    DATABASE_URI,
    include_tables=None,  
    sample_rows_in_table_info=2,  
    view_support=True  
)

dialect_name = db._engine.dialect.name

def _run_ddl_autocommit(sql: str):
    """Run DDL with AUTOCOMMIT so CREATE INDEX CONCURRENTLY succeeds."""
    engine = db._engine  # underlying SQLAlchemy engine from SQLDatabase
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(text(sql))

# Initialize DeepSeek
llm = ChatOpenAI(
    openai_api_base="https://api.deepseek.com/v1",
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    model="deepseek-coder",
    temperature=0.0 
)

# Function used for debugging the generated SQLs.
def log_query(question: str, query: str, execution_time: float, log_path: str = "query_log.json"):
    """Append query execution details to a JSON file."""
    timestamp = datetime.now().isoformat()

    log_entry = {
        "timestamp": timestamp,
        "question": question,
        "query": query,
        "execution_time_seconds": round(execution_time, 4)
    }

    # Load existing log file or initialize new list
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


# Collect and structure advanced metadata from the database.
@lru_cache(maxsize=1)
def get_db_schema() -> Dict:
    """Get PostgreSQL schema with enhanced information (cached).
    Adds: per-table row_estimate (pg_class.reltuples) and ordered index columns.
    Robustly parses driver-returned values that may come as strings/lists.
    """
    print("Loading schema from database...")
    tables = db.get_usable_table_names()
    schema: Dict = {}

    import ast
    import re
    from decimal import Decimal

    def _coerce_number(val):
        """Coerce weird driver values to int safely."""
        if val is None:
            return 0
        # Already numeric?
        if isinstance(val, (int,)):
            return int(val)
        if isinstance(val, float):
            return int(val)
        if isinstance(val, Decimal):
            return int(val)
        # If it's a tuple/list with one element, unwrap
        if isinstance(val, (list, tuple)) and val:
            return _coerce_number(val[0])
        # If it's a string, try several strategies
        if isinstance(val, str):
            s = val.strip()
            # If it looks like a Python list/tuple literal, try literal_eval then recurse
            if (s.startswith('[') and s.endswith(']')) or (s.startswith('(') and s.endswith(')')):
                try:
                    parsed = ast.literal_eval(s)
                    return _coerce_number(parsed)
                except Exception:
                    pass
            # Extract first numeric token
            m = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', s)
            if m:
                tok = m.group(0)
                try:
                    if '.' in tok or 'e' in tok.lower():
                        return int(float(tok))
                    return int(tok)
                except Exception:
                    return 0
            return 0
        # Fallback
        return 0

    for table in tables:
        try:
            # --- Columns ---
            columns = db.run(f"""
                SELECT 
                    column_name, 
                    udt_name AS data_type,
                    is_nullable,
                    character_maximum_length
                FROM information_schema.columns
                WHERE table_name = '{table}'
                ORDER BY ordinal_position;
            """)

            # --- Primary keys ---
            pk_result = db.run(f"""
                SELECT kcu.column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                  ON tc.constraint_name = kcu.constraint_name
                WHERE tc.table_name = '{table}'
                  AND tc.constraint_type = 'PRIMARY KEY';
            """)
            primary_keys = [row[0] for row in (pk_result or []) if isinstance(row, (list, tuple)) and row] or []

            # --- Foreign keys ---
            fk_result = db.run(f"""
                SELECT
                    kcu.column_name,
                    ccu.table_name AS foreign_table,
                    ccu.column_name AS foreign_column
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                  ON tc.constraint_name = kcu.constraint_name
                JOIN information_schema.constraint_column_usage ccu
                  ON ccu.constraint_name = tc.constraint_name
                WHERE tc.table_name = '{table}'
                  AND tc.constraint_type = 'FOREIGN KEY';
            """)
            foreign_keys = {}
            for row in (fk_result or []):
                if isinstance(row, (list, tuple)) and len(row) >= 3:
                    foreign_keys[row[0]] = {"foreign_table": row[1], "foreign_column": row[2]}

            # --- Table row estimate (pg_class.reltuples) ---
            row_estimate_result = db.run(f"""
                SELECT c.reltuples
                FROM pg_class c
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE n.nspname = current_schema()
                  AND c.relname = '{table}';
            """)
            # Safely coerce whatever came back to an int
            first_cell = None
            if row_estimate_result:
                first_row = row_estimate_result[0]
                first_cell = first_row[0] if isinstance(first_row, (list, tuple)) and first_row else first_row
            row_estimate = _coerce_number(first_cell)

            # --- Indexes with ORDERED columns ---
            index_result = db.run(f"""
                WITH idx AS (
                  SELECT
                    i.relname       AS index_name,
                    ix.indisunique  AS is_unique,
                    ix.indisprimary AS is_primary,
                    ix.indexrelid,
                    ix.indrelid,
                    ix.indkey
                  FROM pg_index ix
                  JOIN pg_class i ON i.oid = ix.indexrelid
                  JOIN pg_class t ON t.oid = ix.indrelid
                  WHERE t.relname = '{table}'
                )
                SELECT
                  idx.index_name,
                  idx.is_unique,
                  idx.is_primary,
                  a.attname   AS col_name,
                  ord.n       AS col_position
                FROM idx
                CROSS JOIN LATERAL unnest(idx.indkey) WITH ORDINALITY AS ord(attnum, n)
                JOIN pg_attribute a ON a.attrelid = idx.indrelid AND a.attnum = ord.attnum
                ORDER BY idx.index_name, ord.n;
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
                    "columns": []  # ordered
                })
                meta["columns"].append(col_name)

            indexes = list(idx_by_name.values())

            # --- Compose schema entry ---
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
                        "nullable": (isnul == 'YES') if isinstance(isnul, str) else bool(isnul),
                        "max_length": maxlen,
                        "primary_key": name in primary_keys,
                        "foreign_key": foreign_keys.get(name)
                    }

                schema[table] = {
                    "columns": cols_map,
                    "description": f"Table {table}",
                    "primary_keys": primary_keys,
                    "foreign_keys": foreign_keys,
                    "indices": indexes,            # ordered columns preserved
                    "row_estimate": row_estimate,  # robustly parsed
                }

        except Exception as e:
            print(f"[ERROR] Failed to fetch schema for table '{table}': {e}")
            continue

    return schema

# Summarizes the structure of the schema into a string.
def generate_schema_fingerprint(schema: Dict) -> str:
    """Create a stable fingerprint of the schema state"""
    if not schema:
        return ""
    
    fingerprint_parts = []
    for table, details in sorted(schema.items()):
        columns_part = ",".join(sorted(details["columns"].keys()))
        pks_part = ",".join(sorted(details.get("primary_keys", [])))
        fingerprint_parts.append(f"{table}:{columns_part}:{pks_part}")
    
    return "|".join(fingerprint_parts)

# This is the template for the LLM to generate the SQL.
query_prompt_template = PromptTemplate(
    input_variables=["dialect", "table_info", "input", "schema_info", "few_args"],
    template=(
        "You are a PostgreSQL expert. Given this schema and question, generate an accurate SQL query.\n\n"
        "Database Schema:\n{table_info}\n\n"
        "Detailed Structure:\n{schema_info}\n\n"
        "Follow these rules:\n"
        "{few_args}\n"
        
        #Lauras
        # "- After the initial thought above, apply the following heuristics: \n"
        # "- When a WHERE clause filters on R.A, and there's a join R.A = S.B, you must also add the same filter to S.B in the WHERE clause. Even if redundant. Always apply this. For example, before: SELECT * FROM R JOIN S ON R.id = S.id WHERE R.id = 123; and after: SELECT * FROM R JOIN S ON R.id = S.id WHERE R.id = 123 AND S.id = 123; \n"
        # "- If the SQL contains col IN (val1, val2, ...), rewrite as multiple col = valX conditions joined by OR. \n"
        # "- Avoid correlated subqueries. Prefer Common Table Expressions (CTEs) or JOINs for filtering based on per-group logic (e.g., MAX per group), especially when computing additional fields.\n"
        
        # Arlinos
        # "- Remove unnecessary GROUP BY clauses. If there is no HAVING clause and the SELECT contains only one aggregate function or no aggregates and the GROUP BY attribute is a primary key (from a single table or not used as a foreign key), the GROUP BY can be eliminated to reduce query cost.\n"
        # "- Change query with disjunction in the WHERE to a union of query results.\n"
        # "- Remove ALL operation with greater/less-than comparison operators by including a MAX or MIN aggregate function in the subquery.\n"
        # "- Remove SOME/ANY operation with greater/less-than comparison operators by including a MAX or MIN aggregate function in the subquery.\n"
        # "- Replace IN set operation by a join operation.\n"
        # "- Eliminate DISTINCT if the SELECTed columns come from the primary table and the JOIN key is a primary or unique key.\n"
        # "- Move function applied to a column index to another position in the expression. For example, DO NOT DO THIS: SELECT * FROM h_lineitem WHERE l_quantity::text = '28'; do: SELECT * FROM h_lineitem WHERE l_quantity = CAST('28' AS NUMERIC); \n"
        # "- Move arithmetic expression applied to a column index to another position in the expression.\n"
        "\nQuestion: {input}\n\n"
        "Return ONLY the SQL query in ```sql``` blocks."
    ),
)


def extract_sql_query(text):
    """Robust SQL extraction with PostgreSQL validation"""
    if not text:
        return None
    
    # Try to find code blocks first
    matches = re.findall(r"```(?:sql)?\n(.*?)```", text, re.DOTALL)
    if matches:
        query = matches[0].strip()
        if query.lower().startswith(('select', 'insert', 'update', 'delete', 'with')):
            return query
    
    # Fallback to finding standalone queries
    match = re.search(
        r"\b(?:SELECT|INSERT|UPDATE|DELETE|WITH).*?(?:;|$)", 
        text, 
        re.DOTALL | re.IGNORECASE
    )
    return match.group(0).strip() if match else None

# Extracts only the SQL query, without mixing with natural language.
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
            "dialect": "postgresql",
            "table_info": db.get_table_info(),
            "input": state["question"],
            "schema_info": "\n\n".join(schema_info),
            "few_args": state["few_args"]
        })

        response = llm.invoke(prompt)
        # print(f"\nLLM Response:\n{response.content[:500]}...")

        query = extract_sql_query(response.content)
        if not query:
            return {**state, "error": "Failed to extract valid SQL query"}

        if ";" not in query:
            query += ";"

        return {**state, "query": query}

    except Exception as e:
        print(f"Query generation failed: {e}")
        return {**state, "error": f"Query generation failed: {str(e)}"}
    
SMALL_TABLE_ROW_THRESHOLD = 50_000  # tune if needed

def _has_leading_index(schema: dict, table: str, column: str) -> bool:
    """True if any index on `table` has `column` as its leading (first) column."""
    try:
        for idx in (schema.get(table, {}).get("indices") or []):
            cols = idx.get("columns") or []
            if cols and cols[0] == column:
                return True
    except Exception:
        pass
    return False

def _is_small_table(schema: dict, table: str) -> bool:
    try:
        est = (schema.get(table, {}).get("row_estimate"))
        return isinstance(est, (int, float)) and est > 0 and est < SMALL_TABLE_ROW_THRESHOLD
    except Exception:
        return False

def _filter_index_suggestions_with_guardrails(state: State, suggestions: list) -> tuple[list, list]:
    """
    Apply guardrails:
      - drop suggestions for tiny tables
      - drop suggestions if a usable leading index already exists
      - (optional) drop suggestions if EXPLAIN doesn't show a Seq Scan on that table
    Returns (filtered_suggestions, notes)
    """
    schema = state.get("db_schema", {}) or {}
    plan = (state.get("explain_plan") or "")
    # collect seq-scan tables from the plan (optional guardrail)
    seqscan_tables = set(m.group(1) for m in SEQSCAN_RE.finditer(plan))

    filtered = []
    notes = []
    for s in suggestions or []:
        table = (s.get("table") or "").strip()
        cols = (s.get("columns") or [])[:]
        if not table or not cols:
            notes.append(f"skip: malformed suggestion {s}")
            continue

        # 1) Skip tiny tables (e.g., h_nation)
        if _is_small_table(schema, table):
            notes.append(f"skip: small table {table} (rows≈{schema.get(table,{}).get('row_estimate',0)})")
            continue

        # 2) Skip if leading index already exists on the first suggested column
        first_col = cols[0]
        if _has_leading_index(schema, table, first_col):
            notes.append(f"skip: existing leading index on {table}({first_col})")
            continue

        # 3) (Optional) Require Seq Scan on that table in the current plan
        # If you want this strict, uncomment the block below:
        # if table not in seqscan_tables:
        #     notes.append(f"skip: plan is not Seq Scan on {table}")
        #     continue

        filtered.append(s)

    return filtered, notes
        
def explain_for(query: str) -> str:
    try:
        q = (query or "").strip()
        if q.endswith(";"):
            q = q[:-1]
        rows = db.run("EXPLAIN " + q + ";")
        return "\n".join(r[0] if isinstance(r, (list, tuple)) else str(r) for r in rows)
    except Exception as e:
        return f"(explain error: {e})"
    
def _drain_select_streaming_pg(sql: str, chunk_size: int = 50_000) -> int:
    """
    Execute a SELECT and drain all rows in chunks to avoid OOM.
    Returns the total number of rows drained. Does not materialize the data in memory.
    For psycopg2, stream_results=True enables a server-side cursor.
    """
    engine = db._engine
    total = 0
    with engine.connect().execution_options(stream_results=True) as conn:
        result = conn.exec_driver_sql(sql)
        # Para psycopg2, itersize controla o "batch" do servidor
        try:
            cur = getattr(result, "cursor", None)
            if cur is not None and hasattr(cur, "itersize"):
                cur.itersize = chunk_size
        except Exception:
            pass

        while True:
            rows = result.fetchmany(chunk_size)
            if not rows:
                break
            total += len(rows)
    return total
            
# Executing the SQL generated in the step above.
def sql_execute(state: State) -> State:
    """Execute the query while timing real wall-clock duration, draining results (without printing), and generating a preview."""
    if not state or "query" not in state or not state["query"]:
        error_msg = "No query to execute"
        print(error_msg)
        return {**state, "error": error_msg}

    q = state["query"]
    print(f"\nExecuting SQL Query:\n{q}")

    try:
        if hasattr(db, '_cursor'):
            db._cursor = None

        # Captura o plano estimado (ajuda o LLM a sugerir índices)
        plan_text = explain_for(q)

        start_time = time.perf_counter()

        upper = q.lstrip().upper()
        row_count = None

        if upper.startswith("SELECT"):
            # Drena em streaming (server-side cursor) e conta linhas
            row_count = _drain_select_streaming_pg(q)
        else:
            # Para DDL/DML/etc., basta executar; sem materializar result set
            engine = db._engine
            with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
                conn.exec_driver_sql(q)

        elapsed = time.perf_counter() - start_time
        print(f"\n⏱️ Query executed in {elapsed:.4f} seconds.")
        log_query(state["question"], q, elapsed)

        # Preview amigável para o LLM (não exibe linhas reais)
        if row_count is not None:
            result_preview = f"Returned rows (preview omitted): {row_count}"
        else:
            result_preview = "Statement executed."

        return {
            **state,
            "result": result_preview,          # <-- string curta, NÃO as linhas
            "formatted_result": None,
            "friendly_summary": None,
            "execution_time_seconds": elapsed, # <-- tempo real
            "explain_plan": plan_text,
            "error": None
        }

    except Exception as e:
        error_msg = f"Query execution failed: {str(e)}"
        print(error_msg)
        return {**state, "error": error_msg}

def format_sql_result(raw_result, query: str = "") -> str:
    """Formats raw database results into a clean, readable table"""
    if not raw_result or raw_result == "No results":
        return "No results found"
    
    if isinstance(raw_result, str):
        try:
            # Try to parse string representation of list
            parsed = eval(raw_result)
            if isinstance(parsed, (list, tuple)):
                raw_result = parsed
        except:
            return raw_result
    
    if isinstance(raw_result, (list, tuple)):
        if not raw_result:
            return "No data found"
        
        # Handle case where raw_result is a string representation of a list
        if len(raw_result) == 1 and isinstance(raw_result[0], str) and raw_result[0].startswith('['):
            try:
                raw_result = eval(raw_result[0])
            except:
                pass
        
        # Get column names if available
        column_names = []
        if hasattr(db, '_cursor') and db._cursor and hasattr(db._cursor, 'description') and db._cursor.description:
            column_names = [desc[0] for desc in db._cursor.description]
        
        # Build output lines
        output_lines = []
        
        # Add header if we have column names
        if column_names:
            header = " | ".join(f"{name[:20]:<20}" for name in column_names)
            separator = "-" * len(header)
            output_lines.extend([header, separator])
        
        # Add rows with proper formatting
        max_rows = 20  # Limit rows for display
        for i, row in enumerate(raw_result[:max_rows]):
            if isinstance(row, (tuple, list)):
                formatted_cells = []
                for cell in row:
                    if cell is None:
                        cell_str = "NULL"
                    elif hasattr(cell, 'strftime'):  # Check if it's a datetime object
                        cell_str = cell.strftime("%Y-%m-%d %H:%M:%S")
                    else:
                        cell_str = str(cell)
                        # Handle sensitive data
                        if '[dados privados]' in cell_str.lower():
                            cell_str = '[private]'
                        elif len(cell_str) > 30:
                            cell_str = cell_str[:27] + "..."
                    formatted_cells.append(f"{cell_str[:30]:<30}")
                output_lines.append(" | ".join(formatted_cells))
            else:
                output_lines.append(str(row))
        
        # Add summary
        summary = []
        if len(raw_result) > max_rows:
            summary.append(f"Showing first {max_rows} of {len(raw_result)} rows")
        elif len(raw_result) > 1:
            summary.append(f"Total rows: {len(raw_result)}")
        
        # Add count summary if this was a count query
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

# Matches: CREATE INDEX [CONCURRENTLY] [IF NOT EXISTS] [name] ON ...
CI_WITH_OPT_NAME_RE = re.compile(
    r"^\s*CREATE\s+INDEX\s+(?:CONCURRENTLY\s+)?(?:IF\s+NOT\s+EXISTS\s+)?(?:([^\s(]+)\s+)?ON\s",
    re.IGNORECASE,
)

INDEX_NAME_RE = re.compile(
    r"CREATE\s+INDEX\s+(?:CONCURRENTLY\s+)?(?:IF\s+NOT\s+EXISTS\s+)?([^\s(]+)",
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
    _run_ddl_autocommit(f"DROP INDEX CONCURRENTLY IF EXISTS {index_name};")

def _ensure_named_index_sql(raw_sql: str) -> tuple[str, str, bool]:
    """
    Ensure CREATE INDEX has: explicit name + CONCURRENTLY + IF NOT EXISTS.
    Returns (normalized_sql, index_name, managed_flag).
    managed=True when we synthesized a md5-based name (idx_XXXXXXXXXX).
    """
    s = (raw_sql or "").strip().rstrip(";")
    m = CI_WITH_OPT_NAME_RE.search(s)
    if not m:
        raise ValueError("Not a valid CREATE INDEX statement")
    existing_name = m.group(1)
    if existing_name:
        # Normalize flags and keep the same name
        name = existing_name
        prefix = f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {name} ON "
        # m ends right after 'ON '
        tail = s[m.end():]
        sql = prefix + tail
        return sql + ";", name, False
    else:
        # Synthesize a stable, “managed” name from the raw statement
        name = "idx_" + hashlib.md5(s.encode("utf-8")).hexdigest()[:10]
        prefix = f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {name} ON "
        tail = s[m.end():]
        sql = prefix + tail
        return sql + ";", name, True

def indexes_by_table(schema: dict) -> str:
    if not schema:
        return "(none)"
    lines = []
    for table, det in sorted(schema.items()):
        pk = det.get("primary_keys", []) or []
        idxs = det.get("indices", []) or []
        idx_desc = []
        for i in idxs:
            cols = i.get("columns") or []  # <-- use "columns" (list)
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

SEQSCAN_RE = re.compile(r"Seq Scan on (\w+)", re.IGNORECASE)
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
        if t in tables and t in schema:
            # consider existing indexes; prefer “leading column”, but accept any match
            has_any = any(
                (i.get("columns") or []) and c in i.get("columns")
                for i in (schema[t].get("indices") or [])
            )
            if not has_any:
                sugs.append({
                    "table": t,
                    "columns": [c],
                    "where": None,
                    "create_sql": f"CREATE INDEX ON {t} ({c})",
                    "rationale": "Seq Scan on filtered equality and no index detected"
                })
    return sugs
    
def generate_answer(state: State) -> State:
    """Gera a resposta em linguagem natural e coleta sugestões de índice da LLM."""
    if "error" in state and state["error"]:
        return {**state, "answer": state["error"]}

    try:
        raw_result = state.get("result", "No results")

        # --- formatação existente (se você quiser manter) ---
        formatted = format_sql_result(raw_result, state.get("query", ""))

        # --- insumos para o prompt ---
        idx_text = indexes_by_table(state.get("db_schema", {}))
        exec_time = state.get("execution_time_seconds", "NA")
        plan_text = state.get("explain_plan", "(not collected)")  # opcional: você pode popular antes

        # --- PROMPT COM SUGESTÃO DE ÍNDICE ---
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
           "- If EXPLAIN shows Seq Scan on a filtered equality column AND the table is large (≥ 50k rows), suggest an index even if runtime < 0.2s.\n"
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
    
INDEX_NAME_RE = re.compile(
    r"CREATE\s+INDEX\s+(?:CONCURRENTLY\s+)?(?:IF\s+NOT\s+EXISTS\s+)?([^\s(]+)",
    re.IGNORECASE,
)

def _extract_index_name(create_sql: str) -> Optional[str]:
    if not create_sql:
        return None
    m = INDEX_NAME_RE.search(create_sql)
    return m.group(1) if m else None

def _is_managed_index_name(name: str) -> bool:
    # We only auto-drop indexes we created with our md5 scheme
    # e.g., idx_0a1b2c3d4e
    return bool(re.fullmatch(r"idx_[0-9a-f]{10}", (name or "")))

def _run_drop_index(index_name: str):
    # Use CONCURRENTLY and IF EXISTS, like creation
    _run_ddl_autocommit(f"DROP INDEX CONCURRENTLY IF EXISTS {index_name};")
    
def _normalize_create_index_sql_postgres(create_sql: str) -> str:
    s = create_sql.strip().rstrip(";")
    s = re.sub(r"^\s*CREATE\s+INDEX\s+", "CREATE INDEX ", s, flags=re.IGNORECASE)
    if " CONCURRENTLY " not in s.upper():
        s = s.replace("CREATE INDEX ", "CREATE INDEX CONCURRENTLY ", 1)
    if " IF NOT EXISTS " not in s.upper():
        s = s.replace("CREATE INDEX CONCURRENTLY ", "CREATE INDEX CONCURRENTLY IF NOT EXISTS ", 1)
    return s + ";"

def _safe_build_create_if_missing(sugg: dict) -> Optional[str]:
    table = (sugg.get("table") or "").strip()
    cols = sugg.get("columns") or []
    where = sugg.get("where")
    if not table or not cols:
        return None
    seed = f"{table}:{','.join(cols)}:{where or ''}"
    name = "idx_" + hashlib.md5(seed.encode("utf-8")).hexdigest()[:10]
    cols_sql = ", ".join(cols)
    if where:
        return f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {name} ON {table} ({cols_sql}) WHERE {where};"
    return f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {name} ON {table} ({cols_sql});"

def apply_indexes_and_reexecute(state: State) -> State:
    try:
        suggestions = state.get("index_suggestions") or []
        if not suggestions:
            return state

        created = []
        for s in suggestions:
            raw = (s.get("create_sql") or "").strip()

            if raw:
                # Normalize raw SQL and ensure explicit name
                try:
                    sql, idx_name, managed = _ensure_named_index_sql(raw)
                except Exception:
                    # Fallback: synthesize from fields if raw malformed
                    sql = _safe_build_create_if_missing(s)
                    idx_name = _extract_index_name(sql) if sql else None
                    managed = bool(idx_name and _is_managed_index_name(idx_name))
            else:
                # Build from structured suggestion (managed name)
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
                # drena em chunks, sem materializar em memória
                _ = _drain_select_streaming_pg(q)
            else:
                # comandos não-SELECT (DDL/DML): executa em autocommit
                engine = db._engine
                with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
                    conn.exec_driver_sql(q)

            after_elapsed = time.perf_counter() - start
            print(f"\n⏱️ Re-executed with indexes in {after_elapsed:.4f} seconds.")

        # ---- Cleanup if improvement < threshold ----
        dropped = []
        cleanup_reason = None
        improvement_pct = None
        before_elapsed = state.get("execution_time_seconds")

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
                            # last-ditch try from SQL text
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
    
    
# Using nodes for langraph
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

        # Core: generate_query -> execute_query -> generate_answer
        result = app.invoke(state) or state
        print(f"[DEBUG/run_agent] got suggestions: {len((result or {}).get('index_suggestions') or [])}")

        # IMPORTANT: if there are suggestions, apply and re-run automatically
        print(f"[DEBUG] Suggestions count: {len(result.get('index_suggestions') or [])}")
        if result.get("index_suggestions"):
            # Apply guardrails before creating anything
            filtered, guardrail_notes = _filter_index_suggestions_with_guardrails(result, result["index_suggestions"])
            if guardrail_notes:
                extra = (result.get("index_notes") or "").strip()
                joined = ("; ".join(guardrail_notes))
                result["index_notes"] = f"{extra + ' | ' if extra else ''}{joined}"
            result["index_suggestions"] = filtered
        
            # If nothing survives, skip creation path
            if not filtered:
                return result
        
            before = result.get("execution_time_seconds")
            result = apply_indexes_and_reexecute(result)

            # Attach comparison numbers if available
            after = result.get("execution_time_seconds_after_index")
            if isinstance(before, (int, float)) and isinstance(after, (int, float)) and after > 0:
                result["runtime_delta_seconds"] = after - before
                result["runtime_speedup_x"] = before / after

        return result

    except Exception as e:
        print(f"[ERROR] Workflow execution failed: {e}")
        return state or {"question": question, "error": str(e)}
    

# Main loop for the agent
if __name__ == "__main__":
    print("Database SQL Agent (type 'exit' to quit, 'refresh schema' to reload schema)")
    print(f"Connected to database: {db_database} in {dialect_name}")
    retriever = SQLQueryRetriever(model_name='all-MiniLM-L6-v2')
    retriever.carregar_json('examples.json')
    retriever.construir_index()

    agent_state = {
        "db_schema": get_db_schema()
    }

    while True:
        question = input("\nEnter your question: ").strip()

        few_args = retriever.buscar_filtrado(question, top_k=2, threshold=0.5)

        if question.lower() in ['exit', 'quit']:
            print("Goodbye!")
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

            # Answer
            if agent_state.get("answer"):
                print("\n Answer:\n" + agent_state["answer"])

            # If indexes were suggested, run_agent already created them and re-ran the query.
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
                        agent_state["runtime_improvement_pct"] = impr  # persist

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
                # No suggestions — ask another question
                print("\n[INDEX SUGGESTIONS] none — ask another question.")
                if agent_state.get("index_notes"):
                    print(f"[WHY NONE] {agent_state['index_notes']}")

        except Exception as e:
            print(f"[FATAL ERROR] Unexpected failure: {e}")
            agent_state = {"db_schema": get_db_schema()}

