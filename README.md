#  Athena Local SQL Agent

An intelligent **text-to-SQL agent** that connects natural language queries to a **PostgreSQL database**, automatically generates and executes optimized SQL queries, and suggests **index optimizations** using DeepSeek's LLM API.  
It also integrates a **semantic retriever** for query similarity (`close_query_select.py`), allowing the model to learn from example SQL statements (`examples.json`).

---

##  Overview

This project demonstrates a **local AI agent** capable of:
- Translating natural language questions into **PostgreSQL SQL queries**
- Executing those queries directly on a **TPC-H database**
- Measuring performance, generating **index suggestions**, and re-executing for comparison
- Storing query logs and timing metrics locally
- Using **embeddings (FAISS + SentenceTransformer)** for few-shot example retrieval

The system is modular and locally executable — all processing happens on your machine, except the **DeepSeek API** calls for query generation and analysis.

---

##  Architecture
>  **Note:**  
> This version of the system is implemented for **PostgreSQL**.  
> For copyright and licensing reasons, the original implementation that interacted with a **commercial database** is **not included** here.

```text
+-------------------------------------------------------------+
|                          User Input                         |
|            e.g., "Show customers with the highest orders"   |
+-------------------------------------------------------------+
                               |
                               v
+----------------------+    +---------------------------+
| close_query_select.py|    |   agent.py (Main Logic)   |
|  - Builds embeddings |--->| - Generates SQL (DeepSeek)|
|  - Finds similar SQL |    | - Executes & times query  |
|  - Uses FAISS index  |    | - Suggests indexes        |
+----------------------+    | - Applies improvements    |
                            +---------------------------+
                               |
                               v
                        +-------------------+
                        | PostgreSQL (TPC-H)|
                        +-------------------+
```

### Main Components:
- **`agent.py`** — Orchestrates the pipeline:
  - Connects to PostgreSQL
  - Uses DeepSeek for SQL generation & optimization
  - Executes and measures query performance
  - Suggests indexes automatically and re-runs queries
- **`agent2.py`** — Orchestrates the pipeline:
  - Connects to SQLServer
  - Uses DeepSeek for SQL generation & optimization
  - Executes and measures query performance
  - Suggests indexes automatically and re-runs queries
- **`close_query_select.py`** — Handles example retrieval:
  - Loads question–SQL pairs from `examples.json`
  - Encodes them with SentenceTransformer (`all-MiniLM-L6-v2`)
  - Builds a FAISS vector index
  - Returns the most semantically similar examples for prompt conditioning
- **`examples.json`** — Provides example queries (few-shot context)
- **`.env`** — Stores local configuration (database + API keys)

---

##  Setup

### 1. Requirements
Ensure you have:
- Python **3.12**
- PostgreSQL **with TPC-H dataset (scale factor 10)**
- A **DeepSeek API key** (paid, via [deepseek.com](https://deepseek.com))

---

## Installation

#### Create and activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

#### Install dependencies
```bash
 setup-system.sh
```
```bash
pip install -r requirements.txt
```
#### Environment Configuration
Copy the ```.env-EXAMPLE``` file and fill in your credentials:
```
cp .env-EXAMPLE .env

```
Then open .env and set your variables:
```
# DeepSeek API Key (required)
DEEPSEEK_API_KEY=sk-xxxxxx

# PostgreSQL Connection
You have to install tpch-10 and host it locally on postgresql.
DB_HOST=localhost
DB_PORT=5432
DB_USER=postgres
DB_PASSWORD=yourpassword
DB_DATABASE=tpc_h_sf10

# SQLServer Connection
You have to install tpch-10 and host it locally on sqlserver.
DB_HOST=localhost
DB_PORT=5432
DB_USER=postgres
DB_PASSWORD=yourpassword
DB_DATABASE=tpc_h_sf10
```

## Dataset
This project is designed to work with TPC-H Scale Factor 10, a standard decision support benchmark. (You could also try others, but the embeddings are tuned for TPC-H.)
You can generate it using the [TPC-H Benchmark Tool](https://www.tpc.org/tpch/)

```
# Example: generate and import dataset into PostgreSQL
./dbgen -s 10
psql -U postgres -d tpc_h_sf10 -f dss.ddl
```

Ensure the tables are accessible under your configured database.

## Running the Application
Run the interactive agent locally(for postgresql):
```
python agent.py
```

or for sqlserver:

```
python agent2.py
```

You’ll see:

```
Database SQL Agent (type 'exit' to quit, 'refresh schema' to reload schema)
Connected to database: tpc_h_sf10 in postgresql
```
Then type a natural language question, for example:

```
Enter your question: List customers from Germany and France

```
The agent will:
1. Retrieve similar examples via FAISS
2. Generate the SQL query using DeepSeek
3. Execute it against PostgreSQL
4. Show execution time
5. Suggest indexes if beneficial
6. Re-run query and compare runtime

## Embeddings (Few-Shot Retrieval) -> Only for PostgreSQL (agent.py)
The module ```close_query_select.py``` builds semantic embeddings using SentenceTransformer (all-MiniLM-L6-v2) and FAISS:

* ```examples.json``` stores reference (question → SQL) mappings.
* Before each query, the system finds the most semantically similar examples.
* These examples are injected as few-shot prompts to improve SQL generation accuracy.

This allows the model to learn from previous examples without re-training.

## Example Interaction
Input:
```
Enter your question: Show supplier id and name for suppliers from nation key 15, and include the nation name.

```

Output:
```
Answer:
Here are suppliers from nation key 15 and their associated nation name.

[INDEX SUGGESTIONS]
- CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_123456789a ON h_supplier (s_nationkey)  # Seq Scan on filtered equality and no index detected
Query executed in 0.3421 seconds.
Re-executed with indexes in 0.2175 seconds.
Speedup: 1.57× (36.4% improvement)

```
## Files Summary
| File | Description |
|------|--------------|
| `.env-EXAMPLE` | Example environment configuration |
| `agent.py` | Main orchestrator: LLM calls, execution, optimization |
| `close_query_select.py` | Embeddings and FAISS index builder |
| `examples.json` | Example (question, SQL) pairs for few-shot retrieval |
| `requirements.txt` | Python dependencies for environment setup |

## Notes
* The DeepSeek API is a paid service. You must obtain an API key to use the model.
* All query execution and index creation happen locally on your PostgreSQL database.
* The agent modifies the database by creating and dropping indexes to test optimizations — use a development/test environment, not production.
  
## License
This project is distributed for educational and research purposes.
Feel free to modify and extend it for local experimentation.

