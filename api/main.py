import os
import sys
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import List, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import io

from langchain_core.messages import HumanMessage

from db_loader import load_user_csvs, cleanup_user_schema, schema_name_for, get_conn_details
from utils.database import DatabaseUtil
from utils.llm_pick import pick_llm
from utils.etl_tools import ETLTools
from agents.sql_analyst import sql_analyst
from agents.insights_analyst import insights_analyst
from agents.etl_analyst import extract_load_tool
from Models.schema import ChatRouterSchema

chat_router_llm = pick_llm("low").with_structured_output(ChatRouterSchema)


app = FastAPI(title="Data Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_ROOT = Path(tempfile.gettempdir()) / "data_agent_uploads"
UPLOAD_ROOT.mkdir(exist_ok=True)


# ---------------------------------------------------------------- models

class SessionResponse(BaseModel):
    session_id: str


class UploadResponse(BaseModel):
    session_id: str
    tables: dict


class QueryRequest(BaseModel):
    session_id: str
    question: str


class QueryResponse(BaseModel):
    type: str
    answer: str
    sql: Optional[str] = None
    new_table: Optional[str] = None
    row_count: Optional[int] = None


class ExtractRequest(BaseModel):
    session_id: str
    url: str
    format: str = "csv"


class ExtractResponse(BaseModel):
    session_id: str
    message: str
    tables: dict


class TablesResponse(BaseModel):
    session_id: str
    schema_details: str


# ---------------------------------------------------------------- routes

@app.post("/api/session", response_model=SessionResponse)
def create_session():
    """Start a new isolated workspace. Returns a session_id to use for uploads/queries."""
    session_id = uuid.uuid4().hex[:12]
    return {"session_id": session_id}


@app.post("/api/upload", response_model=UploadResponse)
async def upload_csvs(session_id: str = Form(...), files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    session_dir = UPLOAD_ROOT / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for f in files:
        if not f.filename.lower().endswith(".csv"):
            raise HTTPException(status_code=400, detail=f"Only CSV files supported, got: {f.filename}")
        dest = session_dir / f.filename
        with dest.open("wb") as out:
            shutil.copyfileobj(f.file, out)
        saved_paths.append(str(dest))

    schema_name = schema_name_for(session_id)

    try:
        tables = load_user_csvs(saved_paths, schema_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load CSVs: {e}")

    return {"session_id": session_id, "tables": tables}


@app.get("/api/tables/{session_id}", response_model=TablesResponse)
def list_tables(session_id: str):
    schema_name = schema_name_for(session_id)
    conn_details = get_conn_details()
    db = DatabaseUtil(conn_details)
    try:
        details = db.schema_details(schema_name)
    finally:
        db.close()
    return {"session_id": session_id, "schema_details": details}


def _df_to_csv_response(df, filename: str) -> StreamingResponse:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/download/table/{session_id}/{table_name}")
def download_table(session_id: str, table_name: str):
    """Download an entire loaded table (uploaded CSV or extracted data) as CSV."""
    schema_name = schema_name_for(session_id)
    safe_table = DatabaseUtil._sanitize_identifier(table_name)

    conn_details = get_conn_details()
    db = DatabaseUtil(conn_details)
    try:
        df = db.execute_query_df(f'SELECT * FROM "{schema_name}"."{safe_table}"')
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Table not found or query failed: {e}")
    finally:
        db.close()

    return _df_to_csv_response(df, f"{safe_table}.csv")


@app.post("/api/extract", response_model=ExtractResponse)
def extract_from_api(req: ExtractRequest):
    """
    Structured API extraction — no router guessing. Pulls data from the given
    URL, saves it under this session's folder, then loads it into the same
    Postgres schema as uploaded CSVs so it's immediately queryable.
    """
    session_dir = UPLOAD_ROOT / req.session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    try:
        result_msg = extract_load_tool.invoke(
            {"url": req.url, "output_folder": str(session_dir), "format": req.format}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {e}")

    tables = {}
    extracted_path = session_dir / f"extracted_data.{req.format}"
    if req.format == "csv" and extracted_path.exists():
        schema_name = schema_name_for(req.session_id)
        try:
            tables = load_user_csvs([str(extracted_path)], schema_name)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Extracted but failed to load into DB: {e}")

    return {"session_id": req.session_id, "message": result_msg, "tables": tables}


@app.post("/api/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """
    Chat only ever answers questions against already-loaded data — never
    triggers extraction/scraping (those stay in the sidebar). A small router
    first classifies the question as a specific-answer SQL query or an
    open-ended "key insights" request, then dispatches accordingly.
    """
    schema_name = schema_name_for(req.session_id)

    try:
        route = chat_router_llm.invoke(req.question).answer
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Routing failed: {e}")

    if route == "insights":
        try:
            response = insights_analyst.invoke(
                {
                    "messages": [],
                    "user_question": req.question,
                    "schema_name": schema_name,
                    "table_name": "",
                    "data_summary": "",
                    "final_answer": "",
                }
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Insights generation failed: {e}")

        return {"type": "insights", "answer": response.get("final_answer", ""), "sql": None}

    if route == "transform":
        tools = ETLTools()
        conn_details = get_conn_details()
        try:
            target_table = tools.pick_target_table(conn_details, schema_name, req.question)
            result = tools.apply_transformation(conn_details, schema_name, target_table, req.question)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Transformation failed: {e}")

        return {
            "type": "transform",
            "answer": result["message"],
            "sql": None,
            "new_table": result["target_table"],
            "row_count": result["row_count"],
        }

    input_schema = {
        "messages": [],
        "user_question": req.question,
        "schema_name": schema_name,
        "curated_ques": "",
        "prompt_query_context": "",
        "generated_sql_query": "",
        "is_safe": "No",
        "comments": "",
        "sql_query_execution_result": "",
        "final_answer": "",
    }

    try:
        response = sql_analyst.invoke(input_schema)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")

    return {
        "type": "sql",
        "answer": response.get("final_answer", ""),
        "sql": response.get("generated_sql_query"),
    }


@app.delete("/api/session/{session_id}")
def delete_session(session_id: str):
    schema_name = schema_name_for(session_id)
    try:
        cleanup_user_schema(schema_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clean up: {e}")

    session_dir = UPLOAD_ROOT / session_id
    if session_dir.exists():
        shutil.rmtree(session_dir)

    return {"status": "deleted", "session_id": session_id}


@app.get("/api/health")
def health():
    return {"status": "ok"}


# Serve the test frontend at "/" (keep this mount LAST so /api/* routes above take priority)
frontend_dir = Path(__file__).parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")