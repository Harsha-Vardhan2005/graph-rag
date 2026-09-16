"""
FinReflectKG UI Demo Backend Server
FastAPI backend providing REST endpoints for the interactive review dashboard.
"""

import os
import sys
import time
import re
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv, find_dotenv

# Ensure root directory is in sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "Adaptive_Routing"))
sys.path.insert(0, str(root_dir / "Structured_Formatting"))

load_dotenv(find_dotenv(), override=True)

from Adaptive_Routing.router import QueryRouter, RouteType
from Structured_Formatting.pipeline_with_formatting import FormattedGraphRAG
from Structured_Formatting.formatter import format_structured_evidence

app = FastAPI(
    title="FinReflectKG Agentic GraphRAG API",
    description="Backend service for Financial SEC 10-K Question Answering, Adaptive Routing, and Structured Evidence Formatting",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG Pipeline
print("Initializing FinReflectKG Pipeline Backend...")
rag_pipeline = FormattedGraphRAG(enable_structured_formatting=True)
router_instance = QueryRouter(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name=os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
)
print("Pipeline Backend Ready.")

# Supported Companies in FinReflectKG
SUPPORTED_TICKERS = {"AAPL", "MSFT"}
KNOWN_UNSUPPORTED_ENTITIES = {
    "tesla": "TSLA", "tsla": "TSLA",
    "google": "GOOGL", "alphabet": "GOOGL", "googl": "GOOGL",
    "amazon": "AMZN", "amzn": "AMZN",
    "nvidia": "NVDA", "nvda": "NVDA",
    "meta": "META", "facebook": "META"
}

class QueryRequest(BaseModel):
    query: str
    force_route: Optional[str] = None
    enable_formatting: bool = True

class RouteAblationRequest(BaseModel):
    query: str

class FormatAblationRequest(BaseModel):
    query: str

def check_scope(query: str) -> Optional[Dict[str, Any]]:
    """Checks if query references companies outside indexed SEC 10-K filings."""
    q_low = query.lower()
    for company_name, ticker in KNOWN_UNSUPPORTED_ENTITIES.items():
        if re.search(r'\b' + re.escape(company_name) + r'\b', q_low):
            return {
                "in_scope": False,
                "detected_entity": company_name.title(),
                "detected_ticker": ticker,
                "message": (
                    f"Scope Notice: Current FinReflectKG index contains SEC 10-K filings for "
                    f"Apple Inc. (AAPL) and Microsoft Corp. (MSFT) (51,927 triplets). "
                    f"Coverage for {company_name.title()} ({ticker}) is scheduled for subsequent phases. "
                    f"Please submit queries regarding Apple or Microsoft."
                )
            }
    return None

@app.post("/api/query")
async def execute_query(req: QueryRequest):
    query_text = req.query.strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Query text cannot be empty.")

    # Check company scope guardrail
    scope_notice = check_scope(query_text)
    if scope_notice and not scope_notice["in_scope"]:
        return {
            "status": "out_of_scope",
            "scope_notice": scope_notice,
            "routing": {
                "route": "OUT_OF_SCOPE",
                "confidence": 1.0,
                "reasoning": f"Entity '{scope_notice['detected_entity']}' is outside active AAPL/MSFT index."
            },
            "answer": scope_notice["message"],
            "evidence": [],
            "latency_sec": 0.01,
            "formatted_evidence_table": ""
        }

    t0 = time.time()

    # Determine Route
    route_decision = router_instance.route(query_text)
    selected_route = req.force_route or route_decision["route"].value

    # Execute via pipeline
    if selected_route == "SIMPLE_VECTOR":
        exec_res = rag_pipeline._execute_vector_route(query_text)
    elif selected_route == "GRAPH_MULTIHOP":
        # Temporary override formatting preference
        rag_pipeline.enable_structured_formatting = req.enable_formatting
        exec_res = rag_pipeline._execute_graph_route(query_text, route_decision)
    elif selected_route == "SYMBOLIC_COMPUTE":
        exec_res = rag_pipeline._execute_symbolic_route(query_text, route_decision)
    else:
        exec_res = rag_pipeline._execute_vector_route(query_text)

    total_latency = time.time() - t0

    answer = exec_res.get("answer", "")
    evidence_context = exec_res.get("evidence_context_used", "")
    calc_summary = exec_res.get("calculation_summary", "")

    return {
        "status": "success",
        "query": query_text,
        "routing": {
            "route": selected_route,
            "confidence": route_decision["confidence"],
            "reasoning": route_decision["reasoning"]
        },
        "execution": {
            "answer": answer,
            "latency_sec": round(total_latency, 3),
            "evidence_count": exec_res.get("evidence_count", 0),
            "evidence_type": exec_res.get("evidence_type", selected_route.lower()),
            "evidence_context": evidence_context,
            "calculation_summary": calc_summary
        }
    }

@app.post("/api/route_ablation")
async def route_ablation(req: RouteAblationRequest):
    """Compares Fast-Path Vector vs Full Graph execution latency and output."""
    query_text = req.query.strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Query text cannot be empty.")

    # 1. Routing classification
    route_decision = router_instance.route(query_text)

    # 2. Fast Path (Vector Search)
    t_v0 = time.time()
    vec_res = rag_pipeline._execute_vector_route(query_text)
    t_v = time.time() - t_v0

    # 3. Graph Path (Knowledge Graph Traversal)
    t_g0 = time.time()
    graph_res = rag_pipeline._execute_graph_route(query_text, route_decision)
    t_g = time.time() - t_g0

    savings_pct = round(((t_g - t_v) / t_g) * 100, 1) if t_g > 0 else 0.0

    return {
        "query": query_text,
        "recommended_route": route_decision["route"].value,
        "reasoning": route_decision["reasoning"],
        "confidence": route_decision["confidence"],
        "vector_route": {
            "latency_sec": round(t_v, 3),
            "answer": vec_res.get("answer", "")
        },
        "graph_route": {
            "latency_sec": round(t_g, 3),
            "answer": graph_res.get("answer", "")
        },
        "latency_savings_pct": savings_pct
    }

@app.post("/api/format_ablation")
async def format_ablation(req: FormatAblationRequest):
    """Compares Raw Graph Triples vs Typed Structured Markdown Tables."""
    query_text = req.query.strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Query text cannot be empty.")

    route_decision = router_instance.route(query_text)

    # 1. Raw Flat Triples Execution
    rag_pipeline.enable_structured_formatting = False
    t_r0 = time.time()
    raw_res = rag_pipeline._execute_graph_route(query_text, route_decision)
    t_raw = time.time() - t_r0

    # 2. Structured Markdown Tables Execution
    rag_pipeline.enable_structured_formatting = True
    t_s0 = time.time()
    struct_res = rag_pipeline._execute_graph_route(query_text, route_decision)
    t_struct = time.time() - t_s0

    return {
        "query": query_text,
        "raw_baseline": {
            "evidence_context": raw_res.get("evidence_context_used", ""),
            "answer": raw_res.get("answer", ""),
            "latency_sec": round(t_raw, 3)
        },
        "structured_proposed": {
            "evidence_context": struct_res.get("evidence_context_used", ""),
            "answer": struct_res.get("answer", ""),
            "latency_sec": round(t_struct, 3)
        }
    }

# Mount static frontend directory
static_dir = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/")
async def serve_index():
    index_file = static_dir / "index.html"
    return FileResponse(str(index_file))

if __name__ == "__main__":
    import uvicorn
    print("Starting Web Server at http://127.0.0.1:8000 ...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
