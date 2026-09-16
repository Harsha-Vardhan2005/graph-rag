import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_core.tools import tool

load_dotenv(override=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

NEO4J_USER = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER") or os.getenv("USERNAME", "24708389")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD") or os.getenv("PASSWORD", "")
NEO4J_URI = os.getenv("NEO4J_URI") or os.getenv("URI") or f"neo4j+ssc://{NEO4J_USER}.databases.neo4j.io"

import json

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD),
    max_connection_lifetime=1800,
)

print("Initializing embeddings for vector search tool...")
EMBED_MODEL = SentenceTransformer("BAAI/bge-small-en-v1.5")

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
emb_file = os.path.join(parent_dir, "aapl_embeddings.npy") if os.path.exists(os.path.join(parent_dir, "aapl_embeddings.npy")) else "aapl_embeddings.npy"
chk_file = os.path.join(parent_dir, "aapl_chunks.json") if os.path.exists(os.path.join(parent_dir, "aapl_chunks.json")) else "aapl_chunks.json"
csv_file = os.path.join(parent_dir, "finreflectkg_aapl_msft.csv") if os.path.exists(os.path.join(parent_dir, "finreflectkg_aapl_msft.csv")) else "finreflectkg_aapl_msft.csv"

if os.path.exists(emb_file) and os.path.exists(chk_file):
    print("Loaded cached AAPL embeddings from disk (instant startup).")
    with open(chk_file, "r") as f:
        CHUNKS = pd.Series(json.load(f))
    CHUNK_EMBEDDINGS = np.load(emb_file)
else:
    print("Computing embeddings (first run)...")
    df = pd.read_csv(csv_file)
    df_aapl = df[df["ticker"] == "AAPL"]
    CHUNKS = df_aapl["chunk_text"].dropna().drop_duplicates().reset_index(drop=True)
    CHUNK_EMBEDDINGS = EMBED_MODEL.encode(CHUNKS.tolist(), show_progress_bar=True, batch_size=64)
    np.save(emb_file, CHUNK_EMBEDDINGS)
    with open(chk_file, "w") as f:
        json.dump(CHUNKS.tolist(), f)

print("Ready.\n")


# ---- Tool 1: Vector Search ----
@tool
def vector_search_tool(query: str) -> str:
    """Search filing text passages semantically. Use for open-ended or descriptive questions. Input: a natural language question."""
    q_embedding = EMBED_MODEL.encode(["Represent this sentence for searching relevant passages: " + query])[0]
    sims = np.dot(CHUNK_EMBEDDINGS, q_embedding) / (
        np.linalg.norm(CHUNK_EMBEDDINGS, axis=1) * np.linalg.norm(q_embedding)
    )
    top_idx = np.argsort(sims)[::-1][:5]
    results = CHUNKS.iloc[top_idx].tolist()
    return "\n\n".join(c[:400] for c in results)


# ---- Tool 2: Graph Traversal ----
def graph_lookup(tx, ticker, relation_type):
    query = """
    MATCH (a:Entity)-[r:RELATION {type:$relation_type}]->(b:Entity {ticker:$ticker})
    RETURN a.name AS source, a.entity_type AS source_type, r.chunk_text AS chunk_text
    LIMIT 8
    """
    return list(tx.run(query, ticker=ticker, relation_type=relation_type))

@tool
def graph_traversal_tool(input_str: str) -> str:
    """Look up structured relationships directly from the knowledge graph.
    Use for questions about specific relationship types like regulation, risk impact, ownership.
    Input format: 'ticker|relation_type', e.g. 'AAPL|negatively_impacts'.

    IMPORTANT: relation_type must be one of these EXACT values (case-sensitive):
    discloses, depends_on, negatively_impacts, increase, subject_to, introduces,
    impacted_by, produce, face, operates_in, regulates, has_stake_in, works_for,
    announces, join, member_of, audit, distributes, impact, positively_impacts.

    Use 'regulates' for regulatory bodies/oversight. Use 'subject_to' for litigation/legal exposure.
    Use 'negatively_impacts' or 'impacted_by' for risk/market conditions.
    """
    try:
        ticker, relation_type = [x.strip() for x in input_str.split("|")]
    except ValueError:
        return "Invalid input format. Use: ticker|relation_type (e.g. AAPL|negatively_impacts)"

    with driver.session() as session:
        results = session.execute_read(graph_lookup, ticker, relation_type)

    if not results:
        return f"No facts found for relation '{relation_type}' on {ticker}."

    lines = [f"[{r['source_type']}] {r['source']} --{relation_type}--> {ticker}" for r in results]
    return "\n".join(lines)


# ---- Tool 3: Symbolic Computation (simplified inline version) ----
import re

def fetch_metric_across_years(tx, ticker, metric_name):
    query = """
    MATCH (org:Entity {ticker:$ticker})-[r:RELATION {type:"discloses"}]->(m:Entity {entity_type:"FIN_METRIC"})
    WHERE toLower(m.name) CONTAINS toLower($metric_name)
    RETURN m.name AS metric_name, r.year AS year, r.chunk_text AS chunk_text
    ORDER BY r.year
    """
    return list(tx.run(query, ticker=ticker, metric_name=metric_name))

def extract_numeric_value(text, metric_hint=""):
    if not text:
        return None
    search_text = text
    if metric_hint:
        idx = text.lower().find(metric_hint.lower())
        if idx != -1:
            search_text = text[idx: idx + 150]
    matches = re.findall(r'\$\s?[\d,]+\.?\d*|\b\d{1,3}(?:,\d{3})+\.?\d*\b', search_text)
    numbers = []
    for m in matches:
        cleaned = m.replace("$", "").replace(",", "").strip()
        try:
            numbers.append(float(cleaned))
        except ValueError:
            continue
    return numbers[0] if numbers else None

@tool
def symbolic_computation_tool(input_str: str) -> str:
    """Compute year-over-year growth for a financial metric. Use for questions asking about growth, change, or trend over time. Input format: 'ticker|metric_name|year_start|year_end', e.g. 'AAPL|net sale|2014|2015'."""
    try:
        ticker, metric_name, year_start, year_end = [x.strip() for x in input_str.split("|")]
        year_start, year_end = int(year_start), int(year_end)
    except ValueError:
        return "Invalid input format. Use: ticker|metric_name|year_start|year_end"

    with driver.session() as session:
        records = session.execute_read(fetch_metric_across_years, ticker, metric_name)

    r_start = next((r for r in records if r["year"] == year_start), None)
    r_end = next((r for r in records if r["year"] == year_end), None)

    if not r_start or not r_end:
        return f"Could not find data for both {year_start} and {year_end}."

    val_start = extract_numeric_value(r_start["chunk_text"], metric_name)
    val_end = extract_numeric_value(r_end["chunk_text"], metric_name)

    if val_start is None or val_end is None or val_start == 0:
        return "Could not extract numeric values for computation."

    growth = round(((val_end - val_start) / val_start) * 100, 2)
    return f"{metric_name} for {ticker}: {year_start}=${val_start:,.0f}M, {year_end}=${val_end:,.0f}M, YoY growth={growth}%"


# ---- Assemble Agent ----
tools = [vector_search_tool, graph_traversal_tool, symbolic_computation_tool]

MODEL_NAME = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
llm = ChatGroq(model=MODEL_NAME, api_key=GROQ_API_KEY, temperature=0)

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a financial research assistant. Use the available tools to answer questions about Apple's SEC 10-K filings accurately, preferring structured graph facts over general text search when possible."
)


if __name__ == "__main__":
    questions = [
        "Which regulatory bodies are named in connection with Apple's disclosed net income, and what litigation prompted this?",
        "What financial market conditions does Apple disclose as negatively impacting its financial metrics?",
    ]

    for q in questions:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        response = agent.invoke({"messages": [{"role": "user", "content": q}]})

        print("\n--- Tool calls made by agent ---")
        for msg in response["messages"]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"  Called: {tc['name']}  |  Input: {tc['args']}")
            if msg.__class__.__name__ == "ToolMessage":
                print(f"  -> Result: {str(msg.content)[:200]}")

        final_message = response["messages"][-1]
        print(f"\nFinal Answer: {final_message.content}")

    driver.close()