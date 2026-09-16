"""
Multi-Tier Financial QA Benchmark (Tiers 0 to 3)
Compares:
  - Tier 0: Direct LLM (Zero-shot baseline, no retrieval)
  - Tier 1: Vector RAG (bge-small-en-v1.5 embedding search)
  - Tier 2: Graph-Enriched RAG (Vector passages + Cypher relationship lookups)
  - Tier 3: Agentic GraphRAG (LangChain Agent with dynamic tool routing & symbolic calculator)

Outputs:
  - Detailed reasoning and answer per tier
  - Comparative execution latency (seconds)
  - Markdown summary table and benchmark_results.json
"""

import os
import json
import time
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from neo4j import GraphDatabase
from groq import Groq
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_core.tools import tool

# 1. Config & Environment
load_dotenv(override=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")

NEO4J_USER = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER") or os.getenv("USERNAME", "24708389")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD") or os.getenv("PASSWORD", "")
NEO4J_URI = os.getenv("NEO4J_URI") or os.getenv("URI") or f"neo4j+ssc://{NEO4J_USER}.databases.neo4j.io"

groq_client = Groq(api_key=GROQ_API_KEY)
neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD), max_connection_lifetime=1800)

# 2. Embeddings & Data
print("Initializing embeddings cache...")
EMBED_MODEL = SentenceTransformer("BAAI/bge-small-en-v1.5")

if os.path.exists("aapl_embeddings.npy") and os.path.exists("aapl_chunks.json"):
    print("Loaded cached AAPL chunks & embeddings.")
    with open("aapl_chunks.json", "r") as f:
        CHUNKS = pd.Series(json.load(f))
    CHUNK_EMBEDDINGS = np.load("aapl_embeddings.npy")
else:
    print("Computing embeddings from CSV...")
    df = pd.read_csv("finreflectkg_aapl_msft.csv")
    df_aapl = df[df["ticker"] == "AAPL"]
    CHUNKS = df_aapl["chunk_text"].dropna().drop_duplicates().reset_index(drop=True)
    CHUNK_EMBEDDINGS = EMBED_MODEL.encode(CHUNKS.tolist(), show_progress_bar=True, batch_size=64)
    np.save("aapl_embeddings.npy", CHUNK_EMBEDDINGS)
    with open("aapl_chunks.json", "w") as f:
        json.dump(CHUNKS.tolist(), f)

print("Data & Graph connection ready.\n")


# 3. Benchmark Questions (8 multi-facet questions)
BENCHMARK_QUESTIONS = [
    {
        "id": "Q1",
        "category": "Regulatory & Legal",
        "ticker": "AAPL",
        "question": "Which regulatory bodies are named in connection with Apple's disclosed net income, and what litigation prompted this?",
        "expected_relations": ["regulates", "subject_to"],
    },
    {
        "id": "Q2",
        "category": "Market Risk & Macro",
        "ticker": "AAPL",
        "question": "What financial market conditions does Apple disclose as negatively impacting its financial metrics?",
        "expected_relations": ["negatively_impacts", "impacted_by"],
    },
    {
        "id": "Q3",
        "category": "Temporal YoY Growth",
        "ticker": "AAPL",
        "question": "What was Apple's Net sales in 2021 and 2022, and what was the year-over-year percentage growth?",
        "expected_relations": ["discloses"],
    },
    {
        "id": "Q4",
        "category": "Cross-Entity Ownership",
        "ticker": "MSFT",
        "question": "Which entities or segments does Microsoft hold a stake in that disclose financial metrics?",
        "expected_relations": ["has_stake_in", "discloses"],
    },
    {
        "id": "Q5",
        "category": "Regulatory Oversight",
        "ticker": "MSFT",
        "question": "What regulatory requirements or bodies regulate Microsoft regarding its disclosed operations?",
        "expected_relations": ["regulates", "subject_to"],
    },
    {
        "id": "Q6",
        "category": "Macro Risk Factors",
        "ticker": "MSFT",
        "question": "Which macroeconomic conditions are disclosed by Microsoft as impacting its financial performance?",
        "expected_relations": ["impacted_by", "negatively_impacts"],
    },
    {
        "id": "Q7",
        "category": "Product & Revenue Stream",
        "ticker": "AAPL",
        "question": "What products does Apple produce that disclose positive impacts on financial performance?",
        "expected_relations": ["produce", "positively_impacts"],
    },
    {
        "id": "Q8",
        "category": "Temporal Net Income",
        "ticker": "AAPL",
        "question": "What was Apple's Net Income in 2022 and 2023, and what was the percentage change?",
        "expected_relations": ["discloses"],
    },
]


# ==========================================
# TIER IMPLEMENTATIONS
# ==========================================

# ---- Tier 0: Direct LLM (Zero-Shot Baseline) ----
def run_tier0(question: str) -> dict:
    t0 = time.time()
    prompt = f"Answer this financial question based on SEC 10-K disclosures. Question: {question}"
    resp = groq_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=300
    )
    latency = time.time() - t0
    return {
        "tier": "Tier 0 (Zero-Shot)",
        "answer": resp.choices[0].message.content.strip(),
        "latency_sec": round(latency, 2),
        "context_used": "None"
    }


# ---- Tier 1: Vector RAG ----
def run_tier1(question: str, top_k: int = 5) -> dict:
    t0 = time.time()
    q_emb = EMBED_MODEL.encode(["Represent this sentence for searching relevant passages: " + question])[0]
    sims = np.dot(CHUNK_EMBEDDINGS, q_emb) / (
        np.linalg.norm(CHUNK_EMBEDDINGS, axis=1) * np.linalg.norm(q_emb)
    )
    top_idx = np.argsort(sims)[::-1][:top_k]
    retrieved = CHUNKS.iloc[top_idx].tolist()
    context = "\n\n".join(f"[Passage {i+1}]: {c[:400]}" for i, c in enumerate(retrieved))

    prompt = f"""Answer the question using ONLY the retrieved passages below. If not found in context, say so.

Context:
{context}

Question: {question}"""

    resp = groq_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=300
    )
    latency = time.time() - t0
    return {
        "tier": "Tier 1 (Vector RAG)",
        "answer": resp.choices[0].message.content.strip(),
        "latency_sec": round(latency, 2),
        "context_used": f"{len(retrieved)} vector passages"
    }


# ---- Tier 2: Graph-Enriched RAG ----
def query_graph_relations(ticker: str, rel_types: list) -> list:
    query = """
    MATCH (a:Entity)-[r:RELATION]->(b:Entity)
    WHERE (a.ticker = $ticker OR b.ticker = $ticker) AND r.type IN $rel_types
    RETURN a.name AS src, a.entity_type AS src_type, r.type AS rel, b.name AS tgt, b.entity_type AS tgt_type, r.chunk_text AS chunk
    LIMIT 8
    """
    with neo4j_driver.session() as session:
        return list(session.run(query, ticker=ticker, rel_types=rel_types))

def run_tier2(item: dict) -> dict:
    t0 = time.time()
    # 1. Vector passages
    q_emb = EMBED_MODEL.encode(["Represent this sentence for searching relevant passages: " + item["question"]])[0]
    sims = np.dot(CHUNK_EMBEDDINGS, q_emb) / (
        np.linalg.norm(CHUNK_EMBEDDINGS, axis=1) * np.linalg.norm(q_emb)
    )
    top_idx = np.argsort(sims)[::-1][:3]
    vec_passages = [f"[Passage {i+1}]: {c[:300]}" for i, c in enumerate(CHUNKS.iloc[top_idx])]

    # 2. Graph facts
    graph_records = query_graph_relations(item["ticker"], item.get("expected_relations", []))
    graph_facts = [
        f"[{r['src_type']}] {r['src']} --{r['rel']}--> [{r['tgt_type']}] {r['tgt']}"
        for r in graph_records
    ]

    context = "GRAPH FACTS (Knowledge Graph):\n" + ("\n".join(graph_facts) if graph_facts else "No direct graph facts found.")
    context += "\n\nRETRIEVED PASSAGES:\n" + "\n\n".join(vec_passages)

    prompt = f"""Answer the question using BOTH the structured Knowledge Graph facts and Retrieved Passages. Prefer Graph Facts for structured entities and Passages for narrative context.

Context:
{context}

Question: {item['question']}"""

    resp = groq_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=300
    )
    latency = time.time() - t0
    return {
        "tier": "Tier 2 (Graph-Enriched RAG)",
        "answer": resp.choices[0].message.content.strip(),
        "latency_sec": round(latency, 2),
        "context_used": f"{len(graph_facts)} graph facts + {len(vec_passages)} passages"
    }


# ---- Tier 3: Agentic GraphRAG Tools & Assembly ----
@tool
def vector_search_tool(query: str) -> str:
    """Search filing text passages semantically. Use for open-ended or descriptive questions."""
    q_emb = EMBED_MODEL.encode(["Represent this sentence for searching relevant passages: " + query])[0]
    sims = np.dot(CHUNK_EMBEDDINGS, q_emb) / (
        np.linalg.norm(CHUNK_EMBEDDINGS, axis=1) * np.linalg.norm(q_emb)
    )
    top_idx = np.argsort(sims)[::-1][:4]
    results = CHUNKS.iloc[top_idx].tolist()
    return "\n\n".join(c[:400] for c in results)

@tool
def graph_traversal_tool(input_str: str) -> str:
    """Look up structured relationships from the knowledge graph.
    Input format: 'ticker|relation_type', e.g. 'AAPL|regulates'.
    Valid relation_types: regulates, subject_to, negatively_impacts, impacted_by, discloses, has_stake_in, produce, positively_impacts."""
    try:
        ticker, relation_type = [x.strip() for x in input_str.split("|")]
    except ValueError:
        return "Input must be 'ticker|relation_type'."

    query = """
    MATCH (a:Entity)-[r:RELATION {type:$relation_type}]->(b:Entity)
    WHERE a.ticker = $ticker OR b.ticker = $ticker
    RETURN a.name AS src, a.entity_type AS src_type, r.type AS rel, b.name AS tgt, b.entity_type AS tgt_type
    LIMIT 6
    """
    with neo4j_driver.session() as session:
        records = list(session.run(query, ticker=ticker, relation_type=relation_type))

    if not records:
        return f"No facts found for {ticker} with relation '{relation_type}'."
    return "\n".join(f"[{r['src_type']}] {r['src']} --{r['rel']}--> [{r['tgt_type']}] {r['tgt']}" for r in records)

@tool
def symbolic_computation_tool(input_str: str) -> str:
    """Fetch numeric metric values across fiscal years and compute YoY growth rate.
    Input format: 'ticker|metric_name|year_start|year_end', e.g. 'AAPL|Net sales|2021|2022'."""
    try:
        parts = [x.strip() for x in input_str.split("|")]
        ticker, metric_name, year_start, year_end = parts[0], parts[1], int(parts[2]), int(parts[3])
    except Exception:
        return "Invalid format. Use: ticker|metric_name|year_start|year_end"

    query = """
    MATCH (org:Entity {ticker:$ticker})-[r:RELATION {type:'discloses'}]->(m:Entity {entity_type:'FIN_METRIC'})
    WHERE toLower(m.name) CONTAINS toLower($metric_name)
    RETURN m.name AS metric_name, r.year AS year, r.chunk_text AS chunk_text
    ORDER BY r.year
    """
    with neo4j_driver.session() as session:
        records = list(session.run(query, ticker=ticker, metric_name=metric_name))

    r_start = next((r for r in records if r["year"] == year_start), None)
    r_end = next((r for r in records if r["year"] == year_end), None)

    if not r_start or not r_end:
        return f"Found records for years: {[r['year'] for r in records]}. Missing data for {year_start} or {year_end}."

    # Simple regex extractor for digits
    import re
    def extract_num(text):
        m = re.findall(r"\$?\s*([0-9]{1,3}(?:,[0-9]{3})+(?:\.[0-9]+)?)", text)
        return float(m[0].replace(",", "")) if m else None

    v1 = extract_num(r_start["chunk_text"])
    v2 = extract_num(r_end["chunk_text"])

    if v1 is None or v2 is None or v1 == 0:
        return f"Extracted text for {year_start} and {year_end} but numeric value parsing was ambiguous."

    growth = round(((v2 - v1) / v1) * 100, 2)
    return f"{metric_name} for {ticker}: {year_start}=${v1:,.0f}M, {year_end}=${v2:,.0f}M, YoY Growth={growth}%"

agent_llm = ChatGroq(model=MODEL_NAME, api_key=GROQ_API_KEY, temperature=0)
tier3_agent = create_agent(
    model=agent_llm,
    tools=[vector_search_tool, graph_traversal_tool, symbolic_computation_tool],
    system_prompt="You are a financial research assistant. Use the available tools to answer questions about SEC 10-K filings accurately, preferring structured graph facts over general text search when possible."
)

def run_tier3(question: str) -> dict:
    t0 = time.time()
    resp = tier3_agent.invoke({"messages": [{"role": "user", "content": question}]})
    latency = time.time() - t0
    
    tools_called = []
    for msg in resp["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tools_called.append(f"{tc['name']}({tc['args']})")

    final_ans = resp["messages"][-1].content
    return {
        "tier": "Tier 3 (Agentic GraphRAG)",
        "answer": final_ans,
        "latency_sec": round(latency, 2),
        "tools_called": tools_called,
        "context_used": f"Agent Tools Called: {', '.join(tools_called) if tools_called else 'Direct synthesis'}"
    }


# ==========================================
# BENCHMARK RUNNER
# ==========================================
def main():
    print("=" * 80)
    print("STARTING MULTI-TIER BENCHMARK (Tiers 0 - 3) ACROSS 8 QUESTIONS")
    print(f"Active LLM: {MODEL_NAME}")
    print("=" * 80)

    results = []
    
    for item in BENCHMARK_QUESTIONS:
        qid = item["id"]
        qtext = item["question"]
        cat = item["category"]
        print(f"\n>>> Running [{qid}] ({cat}): {qtext}")

        # Tier 0
        res_t0 = run_tier0(qtext)
        print(f"  [T0] Latency: {res_t0['latency_sec']}s")

        # Tier 1
        res_t1 = run_tier1(qtext)
        print(f"  [T1] Latency: {res_t1['latency_sec']}s")

        # Tier 2
        res_t2 = run_tier2(item)
        print(f"  [T2] Latency: {res_t2['latency_sec']}s")

        # Tier 3
        res_t3 = run_tier3(qtext)
        print(f"  [T3] Latency: {res_t3['latency_sec']}s | Tools: {res_t3.get('tools_called', [])}")

        results.append({
            "id": qid,
            "category": cat,
            "question": qtext,
            "tier0": res_t0,
            "tier1": res_t1,
            "tier2": res_t2,
            "tier3": res_t3
        })

    # Save complete results
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE - RESULTS SUMMARY TABLE")
    print("=" * 80)
    print(f"{'QID':<4} | {'Category':<22} | {'T0 Latency':<10} | {'T1 Latency':<10} | {'T2 Latency':<10} | {'T3 Latency':<10}")
    print("-" * 80)
    for r in results:
        print(f"{r['id']:<4} | {r['category']:<22} | {r['tier0']['latency_sec']:>8}s | {r['tier1']['latency_sec']:>8}s | {r['tier2']['latency_sec']:>8}s | {r['tier3']['latency_sec']:>8}s")

    avg_t0 = np.mean([r['tier0']['latency_sec'] for r in results])
    avg_t1 = np.mean([r['tier1']['latency_sec'] for r in results])
    avg_t2 = np.mean([r['tier2']['latency_sec'] for r in results])
    avg_t3 = np.mean([r['tier3']['latency_sec'] for r in results])
    print("-" * 80)
    print(f"{'AVG':<4} | {'Overall Average':<22} | {avg_t0:>8.2f}s | {avg_t1:>8.2f}s | {avg_t2:>8.2f}s | {avg_t3:>8.2f}s")
    print("=" * 80)
    print("Detailed outputs saved to file: benchmark_results.json\n")


if __name__ == "__main__":
    try:
        main()
    finally:
        neo4j_driver.close()
