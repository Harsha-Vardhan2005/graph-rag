"""
Benchmark: Claim-Level Provenance & Faithfulness Ablation
Compares:
  - Tier 0: Direct LLM (Zero SEC grounding / High hallucination risk)
  - Tier 1: Vector RAG (Unstructured chunk grounding)
  - Tier 3: Formatted Agentic GraphRAG (Full Graph + Vector + Structured Provenance)
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "Provenance_Scoring"))
sys.path.insert(0, str(root_dir / "Structured_Formatting"))
sys.path.insert(0, str(root_dir / "Adaptive_Routing"))

load_dotenv(find_dotenv(), override=True)

from groq import Groq
from Provenance_Scoring.verifier import ClaimVerifier
from Structured_Formatting.pipeline_with_formatting import FormattedGraphRAG

BENCHMARK_PROMPTS = [
    {
        "id": "P1",
        "question": "What are Apple's major risk factors regarding foreign exchange volatility and global supply chain disruptions?",
        "expected_topics": ["foreign exchange", "supply chain", "risk", "currencies"]
    },
    {
        "id": "P2",
        "question": "What are the market conditions and technological risk factors impacting Apple's product sales?",
        "expected_topics": ["market conditions", "technological", "product sales", "competition"]
    },
    {
        "id": "P3",
        "question": "How do regulatory and government compliance requirements affect Apple's operations and disclosures?",
        "expected_topics": ["regulatory", "government", "compliance", "antitrust"]
    },
    {
        "id": "P4",
        "question": "What key financial metrics does Apple disclose regarding net sales and operating income?",
        "expected_topics": ["net sales", "operating income", "revenue", "products"]
    }
]

def safe_chat(groq_client, model_name, messages, max_tokens=180):
    """Executes chat completion with rate-limit backoff."""
    for attempt in range(4):
        try:
            resp = groq_client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.1,
                max_tokens=max_tokens
            )
            time.sleep(1.2)  # Prevent OTPM exhaustion
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                time.sleep(3.0 * (attempt + 1))
            else:
                raise e
    return "Rate limit fallback response."

def run_benchmark():
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        
    print("=" * 80)
    print(" CLAIM-LEVEL PROVENANCE & FAITHFULNESS BENCHMARK")
    print("=" * 80)

    model_name = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    verifier = ClaimVerifier(token_overlap_threshold=0.35)
    formatted_rag = FormattedGraphRAG(enable_structured_formatting=True)

    results = []

    for item in BENCHMARK_PROMPTS:
        qid = item["id"]
        q = item["question"]
        print(f"\n[Evaluating {qid}] {q}")

        # 1. Tier 0: Direct LLM (No SEC context provided -> ungrounded w.r.t filing)
        t0_start = time.time()
        t0_ans = safe_chat(
            groq_client,
            model_name,
            [
                {"role": "system", "content": "You are a financial analyst. Answer the user's question concisely."},
                {"role": "user", "content": q}
            ],
            max_tokens=150
        )
        t0_time = time.time() - t0_start
        t0_audit = verifier.audit_response(t0_ans, vector_chunks=[], graph_facts=[])

        # 2. Tier 1: Vector RAG (Vector passages only)
        t1_start = time.time()
        q_emb = formatted_rag.embed_model.encode(["Represent this sentence for searching relevant passages: " + q])[0]
        sims = np.dot(formatted_rag.chunk_embeddings, q_emb) / (
            np.linalg.norm(formatted_rag.chunk_embeddings, axis=1) * np.linalg.norm(q_emb)
        )
        top_idx = np.argsort(sims)[::-1][:3]
        retrieved_chunks = formatted_rag.chunks.iloc[top_idx].tolist()
        t1_context = "\n---\n".join(retrieved_chunks)
        t1_prompt = f"Context from Apple SEC 10-K:\n{t1_context}\n\nQuestion: {q}\nAnswer concisely based only on context:"
        t1_ans = safe_chat(
            groq_client,
            model_name,
            [{"role": "user", "content": t1_prompt}],
            max_tokens=160
        )
        t1_time = time.time() - t1_start
        t1_audit = verifier.audit_response(t1_ans, vector_chunks=retrieved_chunks, graph_facts=[])

        # 3. Tier 3: Formatted Agentic GraphRAG + Provenance
        t3_start = time.time()
        t3_res = formatted_rag.answer_query(q)
        t3_time = time.time() - t3_start
        t3_ans = t3_res.get("execution", {}).get("answer", "")
        evidence_used = t3_res.get("execution", {}).get("evidence_context_used", "")
        t3_facts = [line for line in evidence_used.split("\n") if line.strip()]
        t3_audit = verifier.audit_response(t3_ans, vector_chunks=retrieved_chunks, graph_facts=t3_facts)

        entry = {
            "id": qid,
            "question": q,
            "tier0_direct_llm": {
                "faithfulness_score": t0_audit["faithfulness_score"],
                "hallucination_rate": t0_audit["hallucination_rate"],
                "total_claims": t0_audit["total_claims"],
                "grounded_claims": t0_audit["grounded_claims"],
                "latency_s": round(t0_time, 2)
            },
            "tier1_vector_rag": {
                "faithfulness_score": t1_audit["faithfulness_score"],
                "hallucination_rate": t1_audit["hallucination_rate"],
                "total_claims": t1_audit["total_claims"],
                "grounded_claims": t1_audit["grounded_claims"],
                "latency_s": round(t1_time, 2)
            },
            "tier3_agentic_provenance": {
                "faithfulness_score": t3_audit["faithfulness_score"],
                "hallucination_rate": t3_audit["hallucination_rate"],
                "total_claims": t3_audit["total_claims"],
                "grounded_claims": t3_audit["grounded_claims"],
                "evidence_sources_count": len(retrieved_chunks) + len(t3_facts),
                "latency_s": round(t3_time, 2)
            }
        }
        results.append(entry)

        print(f"  Tier 0 (Direct LLM): Faithfulness = {t0_audit['faithfulness_score']}% | Hallucination = {t0_audit['hallucination_rate']}%")
        print(f"  Tier 1 (Vector RAG): Faithfulness = {t1_audit['faithfulness_score']}% | Hallucination = {t1_audit['hallucination_rate']}%")
        print(f"  Tier 3 (Agentic Provenance): Faithfulness = {t3_audit['faithfulness_score']}% | Hallucination = {t3_audit['hallucination_rate']}%")

    # Summary
    avg_t0_faith = round(sum(r["tier0_direct_llm"]["faithfulness_score"] for r in results) / len(results), 1)
    avg_t1_faith = round(sum(r["tier1_vector_rag"]["faithfulness_score"] for r in results) / len(results), 1)
    avg_t3_faith = round(sum(r["tier3_agentic_provenance"]["faithfulness_score"] for r in results) / len(results), 1)

    avg_t0_halluc = round(sum(r["tier0_direct_llm"]["hallucination_rate"] for r in results) / len(results), 1)
    avg_t1_halluc = round(sum(r["tier1_vector_rag"]["hallucination_rate"] for r in results) / len(results), 1)
    avg_t3_halluc = round(sum(r["tier3_agentic_provenance"]["hallucination_rate"] for r in results) / len(results), 1)

    summary = {
        "benchmark_date": "2026-09-15",
        "sample_size": len(results),
        "overall_summary": {
            "tier0_direct_llm": {
                "avg_faithfulness": avg_t0_faith,
                "avg_hallucination_rate": avg_t0_halluc
            },
            "tier1_vector_rag": {
                "avg_faithfulness": avg_t1_faith,
                "avg_hallucination_rate": avg_t1_halluc
            },
            "tier3_agentic_provenance": {
                "avg_faithfulness": avg_t3_faith,
                "avg_hallucination_rate": avg_t3_halluc
            },
            "provenance_improvement_vs_vector_rag": f"+{round(avg_t3_faith - avg_t1_faith, 1)}%"
        },
        "per_query_results": results
    }

    out_path = root_dir / "Provenance_Scoring" / "provenance_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print(" BENCHMARK COMPLETED & SAVED TO provenance_results.json")
    print(f" Summary: Direct LLM Faithfulness: {avg_t0_faith}% | Vector RAG: {avg_t1_faith}% | Agentic Provenance: {avg_t3_faith}%")
    print("=" * 80)

if __name__ == "__main__":
    run_benchmark()
