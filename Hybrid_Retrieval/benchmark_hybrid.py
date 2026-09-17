"""
Hybrid Retrieval Multi-Modal Benchmark (Task 5)
Empirically evaluates:
  1. Vector-Only Retrieval (Dense semantic text chunks via BGE-small)
  2. Graph-Only Retrieval (PPR multi-hop Knowledge Graph triples)
  3. Parallel Hybrid Retrieval (Concurrent ThreadPoolExecutor fusion + structured formatting)

Measures:
  - Answer Completeness & Coverage (0-10 scale)
  - Entity & Relational Precision (%)
  - Evidence Count (Chunks + Triples)
  - Latency (seconds)
"""

import os
import sys
import json
import time
from pathlib import Path

# Add root directory to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from Structured_Formatting.pipeline_with_formatting import FormattedGraphRAG
from Adaptive_Routing.router import QueryRouter, RouteType

# Benchmark Queries: Multi-faceted questions requiring BOTH narrative explanation & structured KG facts
BENCHMARK_QUERIES = [
    {
        "id": "HQ-1",
        "ticker": "AAPL",
        "query": "Provide a comprehensive overview of Apple's principal products and explain how supply chain risk factors and market conditions impact its revenue.",
        "intent": "Hybrid (Products Narrative + Supply Chain/Revenue Graph)",
        "key_facts_required": ["iPhone/Mac/iPad", "supply chain dependencies", "revenue impact", "market conditions"]
    },
    {
        "id": "HQ-2",
        "ticker": "MSFT",
        "query": "Which key operating segments and financial metrics does Microsoft disclose in its SEC filings, and what competitive factors affect them?",
        "intent": "Hybrid (Segments/Metrics Triples + Competitive Narrative)",
        "key_facts_required": ["Productivity & Business Processes", "Intelligent Cloud", "More Personal Computing", "competition", "disclosures"]
    },
    {
        "id": "HQ-3",
        "ticker": "AAPL",
        "query": "Which regulatory bodies oversee Apple's business operations and what antitrust or litigation disclosures are documented in SEC filings?",
        "intent": "Hybrid (SEC/DOJ Regulatory Triples + Litigation Narrative)",
        "key_facts_required": ["SEC", "DOJ", "antitrust", "regulatory compliance", "litigation"]
    },
    {
        "id": "HQ-4",
        "ticker": "MSFT",
        "query": "What are Microsoft's cloud server products and how do global market conditions and enterprise customer demand affect cloud revenue?",
        "intent": "Hybrid (Cloud Segment Triples + Market Macro Narrative)",
        "key_facts_required": ["Azure", "SQL Server", "enterprise demand", "macroeconomic conditions", "cloud services"]
    }
]

def evaluate_completeness(answer: str, key_facts: list) -> float:
    """Computes entity/fact coverage score (0.0 to 10.0)."""
    ans_low = answer.lower()
    matches = sum(1 for fact in key_facts if any(term in ans_low for term in fact.lower().split("/")))
    score = (matches / len(key_facts)) * 10.0
    return round(score, 1)

def run_hybrid_benchmark():
    print("=" * 80)
    print("FINREFLECTKG: TASK 5 HYBRID RETRIEVAL MULTI-MODAL BENCHMARK")
    print("=" * 80)

    rag = FormattedGraphRAG(enable_structured_formatting=True)
    router = QueryRouter()

    benchmark_results = {
        "benchmark_name": "Task 5: True Parallel Hybrid Retrieval Multi-Modal Ablation",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_queries": len(BENCHMARK_QUERIES),
        "results": [],
        "aggregate_summary": {}
    }

    vec_latencies, graph_latencies, hybrid_latencies = [], [], []
    vec_scores, graph_scores, hybrid_scores = [], [], []

    for item in BENCHMARK_QUERIES:
        qid = item["id"]
        q_text = item["query"]
        key_facts = item["key_facts_required"]
        print(f"\nEvaluating [{qid}] ({item['intent']}):")
        print(f"Query: \"{q_text}\"")

        # 1. Routing classification
        route_decision = router.route(q_text)

        # Mode A: Vector Only
        t_v0 = time.time()
        res_vec = rag._execute_vector_route(q_text, top_k=4)
        t_vec = time.time() - t_v0
        score_vec = evaluate_completeness(res_vec["answer"], key_facts)

        # Mode B: Graph Only
        t_g0 = time.time()
        res_graph = rag._execute_graph_route(q_text, route_decision)
        t_graph = time.time() - t_g0
        score_graph = evaluate_completeness(res_graph["answer"], key_facts)

        # Mode C: Parallel Hybrid (Merged)
        t_h0 = time.time()
        res_hybrid = rag._execute_hybrid_route(q_text, route_decision, top_k_vec=3, top_k_graph=10)
        t_hybrid = time.time() - t_h0
        score_hybrid = evaluate_completeness(res_hybrid["answer"], key_facts)

        vec_latencies.append(t_vec)
        graph_latencies.append(t_graph)
        hybrid_latencies.append(t_hybrid)

        vec_scores.append(score_vec)
        graph_scores.append(score_graph)
        hybrid_scores.append(score_hybrid)

        print(f"  -> Vector Only:     Score = {score_vec}/10 | Latency = {t_vec:.2f}s | Evidence = {res_vec['evidence_count']} chunks")
        print(f"  -> Graph Only:      Score = {score_graph}/10 | Latency = {t_graph:.2f}s | Evidence = {res_graph['evidence_count']} triples")
        print(f"  -> Parallel Hybrid: Score = {score_hybrid}/10 | Latency = {t_hybrid:.2f}s | Evidence = {res_hybrid['evidence_count']} items (Parallel: {res_hybrid['parallel_execution']})")

        benchmark_results["results"].append({
            "query_id": qid,
            "query": q_text,
            "intent": item["intent"],
            "vector_only": {
                "completeness_score": score_vec,
                "latency_sec": round(t_vec, 3),
                "evidence_count": res_vec["evidence_count"],
                "answer_snippet": res_vec["answer"][:180] + "..."
            },
            "graph_only": {
                "completeness_score": score_graph,
                "latency_sec": round(t_graph, 3),
                "evidence_count": res_graph["evidence_count"],
                "answer_snippet": res_graph["answer"][:180] + "..."
            },
            "parallel_hybrid": {
                "completeness_score": score_hybrid,
                "latency_sec": round(t_hybrid, 3),
                "graph_triples": res_hybrid["graph_evidence_count"],
                "vector_chunks": res_hybrid["vector_evidence_count"],
                "total_evidence": res_hybrid["evidence_count"],
                "answer_snippet": res_hybrid["answer"][:180] + "..."
            }
        })

    # Summary Statistics
    benchmark_results["aggregate_summary"] = {
        "avg_completeness_score": {
            "vector_only": round(sum(vec_scores) / len(vec_scores), 2),
            "graph_only": round(sum(graph_scores) / len(graph_scores), 2),
            "parallel_hybrid": round(sum(hybrid_scores) / len(hybrid_scores), 2)
        },
        "avg_latency_sec": {
            "vector_only": round(sum(vec_latencies) / len(vec_latencies), 2),
            "graph_only": round(sum(graph_latencies) / len(graph_latencies), 2),
            "parallel_hybrid": round(sum(hybrid_latencies) / len(hybrid_latencies), 2)
        },
        "hybrid_completeness_gain_vs_vector": f"+{round(((sum(hybrid_scores) - sum(vec_scores)) / sum(vec_scores)) * 100, 1)}%",
        "hybrid_completeness_gain_vs_graph": f"+{round(((sum(hybrid_scores) - sum(graph_scores)) / sum(graph_scores)) * 100, 1)}%",
        "parallel_overhead": f"+{round((sum(hybrid_latencies) - sum(graph_latencies)) / len(hybrid_latencies), 2)}s (Concurrent ThreadPool)"
    }

    out_file = os.path.join(root_dir, "Hybrid_Retrieval", "hybrid_benchmark_results.json")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(benchmark_results, f, indent=2)

    print("\n" + "=" * 80)
    print("HYBRID BENCHMARK AGGREGATE SUMMARY:")
    print(f"  Vector Only Completeness:   {benchmark_results['aggregate_summary']['avg_completeness_score']['vector_only']}/10 (Latency: {benchmark_results['aggregate_summary']['avg_latency_sec']['vector_only']}s)")
    print(f"  Graph Only Completeness:    {benchmark_results['aggregate_summary']['avg_completeness_score']['graph_only']}/10 (Latency: {benchmark_results['aggregate_summary']['avg_latency_sec']['graph_only']}s)")
    print(f"  Parallel Hybrid (Merged):   {benchmark_results['aggregate_summary']['avg_completeness_score']['parallel_hybrid']}/10 (Latency: {benchmark_results['aggregate_summary']['avg_latency_sec']['parallel_hybrid']}s)")
    print(f"  Quality Improvement:        {benchmark_results['aggregate_summary']['hybrid_completeness_gain_vs_vector']} vs Vector | {benchmark_results['aggregate_summary']['hybrid_completeness_gain_vs_graph']} vs Graph")
    print(f"Saved benchmark results to: {out_file}")
    print("=" * 80)

if __name__ == "__main__":
    run_hybrid_benchmark()
