"""
Structured Evidence Formatting Ablation Benchmark (Enhanced Scientific Evaluation)
Quantifies Contribution #2 (Structured Markdown Tables vs Raw Flat Triples).

Evaluates 3 Core Research Metrics:
  1. Distractor Resistance & Entity Prioritization (%):
     - When target facts are mixed with non-target distractor triples, does the model
       correctly prioritize the target entity type over distractors?
  2. Categorical Structure & Coherence Score (0 to 10):
     - Evaluates whether the answer is organized into domain-specific categories vs a flat unranked dump.
  3. Inference Latency (seconds):
     - Measures generation speed improvements from attention-aligned table prompts.

Outputs:
  - formatting_ablation_results.json
  - Slide-ready Comparison Table for Review Presentation
"""

import os
import sys
import json
import time
import re
import numpy as np
from dotenv import load_dotenv, find_dotenv
from groq import Groq

# Ensure parent imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from Structured_Formatting.formatter import format_structured_evidence

load_dotenv(find_dotenv(), override=True)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")

groq_client = Groq(api_key=GROQ_API_KEY)


# Benchmark Questions designed with controlled target triples + real distractors
EVALUATION_SUITE = [
    {
        "id": "E1",
        "category": "Regulatory Prioritization",
        "query": "Which regulatory bodies oversee Apple regarding its financial disclosures?",
        "target_entity_type": "ORG_REG",
        "facts_pool": [
            "[COMP] channel partner --negatively_impacts--> [ORG] aapl",
            "[CONCEPT] price reduction --negatively_impacts--> [ORG] aapl",
            "[ORG_REG] sec --regulates--> [ORG] aapl",
            "[ORG_REG] u.s. department of justice --regulates--> [ORG] aapl",
            "[REGULATORY_REQUIREMENT] sec rule 10b-5 --subject_to--> [ORG] aapl",
            "[FIN_MARKET] liquidity --negatively_impacts--> [ORG] aapl"
        ],
        "gold_targets": ["sec", "department of justice"],
        "distractors": ["channel partner", "price reduction", "liquidity"]
    },
    {
        "id": "E2",
        "category": "Market Risk vs General Concepts",
        "query": "What financial market conditions does Apple disclose as negatively impacting its financial metrics?",
        "target_entity_type": "FIN_MARKET",
        "facts_pool": [
            "[CONCEPT] frequent product introduction --impacted_by--> [ORG] aapl",
            "[COMP] foreign currency exchange rate --impacted_by--> [ORG] aapl",
            "[FIN_MARKET] liquidity --negatively_impacts--> [ORG] aapl",
            "[FIN_MARKET] credit market --negatively_impacts--> [ORG] aapl",
            "[RISK_FACTOR] single or limited source for component --impacted_by--> [ORG] aapl",
            "[CONCEPT] rapid technological advance --impacted_by--> [ORG] aapl"
        ],
        "gold_targets": ["liquidity", "credit market"],
        "distractors": ["product introduction", "technological advance", "limited source"]
    },
    {
        "id": "E3",
        "category": "Litigation vs General Requirements",
        "query": "What specific litigation proceeding is Apple subject to regarding antitrust?",
        "target_entity_type": "LITIGATION",
        "facts_pool": [
            "[REGULATORY_REQUIREMENT] injunction --subject_to--> [ORG] aapl",
            "[REGULATORY_REQUIREMENT] local law --subject_to--> [ORG] aapl",
            "[LITIGATION] apple ebooks antitrust litigation --subject_to--> [ORG] aapl",
            "[ORG_REG] u.s. district court for the southern district of new york --regulates--> [ORG] aapl",
            "[ACCOUNTING_POLICY] revenue recognition policy --discloses--> [ORG] aapl"
        ],
        "gold_targets": ["apple ebooks antitrust litigation"],
        "distractors": ["injunction", "local law", "revenue recognition policy"]
    },
    {
        "id": "E4",
        "category": "Financial Metric Disclosures",
        "query": "Which key financial metrics does Microsoft disclose in its SEC reports?",
        "target_entity_type": "FIN_METRIC",
        "facts_pool": [
            "[PERSON] satya nadella --works_for--> [ORG] msft",
            "[FIN_METRIC] net income --discloses--> [ORG] msft",
            "[FIN_METRIC] operate income --discloses--> [ORG] msft",
            "[ACCOUNTING_POLICY] stock-based compensation --discloses--> [ORG] msft",
            "[PRODUCT] azure cloud platform --produce--> [ORG] msft"
        ],
        "gold_targets": ["net income", "operate income"],
        "distractors": ["satya nadella", "stock-based compensation", "azure"]
    }
]


def score_categorical_structure(answer: str) -> float:
    """
    Evaluates formatting structure quality on a 0 to 10 scale:
      - Has categorized bold sections or headers (+4)
      - Has clean bullet hierarchy (+3)
      - Has entity-type annotations or badges (+3)
    """
    score = 0.0
    # Check for bold categories or markdown headers
    if re.search(r"(\*\*.*?:?\*\*|###\s+.*)", answer):
        score += 4.0
    # Check for structured lists
    if re.search(r"(\*\s+|\d+\.\s+)", answer):
        score += 3.0
    # Check for entity-type tags like `[ORG_REG]` or `(ORG)`
    if re.search(r"(\[[A-Z_]+\]|\([A-Z_]+\))", answer):
        score += 3.0
    return min(score, 10.0)


def evaluate_response(answer: str, gold_targets: list, distractors: list) -> dict:
    ans_lower = answer.lower()
    
    # 1. Target Entity Recall
    target_hits = sum(1 for g in gold_targets if g in ans_lower)
    target_recall = (target_hits / len(gold_targets)) * 100

    # 2. Distractor Resistance (100% if no distractors mistakenly presented as primary answers)
    distractor_mentions = sum(1 for d in distractors if d in ans_lower)
    distractor_resistance = max(0.0, 100.0 - (distractor_mentions * (100.0 / max(len(distractors), 1))))

    # 3. Structure Score
    struct_score = score_categorical_structure(answer)

    return {
        "target_recall_pct": round(target_recall, 1),
        "target_hits": f"{target_hits}/{len(gold_targets)}",
        "distractor_resistance_pct": round(distractor_resistance, 1),
        "distractor_mentions": distractor_mentions,
        "structure_score_out_of_10": struct_score
    }


def run_llm(prompt_context: str, query: str) -> tuple:
    prompt = f"""You are a financial research analyst assistant. Answer the question using ONLY the provided Knowledge Graph evidence.
Highlight the specific requested entities clearly.

Evidence:
{prompt_context}

Question: {query}"""

    t0 = time.time()
    resp = groq_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=300
    )
    latency = time.time() - t0
    return resp.choices[0].message.content.strip(), round(latency, 2)


def main():
    print("=" * 90)
    print("ENHANCED STRUCTURED EVIDENCE FORMATTING ABLATION BENCHMARK")
    print("Comparing: Raw Flat Triples (Baseline) vs Structured Markdown Tables (Proposed)")
    print("=" * 90)

    results = []

    for item in EVALUATION_SUITE:
        qid = item["id"]
        cat = item["category"]
        q = item["query"]
        facts = item["facts_pool"]
        gold = item["gold_targets"]
        dist = item["distractors"]

        print(f"\n>>> [{qid}] ({cat}): {q}")

        # 1. Condition A: Raw Flat Triples (Baseline)
        raw_context = "\n".join(facts)
        ans_raw, lat_raw = run_llm(raw_context, q)
        eval_raw = evaluate_response(ans_raw, gold, dist)

        # 2. Condition B: Structured Markdown Tables (Proposed)
        struct_context = format_structured_evidence(facts, query=q)
        ans_struct, lat_struct = run_llm(struct_context, q)
        eval_struct = evaluate_response(ans_struct, gold, dist)

        print(f"  [Raw Triples Baseline]        Target Recall: {eval_raw['target_recall_pct']}% | Distractor Resistance: {eval_raw['distractor_resistance_pct']}% | Structure: {eval_raw['structure_score_out_of_10']}/10 | Latency: {lat_raw}s")
        print(f"  [Structured Markdown Tables]  Target Recall: {eval_struct['target_recall_pct']}% | Distractor Resistance: {eval_struct['distractor_resistance_pct']}% | Structure: {eval_struct['structure_score_out_of_10']}/10 | Latency: {lat_struct}s")

        results.append({
            "id": qid,
            "category": cat,
            "query": q,
            "raw_baseline": {
                "answer": ans_raw,
                "latency_sec": lat_raw,
                "metrics": eval_raw
            },
            "structured_proposed": {
                "answer": ans_struct,
                "latency_sec": lat_struct,
                "metrics": eval_struct
            },
            "deltas": {
                "recall_gain": round(eval_struct["target_recall_pct"] - eval_raw["target_recall_pct"], 1),
                "distractor_resistance_gain": round(eval_struct["distractor_resistance_pct"] - eval_raw["distractor_resistance_pct"], 1),
                "structure_gain": round(eval_struct["structure_score_out_of_10"] - eval_raw["structure_score_out_of_10"], 1),
                "latency_reduction_pct": round(((lat_raw - lat_struct) / lat_raw) * 100, 1)
            }
        })

    # Save to JSON
    out_file = os.path.join(current_dir, "formatting_ablation_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    # Print Summary Table
    print("\n" + "=" * 95)
    print("STRUCTURED EVIDENCE FORMATTING: ABLATION BENCHMARK RESULTS")
    print("=" * 95)
    print(f"{'QID':<4} | {'Category':<26} | {'Target Recall':<15} | {'Distractor Resist':<18} | {'Structure Score':<16} | {'Latency':<10}")
    print(f"{'':<4} | {'':<26} | {'Raw -> Struct':<15} | {'Raw -> Struct':<18} | {'Raw -> Struct':<16} | {'Raw -> Struct':<10}")
    print("-" * 95)

    for r in results:
        raw_m = r["raw_baseline"]["metrics"]
        str_m = r["structured_proposed"]["metrics"]
        print(f"{r['id']:<4} | {r['category']:<26} | {raw_m['target_recall_pct']:>5.0f}% -> {str_m['target_recall_pct']:>5.0f}% | {raw_m['distractor_resistance_pct']:>6.0f}% -> {str_m['distractor_resistance_pct']:>6.0f}% | {raw_m['structure_score_out_of_10']:>5.1f} -> {str_m['structure_score_out_of_10']:>5.1f}/10 | {r['raw_baseline']['latency_sec']:>4.2f}s -> {r['structured_proposed']['latency_sec']:>4.2f}s")

    avg_raw_rec = np.mean([r["raw_baseline"]["metrics"]["target_recall_pct"] for r in results])
    avg_str_rec = np.mean([r["structured_proposed"]["metrics"]["target_recall_pct"] for r in results])
    
    avg_raw_dis = np.mean([r["raw_baseline"]["metrics"]["distractor_resistance_pct"] for r in results])
    avg_str_dis = np.mean([r["structured_proposed"]["metrics"]["distractor_resistance_pct"] for r in results])

    avg_raw_str = np.mean([r["raw_baseline"]["metrics"]["structure_score_out_of_10"] for r in results])
    avg_str_str = np.mean([r["structured_proposed"]["metrics"]["structure_score_out_of_10"] for r in results])

    avg_raw_lat = np.mean([r["raw_baseline"]["latency_sec"] for r in results])
    avg_str_lat = np.mean([r["structured_proposed"]["latency_sec"] for r in results])

    print("-" * 95)
    print(f"{'AVG':<4} | {'OVERALL WORKLOAD AVERAGE':<26} | {avg_raw_rec:>5.1f}% -> {avg_str_rec:>5.1f}% | {avg_raw_dis:>6.1f}% -> {avg_str_dis:>6.1f}% | {avg_raw_str:>5.1f} -> {avg_str_str:>5.1f}/10 | {avg_raw_lat:>4.2f}s -> {avg_str_lat:>4.2f}s")
    print("=" * 95)
    print(f"Full benchmark data saved to: {out_file}\n")


if __name__ == "__main__":
    main()
