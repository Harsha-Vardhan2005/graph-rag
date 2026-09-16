"""
Format Comparison Test — same gold facts, two evidence formats:
  (A) Clean markdown table (what we tested before)
  (B) Raw unordered triples + distractors (what the actual KG pipeline retrieves)

If accuracy drops meaningfully from (A) to (B), that's direct evidence that
evidence FORMAT (not raw reasoning capacity) drives the retrieval-vs-reasoning
gap — supporting contribution #2 (structured evidence formatting as the fix).

Usage:
    pip install groq
    export GROQ_API_KEY=your_key_here
    python compare_formats.py
"""

import json
import os
import re
from dotenv import load_dotenv, find_dotenv
from groq import Groq

load_dotenv(find_dotenv(), override=True)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
MODEL = "llama-3.1-8b-instant"

CLEAN_PROMPT = """You are a financial analyst assistant. Answer the question using ONLY the evidence provided below. Do not use any outside knowledge.

Evidence:
{evidence}

Question: {question}

Give a direct, concise answer with the specific number(s) requested. If the question requires a calculation, show the calculation briefly, then state the final answer clearly at the end as "Final Answer: <answer>".
"""

RAW_PROMPT = """You are a financial analyst assistant. Answer the question using ONLY the knowledge graph triples provided below. The triples are unordered and may include irrelevant ones — ignore what isn't needed. Do not use any outside knowledge.

Triples:
{evidence}

Question: {question}

Give a direct, concise answer with the specific number(s) requested. If the question requires a calculation, show the calculation briefly, then state the final answer clearly at the end as "Final Answer: <answer>".
"""


def call_llm(client, prompt_template, evidence, question):
    prompt = prompt_template.format(evidence=evidence, question=question)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=500,
    )
    return response.choices[0].message.content


def extract_final_answer(text):
    match = re.search(r"Final Answer:\s*(.+)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()


def extract_number(text):
    cleaned = text.replace(",", "").replace("$", "")
    match = re.search(r"-?\d+\.?\d*", cleaned)
    if match:
        return float(match.group())
    return None


def score(gold_num, model_output):
    final_answer_text = extract_final_answer(model_output)
    model_num = extract_number(final_answer_text)
    if model_num is None:
        return False, final_answer_text
    tolerance = max(abs(gold_num) * 0.005, 0.01)
    return abs(model_num - gold_num) <= tolerance, final_answer_text


def main():
    with open("gold_qa_set.json", "r") as f:
        clean_set = {item["id"]: item for item in json.load(f)}
    with open("raw_triplet_qa_set.json", "r") as f:
        raw_set = {item["id"]: item for item in json.load(f)}

    client = Groq(api_key=GROQ_API_KEY)

    results = []
    clean_correct, raw_correct = 0, 0
    by_difficulty = {}

    for qid, clean_item in clean_set.items():
        raw_item = raw_set[qid]
        question = clean_item["question"]
        gold_num = clean_item["gold_answer_numeric"]
        difficulty = clean_item["difficulty"]

        print(f"\n{'='*70}\n[{qid}] {question}")

        clean_out = call_llm(client, CLEAN_PROMPT, clean_item["evidence"], question)
        clean_ok, clean_ans = score(gold_num, clean_out)

        raw_evidence_str = "\n".join(raw_item["raw_triples"])
        raw_out = call_llm(client, RAW_PROMPT, raw_evidence_str, question)
        raw_ok, raw_ans = score(gold_num, raw_out)

        print(f"  CLEAN table  -> {clean_ans}  [{'CORRECT' if clean_ok else 'WRONG'}]")
        print(f"  RAW triples  -> {raw_ans}  [{'CORRECT' if raw_ok else 'WRONG'}]")

        clean_correct += int(clean_ok)
        raw_correct += int(raw_ok)

        by_difficulty.setdefault(difficulty, {"clean": 0, "raw": 0, "total": 0})
        by_difficulty[difficulty]["total"] += 1
        by_difficulty[difficulty]["clean"] += int(clean_ok)
        by_difficulty[difficulty]["raw"] += int(raw_ok)

        results.append({
            "id": qid, "question": question, "difficulty": difficulty,
            "gold_answer": clean_item["gold_answer"],
            "clean_answer": clean_ans, "clean_correct": clean_ok,
            "raw_answer": raw_ans, "raw_correct": raw_ok,
        })

    n = len(clean_set)
    print(f"\n\n{'='*70}")
    print(f"CLEAN TABLE ACCURACY: {clean_correct}/{n} = {100*clean_correct/n:.1f}%")
    print(f"RAW TRIPLES ACCURACY: {raw_correct}/{n} = {100*raw_correct/n:.1f}%")
    print(f"GAP: {clean_correct - raw_correct} questions ({100*(clean_correct-raw_correct)/n:.1f} pp)")
    print(f"\nBreakdown by difficulty:")
    for diff, stats in by_difficulty.items():
        print(f"  {diff}: clean={stats['clean']}/{stats['total']}  raw={stats['raw']}/{stats['total']}")

    with open("format_comparison_results.json", "w") as f:
        json.dump({
            "clean_accuracy": clean_correct / n,
            "raw_accuracy": raw_correct / n,
            "by_difficulty": by_difficulty,
            "results": results,
        }, f, indent=2)
    print("\nFull results saved to format_comparison_results.json")


if __name__ == "__main__":
    main()
