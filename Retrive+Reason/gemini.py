"""
Hard Oracle Test — Gemini version (cross-model comparison, not your production model).

Usage:
    pip install google-genai
    python hard_oracle_test_gemini.py
"""

import os
import json
import re
from dotenv import load_dotenv, find_dotenv
from google import genai

load_dotenv(find_dotenv(), override=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = "gemini-3.5-flash"

PROMPT_TEMPLATE = """You are a financial analyst assistant. Answer the question using ONLY the knowledge graph triples provided below. The triples are unordered, cover multiple fiscal years, and include some GAAP vs non-GAAP variants and segment-level data — many are irrelevant to any given question, so identify precisely which ones you need. Do not use any outside knowledge.

Triples:
{evidence}

Question: {question}

Show your work briefly (which triples you used and any calculation), then give your final answer clearly as "Final Answer: <answer>".
"""


def call_llm(client, evidence, question):
    prompt = PROMPT_TEMPLATE.format(evidence=evidence, question=question)
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
    )
    return response.text


def extract_final_answer(text):
    match = re.search(r"Final Answer:\s*(.+)", text, re.IGNORECASE)
    return match.group(1).strip() if match else text.strip()


def extract_number(text):
    cleaned = text.replace(",", "").replace("$", "").replace("%", "")
    match = re.search(r"-?\d+\.?\d*", cleaned)
    return float(match.group()) if match else None


def score_numeric(gold_num, model_output):
    final_text = extract_final_answer(model_output)
    model_num = extract_number(final_text)
    if model_num is None:
        return False, final_text
    tolerance = max(abs(gold_num) * 0.01, 0.05)
    return abs(model_num - gold_num) <= tolerance, final_text


def main():
    with open("hard_evidence_pool.json", "r") as f:
        pool = json.load(f)
    evidence_str = "\n".join(pool)
    print(f"Evidence pool size: {len(pool)} triples\n")

    with open("hard_qa_set.json", "r") as f:
        qa_set = json.load(f)

    client = genai.Client(api_key=GEMINI_API_KEY)

    results = []
    correct_count = 0
    flagged = []

    for item in qa_set:
        print(f"\n{'='*70}")
        print(f"[{item['id']}] ({item['difficulty']}) {item['question']}")

        raw_output = call_llm(client, evidence_str, item["question"])

        if item["type"] == "numeric":
            is_correct, extracted = score_numeric(item["gold_answer_numeric"], raw_output)
        else:
            extracted = extract_final_answer(raw_output)
            is_correct = None

        print(f"  Model final answer: {extracted}")
        print(f"  Gold answer:        {item['gold_answer']}")
        if is_correct is not None:
            status = "CORRECT" if is_correct else "*** WRONG ***"
            print(f"  Status: {status}")
            if not is_correct:
                flagged.append(item["id"])
            correct_count += int(is_correct)
        else:
            print(f"  Status: NEEDS MANUAL REVIEW")

        print(f"\n  --- Full reasoning trace ---")
        print(f"  {raw_output}")

        results.append({
            "id": item["id"],
            "question": item["question"],
            "difficulty": item["difficulty"],
            "gold_answer": item["gold_answer"],
            "model_extracted_answer": extracted,
            "is_correct": is_correct,
            "full_raw_output": raw_output,
        })

    numeric_total = sum(1 for i in qa_set if i["type"] == "numeric")
    print(f"\n\n{'='*70}")
    print(f"AUTO-SCORED ACCURACY (numeric only): {correct_count}/{numeric_total} = {100*correct_count/numeric_total:.1f}%")
    print(f"FLAGGED FOR REVIEW (wrong): {flagged}")
    print(f"Categorical questions need manual check (see full output above).")

    with open("hard_oracle_results_gemini.json", "w") as f:
        json.dump({
            "model": MODEL,
            "evidence_pool_size": len(pool),
            "numeric_accuracy": correct_count / numeric_total,
            "flagged": flagged,
            "results": results,
        }, f, indent=2)
    print("\nFull results (with reasoning traces) saved to hard_oracle_results_gemini.json")


if __name__ == "__main__":
    main()