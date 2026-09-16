"""
Oracle Retrieval Test — isolates reasoning quality from retrieval quality.
Feeds gold evidence directly to the LLM (no retrieval, no agent, no tools)
and checks if it can reason to the correct answer.

Usage:
    pip install groq
    export GROQ_API_KEY=your_key_here     (or set below directly)
    python oracle_test.py
"""

import json
import os
import re
from dotenv import load_dotenv, find_dotenv
from groq import Groq

load_dotenv(find_dotenv(), override=True)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
MODEL = "qwen/qwen3.8-27b"

ORACLE_PROMPT_TEMPLATE = """You are a financial analyst assistant. Answer the question using ONLY the evidence provided below. Do not use any outside knowledge.

Evidence:
{evidence}

Question: {question}

Give a direct, concise answer with the specific number(s) requested. If the question requires a calculation, show the calculation briefly, then state the final answer clearly at the end as "Final Answer: <answer>".
"""


def call_llm(client, evidence, question):
    prompt = ORACLE_PROMPT_TEMPLATE.format(evidence=evidence, question=question)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=500,
    )
    return response.choices[0].message.content


def extract_final_answer(text):
    """Pull out the 'Final Answer:' line if present, else return full text."""
    match = re.search(r"Final Answer:\s*(.+)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()


def extract_number(text):
    """Extract the first numeric value from a string, handling $, commas, %."""
    cleaned = text.replace(",", "").replace("$", "")
    match = re.search(r"-?\d+\.?\d*", cleaned)
    if match:
        return float(match.group())
    return None


def score_answer(item, model_output):
    final_answer_text = extract_final_answer(model_output)
    model_num = extract_number(final_answer_text)
    gold_num = item["gold_answer_numeric"]

    if model_num is None:
        return False, final_answer_text, "no_number_found"

    # Allow small relative tolerance (0.5%) for rounding differences
    tolerance = max(abs(gold_num) * 0.005, 0.01)
    is_correct = abs(model_num - gold_num) <= tolerance

    return is_correct, final_answer_text, f"model={model_num}, gold={gold_num}"


def main():
    with open("gold_qa_set.json", "r") as f:
        qa_set = json.load(f)

    client = Groq(api_key=GROQ_API_KEY)

    results = []
    correct_count = 0
    by_difficulty = {}

    for item in qa_set:
        print(f"\n{'='*70}")
        print(f"[{item['id']}] {item['question']}")

        raw_output = call_llm(client, item["evidence"], item["question"])
        is_correct, extracted, detail = score_answer(item, raw_output)

        print(f"  Model answer: {extracted}")
        print(f"  Gold answer:  {item['gold_answer']}")
        print(f"  Correct: {is_correct}  ({detail})")

        results.append({
            "id": item["id"],
            "question": item["question"],
            "difficulty": item["difficulty"],
            "gold_answer": item["gold_answer"],
            "model_raw_output": raw_output,
            "model_extracted_answer": extracted,
            "is_correct": is_correct,
            "detail": detail,
        })

        correct_count += int(is_correct)
        diff = item["difficulty"]
        by_difficulty.setdefault(diff, {"correct": 0, "total": 0})
        by_difficulty[diff]["total"] += 1
        by_difficulty[diff]["correct"] += int(is_correct)

    print(f"\n\n{'='*70}")
    print(f"ORACLE ACCURACY: {correct_count}/{len(qa_set)} = {100*correct_count/len(qa_set):.1f}%")
    print(f"\nBreakdown by difficulty:")
    for diff, stats in by_difficulty.items():
        pct = 100 * stats["correct"] / stats["total"]
        print(f"  {diff}: {stats['correct']}/{stats['total']} = {pct:.1f}%")

    with open("oracle_results.json", "w") as f:
        json.dump({
            "overall_accuracy": correct_count / len(qa_set),
            "by_difficulty": by_difficulty,
            "results": results,
        }, f, indent=2)
    print("\nFull results saved to oracle_results.json")


if __name__ == "__main__":
    main()
