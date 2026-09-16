"""
FinQA Oracle Test — runs your production model (llama-3.3-70b-versatile) against
real, peer-reviewed multi-step financial reasoning questions with gold evidence.

Usage:
    pip install groq
    export GROQ_API_KEY=your_key_here
    python finqa_oracle_test.py
    (requires finqa_oracle_set.json from load_finqa.py in the same folder)
"""

import json
import os
import re
from dotenv import load_dotenv, find_dotenv
from groq import Groq

load_dotenv(find_dotenv(), override=True)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
MODEL = "qwen/qwen3.8-27b"

PROMPT_TEMPLATE = """You are a financial analyst assistant. Answer the question using ONLY the evidence provided below (text and table from a financial report). Do not use any outside knowledge.

Evidence:
{evidence}

Question: {question}

Show your calculation briefly, then give your final numeric answer clearly as "Final Answer: <number>". If the answer is a percentage, express it as a number followed by %.
"""


def call_llm(client, evidence, question):
    prompt = PROMPT_TEMPLATE.format(evidence=evidence, question=question)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=600,
    )
    return response.choices[0].message.content


def extract_final_answer(text):
    match = re.search(r"Final Answer:\s*(.+)", text, re.IGNORECASE)
    return match.group(1).strip() if match else text.strip()


def extract_number(text):
    cleaned = text.replace(",", "").replace("$", "")
    is_pct = "%" in cleaned
    cleaned = cleaned.replace("%", "")
    match = re.search(r"-?\d+\.?\d*", cleaned)
    if not match:
        return None
    num = float(match.group())
    return num


def score(gold_num, model_output):
    final_text = extract_final_answer(model_output)
    model_num = extract_number(final_text)
    if model_num is None:
        return False, final_text

    # FinQA exe_ans for percentages is often expressed as a decimal (e.g. 0.16 for 16%)
    # try both direct match and *100 / /100 to handle format mismatches
    tolerance = max(abs(gold_num) * 0.02, 0.01)
    candidates = [model_num, model_num / 100, model_num * 100]
    is_correct = any(abs(c - gold_num) <= tolerance for c in candidates)
    return is_correct, final_text


def main():
    with open("finqa_oracle_set.json", "r") as f:
        qa_set = json.load(f)

    client = Groq(api_key=GROQ_API_KEY)

    results = []
    correct_count = 0
    by_steps = {}

    for item in qa_set:
        print(f"\n{'='*70}")
        print(f"[{item['id']}] ({item['num_steps']} steps) {item['question']}")
        print(f"  Gold program: {item['gold_program']}")

        raw_output = call_llm(client, item["evidence"], item["question"])
        is_correct, extracted = score(item["gold_answer_numeric"], raw_output)

        print(f"  Model answer: {extracted}")
        print(f"  Gold answer:  {item['gold_answer_numeric']}")
        print(f"  Status: {'CORRECT' if is_correct else '*** WRONG ***'}")

        correct_count += int(is_correct)
        steps = item["num_steps"]
        by_steps.setdefault(steps, {"correct": 0, "total": 0})
        by_steps[steps]["total"] += 1
        by_steps[steps]["correct"] += int(is_correct)

        results.append({
            "id": item["id"],
            "question": item["question"],
            "num_steps": item["num_steps"],
            "gold_program": item["gold_program"],
            "gold_answer": item["gold_answer_numeric"],
            "model_answer": extracted,
            "is_correct": is_correct,
            "full_raw_output": raw_output,
        })

    n = len(qa_set)
    print(f"\n\n{'='*70}")
    print(f"OVERALL ACCURACY: {correct_count}/{n} = {100*correct_count/n:.1f}%")
    print(f"\nBreakdown by number of reasoning steps:")
    for steps in sorted(by_steps.keys()):
        stats = by_steps[steps]
        pct = 100 * stats["correct"] / stats["total"]
        print(f"  {steps} steps: {stats['correct']}/{stats['total']} = {pct:.1f}%")

    with open("finqa_oracle_results.json", "w") as f:
        json.dump({
            "model": MODEL,
            "overall_accuracy": correct_count / n,
            "by_steps": by_steps,
            "results": results,
        }, f, indent=2)
    print("\nFull results saved to finqa_oracle_results.json")


if __name__ == "__main__":
    main()
