"""
Load FinQA dataset, filter for genuinely hard multi-step questions,
and format into our oracle test format.

Setup (run locally):
    git clone https://github.com/czyssrs/FinQA.git
    cd FinQA
    # the dataset json files are in dataset/train.json, dataset/dev.json, dataset/test.json
    # (test.json has no gold answers - use train.json or dev.json)

Usage:
    python load_finqa.py /path/to/FinQA/dataset/dev.json
"""

import json
import sys
import re


def format_evidence(example):
    """Combine pre_text + table + post_text into a single evidence string."""
    pre_text = " ".join(example.get("pre_text", []))
    post_text = " ".join(example.get("post_text", []))
    table = example.get("table", [])

    table_str = ""
    if table:
        rows = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in table]
        table_str = "\n".join(rows)

    parts = []
    if pre_text:
        parts.append(pre_text)
    if table_str:
        parts.append("Table:\n" + table_str)
    if post_text:
        parts.append(post_text)
    return "\n\n".join(parts)


def count_program_steps(program_str):
    """Count number of operations in a FinQA gold program string.
    e.g. 'subtract(206588, 181001), divide(#0, 181001)' -> 2 steps
    """
    if not program_str:
        return 0
    # operations are separated by "), " and end with ")"
    ops = re.findall(r'\b(add|subtract|multiply|divide|exp|greater|table_max|table_min|table_sum|table_average)\(', program_str)
    return len(ops)


def extract_qa(example):
    """FinQA examples have either 'qa' (single question) or 'qa_0'/'qa_1' (multi-turn)."""
    qa_blocks = []
    if "qa" in example:
        qa_blocks.append(example["qa"])
    for key in example:
        if key.startswith("qa_") and key not in ("qa",):
            qa_blocks.append(example[key])
    return qa_blocks


def main():
    if len(sys.argv) < 2:
        print("Usage: python load_finqa.py /path/to/dev.json [max_questions] [min_steps]")
        sys.exit(1)

    filepath = sys.argv[1]
    max_questions = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    min_steps = int(sys.argv[3]) if len(sys.argv) > 3 else 2  # only multi-step (hard) questions

    with open(filepath, "r") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} examples from {filepath}")

    filtered = []
    for example in data:
        evidence = format_evidence(example)
        for qa in extract_qa(example):
            program = qa.get("program", "")
            steps = count_program_steps(program)
            if steps < min_steps:
                continue
            exe_ans = qa.get("exe_ans")
            if exe_ans is None:
                continue
            filtered.append({
                "id": example.get("id", f"finqa_{len(filtered)}"),
                "question": qa.get("question", ""),
                "evidence": evidence,
                "gold_answer_numeric": exe_ans,
                "gold_program": program,
                "num_steps": steps,
            })

    print(f"Found {len(filtered)} multi-step (>={min_steps} ops) questions")

    # Sample across difficulty levels: prioritize a spread of step counts
    filtered.sort(key=lambda x: x["num_steps"])
    step_counts = {}
    for item in filtered:
        step_counts[item["num_steps"]] = step_counts.get(item["num_steps"], 0) + 1
    print(f"Step-count distribution: {step_counts}")

    # Stratified sample: spread across step counts so we can compare accuracy by difficulty
    from collections import defaultdict
    by_steps = defaultdict(list)
    for item in filtered:
        by_steps[item["num_steps"]].append(item)

    sampled = []
    step_keys = sorted(by_steps.keys())
    per_bucket = max(1, max_questions // len(step_keys))
    for steps in step_keys:
        sampled.extend(by_steps[steps][:per_bucket])
    sampled = sampled[:max_questions]

    with open("finqa_oracle_set.json", "w") as f:
        json.dump(sampled, f, indent=2)

    print(f"\nSaved {len(sampled)} questions to finqa_oracle_set.json")
    print("\nSample question:")
    if sampled:
        print(json.dumps(sampled[0], indent=2)[:800])


if __name__ == "__main__":
    main()