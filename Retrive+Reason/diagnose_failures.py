import json
import os
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

def main():
    with open("gold_qa_set.json", "r") as f:
        clean_set = {item["id"]: item for item in json.load(f)}
    with open("raw_triplet_qa_set.json", "r") as f:
        raw_set = {item["id"]: item for item in json.load(f)}

    client = Groq(api_key=GROQ_API_KEY)

    # oracle_6 failed on CLEAN table
    print("="*70)
    print("FULL OUTPUT: oracle_6 / CLEAN TABLE")
    print("="*70)
    item = clean_set["oracle_6"]
    out = call_llm(client, CLEAN_PROMPT, item["evidence"], item["question"])
    print(out)

    # oracle_7 failed on RAW triples
    print("\n\n" + "="*70)
    print("FULL OUTPUT: oracle_7 / RAW TRIPLES")
    print("="*70)
    item = raw_set["oracle_7"]
    evidence_str = "\n".join(item["raw_triples"])
    out = call_llm(client, RAW_PROMPT, evidence_str, item["question"])
    print(out)

if __name__ == "__main__":
    main()
