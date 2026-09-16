import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
from groq import Groq

load_dotenv(override=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
client = Groq(api_key=GROQ_API_KEY)

USERNAME = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER") or os.getenv("USERNAME", "24708389")
PASSWORD = os.getenv("NEO4J_PASSWORD") or os.getenv("PASSWORD", "")
URI = os.getenv("NEO4J_URI") or os.getenv("URI") or f"neo4j+ssc://{USERNAME}.databases.neo4j.io"

driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))

TOP_K_VECTOR = 5

QUESTIONS = [
    {
        "text": "Which regulatory bodies are named in connection with Apple's disclosed net income, and what litigation prompted this?",
        "ticker": "AAPL",
        "relation_types": ["regulates", "subject_to"],
    },
    {
        "text": "What financial market conditions does Apple disclose as negatively impacting its financial metrics?",
        "ticker": "AAPL",
        "relation_types": ["negatively_impacts", "impacted_by"],
    },
]

def graph_lookup(tx, ticker, relation_types):
    query = """
    MATCH (a:Entity)-[r:RELATION]->(b:Entity {ticker:$ticker})
    WHERE r.type IN $relation_types
    RETURN a.name AS source, a.entity_type AS source_type,
           r.type AS relation, r.chunk_text AS chunk_text
    ORDER BY a.entity_type, a.name
    LIMIT 10
    """
    return list(tx.run(query, ticker=ticker, relation_types=relation_types))

def vector_search(model, chunk_embeddings, chunks, question_text, top_k):
    q_embedding = model.encode(["Represent this sentence for searching relevant passages: " + question_text])[0]
    sims = np.dot(chunk_embeddings, q_embedding) / (
        np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(q_embedding)
    )
    top_idx = np.argsort(sims)[::-1][:top_k]
    return chunks.iloc[top_idx].tolist()

def main():
    print("Loading CSV and embeddings...")
    import json
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    emb_file = os.path.join(parent_dir, "aapl_embeddings.npy") if os.path.exists(os.path.join(parent_dir, "aapl_embeddings.npy")) else "aapl_embeddings.npy"
    chk_file = os.path.join(parent_dir, "aapl_chunks.json") if os.path.exists(os.path.join(parent_dir, "aapl_chunks.json")) else "aapl_chunks.json"
    csv_file = os.path.join(parent_dir, "finreflectkg_aapl_msft.csv") if os.path.exists(os.path.join(parent_dir, "finreflectkg_aapl_msft.csv")) else "finreflectkg_aapl_msft.csv"

    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    if os.path.exists(emb_file) and os.path.exists(chk_file):
        with open(chk_file, "r") as f:
            chunks = pd.Series(json.load(f))
        chunk_embeddings = np.load(emb_file)
    else:
        df = pd.read_csv(csv_file)
        df_aapl = df[df["ticker"] == "AAPL"]
        chunks = df_aapl["chunk_text"].dropna().drop_duplicates().reset_index(drop=True)
        chunk_embeddings = model.encode(chunks.tolist(), show_progress_bar=True, batch_size=64)

    with driver.session() as session:
        for q in QUESTIONS:
            print(f"\n{'='*70}")
            print(f"Q: {q['text']}")

            vec_chunks = vector_search(model, chunk_embeddings, chunks, q["text"], TOP_K_VECTOR)

            graph_results = session.execute_read(graph_lookup, q["ticker"], q["relation_types"])

            print(f"\nGraph facts found ({len(graph_results)}):")
            graph_context_parts = []
            for g in graph_results:
                fact_line = f"[{g['source_type']}] {g['source']} --{g['relation']}--> {q['ticker']}"
                print(f"  {fact_line}")
                if g["chunk_text"]:
                    graph_context_parts.append(f"{fact_line}\nEvidence: {g['chunk_text'][:400]}")

            combined_context = "GRAPH FACTS:\n" + "\n\n".join(graph_context_parts[:5])
            combined_context += "\n\nRETRIEVED PASSAGES:\n" + "\n\n".join(c[:500] for c in vec_chunks[:5])

            prompt = f"""Answer the question using ONLY the context below. The context has two parts:
GRAPH FACTS (structured relationships extracted from filings) and RETRIEVED PASSAGES (raw text).
Prefer graph facts for precision, use passages for supporting detail.

Context:
{combined_context}

Question: {q['text']}"""

            response = client.chat.completions.create(
                model=MODEL_NAME,
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}]
            )
            print(f"\nTier 2 Answer (graph-enriched RAG):")
            print(response.choices[0].message.content)

if __name__ == "__main__":
    main()
    driver.close()