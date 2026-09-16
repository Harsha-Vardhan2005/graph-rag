import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from groq import Groq

load_dotenv(override=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
client = Groq(api_key=GROQ_API_KEY)

QUESTIONS = [
    "Which regulatory bodies are named in connection with Apple's disclosed net income, and what litigation prompted this?",
    "What financial market conditions does Apple disclose as negatively impacting its financial metrics?",
]

TOP_K = 8

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

    for i, question in enumerate(QUESTIONS, 1):
        print(f"\n{'='*70}")
        print(f"Q{i}: {question}")

        q_embedding = model.encode(["Represent this sentence for searching relevant passages: " + question])[0]

        # cosine similarity
        sims = np.dot(chunk_embeddings, q_embedding) / (
            np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(q_embedding)
        )
        top_idx = np.argsort(sims)[::-1][:TOP_K]
        retrieved = chunks.iloc[top_idx].tolist()

        print(f"\nTop {TOP_K} retrieved chunks (similarity scores: {sims[top_idx].round(3)}):")
        for j, chunk in enumerate(retrieved, 1):
            print(f"  [{j}] {chunk[:150]}...")

        retrieved = [c[:600] for c in retrieved] 
        context = "\n\n".join(retrieved)
        prompt = f"""Answer the question using ONLY the context below. If the context doesn't contain the answer, say so.

Context:
{context}

Question: {question}"""

        response = client.chat.completions.create(
            model=MODEL_NAME,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        print(f"\nTier 1 Answer (vector RAG):")
        print(response.choices[0].message.content)

if __name__ == "__main__":
    main()