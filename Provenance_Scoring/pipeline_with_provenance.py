"""
Pipeline with Claim-Level Provenance & Auditability
Runs Agentic GraphRAG with Structured Evidence Formatting and attaches provenance citations.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

# Ensure roots in python path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "Structured_Formatting"))
sys.path.insert(0, str(root_dir / "Adaptive_Routing"))

load_dotenv(find_dotenv(), override=True)

from Structured_Formatting.pipeline_with_formatting import FormattedGraphRAG
from Provenance_Scoring.verifier import ClaimVerifier

class ProvenanceAuditedPipeline:
    def __init__(self, enable_structured_formatting: bool = True):
        self.rag = FormattedGraphRAG(enable_structured_formatting=enable_structured_formatting)
        self.verifier = ClaimVerifier(token_overlap_threshold=0.35)

    def query_with_audit(self, query: str) -> dict:
        """
        Executes query through pipeline and runs claim-level provenance audit.
        """
        exec_res = self.rag.answer_query(query)
        execution = exec_res.get("execution", {})
        raw_answer = execution.get("answer", "")
        route_taken = execution.get("route_taken", "UNKNOWN")

        # Collect evidence sources
        chunks = []
        graph_facts = []
        if route_taken == "SIMPLE_VECTOR":
            chunks = getattr(self.rag, "last_retrieved_chunks", [])
            if not chunks:
                # Retrieve top passages for auditing context
                q_emb = self.rag.embed_model.encode(["Represent this sentence for searching relevant passages: " + query])[0]
                import numpy as np
                sims = np.dot(self.rag.chunk_embeddings, q_emb) / (
                    np.linalg.norm(self.rag.chunk_embeddings, axis=1) * np.linalg.norm(q_emb)
                )
                top_idx = np.argsort(sims)[::-1][:4]
                chunks = self.rag.chunks.iloc[top_idx].tolist()
        elif route_taken == "GRAPH_MULTIHOP":
            evidence_used = execution.get("evidence_context_used", "")
            graph_facts = [line for line in evidence_used.split("\n") if line.strip()]
        elif route_taken == "SYMBOLIC_COMPUTE":
            calc_summary = execution.get("calculation_summary", "")
            graph_facts = [calc_summary] if calc_summary else []

        # Run provenance audit
        audit = self.verifier.audit_response(
            response_text=raw_answer,
            vector_chunks=chunks,
            graph_facts=graph_facts
        )

        # Construct verified answer with inline citations
        cited_lines = []
        for claim_info in audit["claims"]:
            c_text = claim_info["claim"]
            src_id = claim_info["matched_source_id"]
            if claim_info["is_grounded"]:
                cited_lines.append(f"{c_text} [[{src_id}]]")
            else:
                cited_lines.append(f"{c_text} [⚠️ UNVERIFIED]")

        audited_answer = "\n\n".join(cited_lines) if cited_lines else raw_answer

        return {
            "query": query,
            "route_taken": route_taken,
            "raw_answer": raw_answer,
            "audited_answer": audited_answer,
            "faithfulness_score": audit["faithfulness_score"],
            "hallucination_rate": audit["hallucination_rate"],
            "total_claims": audit["total_claims"],
            "grounded_claims": audit["grounded_claims"],
            "hallucinated_claims": audit["hallucinated_claims"],
            "audit_details": audit["claims"],
            "latency_sec": exec_res.get("total_latency_sec", 0.0),
            "evidence_sources_count": len(chunks) + len(graph_facts)
        }

if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    
    print("Testing ProvenanceAuditedPipeline...")
    pipeline = ProvenanceAuditedPipeline(enable_structured_formatting=True)
    res = pipeline.query_with_audit("What risk factors and market conditions affect Apple's supply chain and revenue?")
    print(f"\nRoute: {res['route_taken']}")
    print(f"Faithfulness Score: {res['faithfulness_score']}%")
    print(f"Total Claims: {res['total_claims']} (Grounded: {res['grounded_claims']}, Hallucinated: {res['hallucinated_claims']})")
    print(f"\nAudited Answer Preview:\n{res['audited_answer'][:400]}...")
