"""
FinReflectKG Agentic GraphRAG - Unified Review & Defense Showcase
Interactive CLI Demonstration for Academic Review / Project Evaluation.

Demonstrates:
  1. Adaptive Routing & Latency Reduction
  2. Structured Evidence Formatting & Distractor Resistance
  3. Claim-Level Provenance & SEC 10-K Auditability
"""

import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

root_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "Adaptive_Routing"))
sys.path.insert(0, str(root_dir / "Structured_Formatting"))
sys.path.insert(0, str(root_dir / "Provenance_Scoring"))

load_dotenv(find_dotenv(), override=True)

from Adaptive_Routing.router import QueryRouter
from Structured_Formatting.pipeline_with_formatting import FormattedGraphRAG
from Structured_Formatting.formatter import format_structured_evidence
from Provenance_Scoring.verifier import ClaimVerifier

# ANSI Colors for sleek presentation
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
BLUE = "\033[94m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"

DEMO_QUESTIONS = [
    {
        "title": "1. Descriptive Query (Adaptive Routing Demo)",
        "query": "What is the general business description and principal products of Apple Inc?",
        "focus": "Shows router picking SIMPLE_VECTOR route and saving ~64% latency."
    },
    {
        "title": "2. Multi-Hop Relational Query (Structured Formatting Demo)",
        "query": "What risk factors and market conditions affect Apple's supply chain and revenue?",
        "focus": "Shows Structured Evidence Formatter grouping triples into clean Markdown tables."
    },
    {
        "title": "3. Symbolic Computation Query (Exact Math Demo)",
        "query": "What was Apple's percentage change in net sales from 2021 to 2022?",
        "focus": "Shows router invoking deterministic symbolic math engine to prevent arithmetic hallucination."
    },
    {
        "title": "4. Audit & Provenance Verification Query (Hallucination-Check Demo)",
        "query": "What regulatory and government compliance requirements affect Apple's operations?",
        "focus": "Shows ClaimVerifier auditing sentence-by-sentence against SEC 10-K and KG sources."
    }
]

def print_header(title: str, color: str = CYAN):
    print(f"\n{color}{BOLD}{'=' * 80}")
    print(f" {title.center(78)}")
    print(f"{'=' * 80}{RESET}\n")

class ReviewShowcaseRunner:
    def __init__(self):
        print(f"{CYAN}Initializing FinReflectKG Agentic Engine and Knowledge Graph...{RESET}")
        self.rag = FormattedGraphRAG(enable_structured_formatting=True)
        self.verifier = ClaimVerifier(token_overlap_threshold=0.35)
        print(f"{GREEN}Initialization complete! Engine ready for evaluation.{RESET}\n")

    def run_query(self, query: str, q_idx: int = None):
        print(f"{BOLD}{YELLOW}>>> USER QUERY:{RESET} {BOLD}{query}{RESET}\n")

        # STEP 1: ROUTING
        print(f"{BLUE}[STEP 1: ADAPTIVE ROUTING DECISION]{RESET}")
        route_decision = self.rag.router.route(query)
        r_type = route_decision["route"].value
        print(f"  -> Selected Route: {GREEN}{BOLD}{r_type}{RESET}")
        print(f"  -> Confidence:     {YELLOW}{route_decision['confidence'] * 100}%{RESET}")
        print(f"  -> Rationale:      {CYAN}{route_decision['reasoning']}{RESET}")

        # STEP 2: EXECUTION
        t_start = time.time()
        res = self.rag.answer_query(query)
        elapsed = time.time() - t_start
        exec_info = res.get("execution", {})

        print(f"\n{BLUE}[STEP 2: RETRIEVAL & STRUCTURED EVIDENCE]{RESET}")
        evidence_used = exec_info.get("evidence_context_used", "")
        if r_type == "GRAPH_MULTIHOP" and evidence_used:
            print(f"{MAGENTA}--- Formatted Structured Evidence (Grouped Tables) ---{RESET}")
            lines = evidence_used.split("\n")
            preview = "\n".join(lines[:12])
            print(preview)
            if len(lines) > 12:
                print("... [Full grouped tables passed to Qwen-2.5]")
        elif r_type == "SYMBOLIC_COMPUTE":
            calc = exec_info.get("calculation_summary", "")
            print(f"{MAGENTA}--- Verified Symbolic Computation ---{RESET}\n{calc}")
        else:
            count = exec_info.get("evidence_count", 4)
            print(f"Retrieved {count} high-density SEC 10-K text passages.")

        # STEP 3: ANSWER
        print(f"\n{BLUE}[STEP 3: SYNTHESIS & REASONING (Qwen 3.8-27B / Engine)]{RESET}")
        answer = exec_info.get("answer", "")
        print(f"{GREEN}{answer}{RESET}")
        print(f"  -> Total Pipeline Latency: {YELLOW}{round(elapsed, 3)}s{RESET}")

        # STEP 4: PROVENANCE AUDIT
        print(f"\n{BLUE}[STEP 4: CLAIM-LEVEL PROVENANCE AUDIT & GROUNDING CHECK]{RESET}")
        audit_facts = []
        audit_chunks = []
        
        if r_type == "GRAPH_MULTIHOP" and evidence_used:
            audit_facts = [line for line in evidence_used.split("\n") if line.strip()]
        elif r_type == "SYMBOLIC_COMPUTE":
            calc = exec_info.get("calculation_summary", "")
            if calc:
                audit_facts = [calc]
        elif r_type == "SIMPLE_VECTOR":
            # Retrieve vector chunks for audit
            q_emb = self.rag.embed_model.encode(["Represent this sentence for searching relevant passages: " + query])[0]
            import numpy as np
            sims = np.dot(self.rag.chunk_embeddings, q_emb) / (
                np.linalg.norm(self.rag.chunk_embeddings, axis=1) * np.linalg.norm(q_emb)
            )
            top_idx = np.argsort(sims)[::-1][:4]
            audit_chunks = self.rag.chunks.iloc[top_idx].tolist()

        audit = self.verifier.audit_response(answer, vector_chunks=audit_chunks, graph_facts=audit_facts)

        print(f"  -> Total Claims Extracted:  {audit['total_claims']}")
        print(f"  -> Grounded in SEC/KG:      {GREEN}{audit['grounded_claims']}{RESET}")
        print(f"  -> Ungrounded/Hallucinated: {RED if audit['hallucinated_claims'] > 0 else GREEN}{audit['hallucinated_claims']}{RESET}")
        print(f"  -> Audit Faithfulness:      {GREEN}{BOLD}{audit['faithfulness_score']}%{RESET}")
        
        if audit["claims"]:
            print(f"\n{CYAN}--- Sample Sentence-by-Sentence Audit Verification ---{RESET}")
            for c in audit["claims"][:3]:
                status = f"{GREEN}VERIFIED{RESET}" if c["is_grounded"] else f"{RED}UNVERIFIED{RESET}"
                print(f"  [{status}] Claim: \"{c['claim'][:70]}...\"")
                print(f"            Source: {YELLOW}{c['matched_source_id']}{RESET} (Confidence: {c['confidence']})")

        print(f"\n{GREEN}{'─' * 80}{RESET}")

def main():
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    print_header("FinReflectKG: Agentic GraphRAG Defense Showcase", MAGENTA)
    print("Welcome to the unified evaluation runner for the project defense.\n")

    auto_mode = "--auto" in sys.argv
    runner = ReviewShowcaseRunner()

    if auto_mode:
        for idx, item in enumerate(DEMO_QUESTIONS, 1):
            print_header(f"DEMO CASE {idx}: {item['title']}", CYAN)
            print(f"{YELLOW}Defense Focus:{RESET} {item['focus']}\n")
            runner.run_query(item["query"], idx)
            time.sleep(1)
        print_header("ALL DEMO CASES COMPLETED SUCCESSFULLY", GREEN)
        return

    while True:
        print("\nAvailable Showcase Options:")
        for idx, item in enumerate(DEMO_QUESTIONS, 1):
            print(f"  [{idx}] {item['title']}")
        print("  [5] Enter Custom Query")
        print("  [6] Run All Preset Defense Queries")
        print("  [0] Exit")

        choice = input(f"\n{BOLD}Select an option (0-6): {RESET}").strip()

        if choice == "0":
            print("\nExiting Defense Showcase. Good luck with the review!")
            break
        elif choice in ["1", "2", "3", "4"]:
            q_data = DEMO_QUESTIONS[int(choice) - 1]
            print_header(q_data["title"], CYAN)
            print(f"{YELLOW}Defense Focus:{RESET} {q_data['focus']}\n")
            runner.run_query(q_data["query"])
        elif choice == "5":
            custom_q = input(f"\n{BOLD}Enter your financial question: {RESET}").strip()
            if custom_q:
                runner.run_query(custom_q)
        elif choice == "6":
            for idx, item in enumerate(DEMO_QUESTIONS, 1):
                print_header(f"DEMO CASE {idx}: {item['title']}", CYAN)
                print(f"{YELLOW}Defense Focus:{RESET} {item['focus']}\n")
                runner.run_query(item["query"], idx)
        else:
            print(f"{RED}Invalid selection. Please choose 0-6.{RESET}")

if __name__ == "__main__":
    main()
