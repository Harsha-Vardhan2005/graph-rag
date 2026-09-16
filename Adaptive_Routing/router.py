"""
Hybrid Adaptive Query Router with Graph Connectivity Signals & PPR Ranking
Combines:
  1. Financial Entity Extraction (FinReflectKG schema grounded)
  2. Graph Connectivity Confidence Scorer (Topology-based routing signals)
  3. Personalized PageRank (PPR) Subgraph Ranking (NetworkX random-walk ranking)
  4. Fast-Path Regex & LLM Intent Classifier Fallback
"""

import re
import os
import sys
from pathlib import Path
from enum import Enum
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv, find_dotenv
from groq import Groq

# Ensure root directory in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Subsystem modules
from Adaptive_Routing.entity_extractor import FinancialEntityExtractor
from Adaptive_Routing.graph_connectivity import GraphConnectivityScorer
from Adaptive_Routing.ppr_engine import LocalPPRRanker

load_dotenv(find_dotenv(), override=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")


class RouteType(str, Enum):
    SIMPLE_VECTOR = "SIMPLE_VECTOR"
    GRAPH_MULTIHOP = "GRAPH_MULTIHOP"
    SYMBOLIC_COMPUTE = "SYMBOLIC_COMPUTE"


class QueryRouter:
    def __init__(self, groq_api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.api_key = groq_api_key or GROQ_API_KEY
        self.model = model_name or MODEL_NAME
        self.client = Groq(api_key=self.api_key) if self.api_key else None

        # Initialize topology and extraction engines
        self.extractor = FinancialEntityExtractor()
        self.connectivity_scorer = GraphConnectivityScorer()
        self.ppr_ranker = LocalPPRRanker()

        # Relation types present in FinReflectKG
        self.graph_keywords = {
            "regulates", "regulatory", "regulator", "oversee", "overseeing", "sec", "doj", "court", "antitrust",
            "litigation", "lawsuit", "subject_to", "legal", "investigation",
            "negatively_impacts", "impacted_by", "positively_impacts", "market condition", "market conditions",
            "liquidity", "credit market", "risk factor", "risk factors", "risk", "risks", "macro", "macroeconomic",
            "has_stake_in", "stake", "subsidiary", "segment", "partner", "ownership",
            "depends_on", "produce", "discloses", "disclose", "disclosed", "disclosure", "disclosures",
            "metric", "metrics", "financial metric", "financial metrics"
        }

        # Calculation & temporal keywords
        self.symbolic_keywords = {
            "yoy", "growth", "percentage", "percent", "increase", "decrease", "change",
            "growth rate", "margin", "ratio", "difference", "compare", "between",
            "from 20", "to 20", "in 20"
        }

        # Financial metric patterns
        self.metric_names = [
            "net sales", "revenue", "net income", "operating income", "gross margin",
            "total assets", "cash flow", "earnings per share", "eps", "operating expenses",
            "research and development", "r&d", "cost of sales"
        ]

    def _fast_path_classify(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Stage 1: Ultra-fast (0ms) regex and keyword matching.
        """
        q_lower = query.lower()

        # Check for multi-year pattern e.g., '2021 and 2022', '2020 to 2023', '2019, 2020'
        years = re.findall(r"\b(201[4-9]|202[0-4])\b", q_lower)
        unique_years = sorted(list(set(int(y) for y in years)))

        # 1. Check for Symbolic / YoY computation
        has_calc_word = any(w in q_lower for w in self.symbolic_keywords)
        matched_metrics = [m for m in self.metric_names if m in q_lower]

        if len(unique_years) >= 2 and (has_calc_word or matched_metrics):
            metric_target = matched_metrics[0] if matched_metrics else "revenue"
            ticker = "MSFT" if "microsoft" in q_lower or "msft" in q_lower else "AAPL"
            return {
                "route": RouteType.SYMBOLIC_COMPUTE,
                "confidence": 0.95,
                "reasoning": "Detected multi-year temporal window and metric calculation.",
                "metadata": {
                    "ticker": ticker,
                    "metric_name": metric_target,
                    "year_start": unique_years[0],
                    "year_end": unique_years[-1]
                }
            }

        # 2. Check for Structured Graph Relations
        matched_graph_keys = [k for k in self.graph_keywords if k in q_lower]
        if len(matched_graph_keys) >= 2 or any(k in q_lower for k in [
            "regulatory bodies", "litigation", "market conditions", "stake in", "regulates",
            "financial metrics", "disclose", "discloses", "disclosures", "oversee"
        ]):
            ticker = "MSFT" if "microsoft" in q_lower or "msft" in q_lower else "AAPL"
            
            # Map query keywords to exact graph relation types
            rel_types = []
            if any(k in q_lower for k in ["regulat", "oversee", "sec", "doj"]):
                rel_types.append("regulates")
            if any(k in q_lower for k in ["litigat", "lawsuit", "subject_to", "legal"]):
                rel_types.append("subject_to")
            if any(k in q_lower for k in ["negatively", "impact", "risk", "condition"]):
                rel_types.extend(["negatively_impacts", "impacted_by"])
            if any(k in q_lower for k in ["stake", "ownership", "segment"]):
                rel_types.extend(["has_stake_in", "discloses"])
            if any(k in q_lower for k in ["disclose", "metric"]):
                rel_types.extend(["discloses", "has_stake_in"])
            if any(k in q_lower for k in ["produce", "product"]):
                rel_types.extend(["produce", "positively_impacts"])

            final_rels = list(set(rel_types)) or ["discloses"]

            # Compute topological connectivity confidence signal
            conn_info = self.connectivity_scorer.evaluate_connectivity(ticker, final_rels)

            return {
                "route": RouteType.GRAPH_MULTIHOP,
                "confidence": conn_info["graph_confidence"],
                "reasoning": f"Detected structured relation intent: {matched_graph_keys[:3]} (Graph Density Signal: {conn_info['matched_edge_count']} matching edges).",
                "metadata": {
                    "ticker": ticker,
                    "relation_types": final_rels,
                    "graph_connectivity": conn_info
                }
            }

        return None

    def _llm_classify(self, query: str) -> Dict[str, Any]:
        """
        Stage 2: LLM classifier fallback for ambiguous queries.
        """
        if not self.client:
            return {
                "route": RouteType.SIMPLE_VECTOR,
                "confidence": 0.70,
                "reasoning": "Fallback default to vector retrieval.",
                "metadata": {}
            }

        prompt = f"""You are a query routing controller for a Financial Knowledge Graph RAG system.
Classify the following query into exactly ONE route:

1. SIMPLE_VECTOR: Open-ended descriptions, business philosophy, narrative text, general business overviews.
2. GRAPH_MULTIHOP: Questions about specific named relationships, regulatory bodies, disclosures, litigations, market conditions, suppliers, or multi-hop entity connections.
3. SYMBOLIC_COMPUTE: Questions requiring numeric arithmetic, YoY growth, percentage change across multiple fiscal years.

Query: "{query}"

Output ONLY valid JSON in this exact structure:
{{
  "route": "SIMPLE_VECTOR" | "GRAPH_MULTIHOP" | "SYMBOLIC_COMPUTE",
  "confidence": 0.0 to 1.0,
  "reasoning": "Brief explanation of decision.",
  "metadata": {{
    "ticker": "AAPL" | "MSFT",
    "relation_types": ["discloses" | "regulates" | "subject_to" | "negatively_impacts" | "impacted_by" | "has_stake_in"],
    "metric_name": "string (optional)",
    "year_start": 2021 (optional),
    "year_end": 2022 (optional)
  }}
}}"""

        try:
            import json
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a specialized router for financial queries. Respond ONLY with JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=200,
                response_format={"type": "json_object"}
            )
            parsed = json.loads(resp.choices[0].message.content.strip())
            route_str = parsed.get("route", "SIMPLE_VECTOR")
            try:
                route_type = RouteType(route_str)
            except ValueError:
                route_type = RouteType.SIMPLE_VECTOR

            return {
                "route": route_type,
                "confidence": float(parsed.get("confidence", 0.85)),
                "reasoning": parsed.get("reasoning", "LLM classified intent."),
                "metadata": parsed.get("metadata", {})
            }
        except Exception as e:
            return {
                "route": RouteType.SIMPLE_VECTOR,
                "confidence": 0.60,
                "reasoning": f"LLM classification error ({str(e)}), default to vector fast path.",
                "metadata": {}
            }

    def route(self, query: str) -> Dict[str, Any]:
        """
        Main routing method:
          1. Extracts FinReflectKG entities
          2. Applies fast rule path with Graph Connectivity Signals
          3. Falls back to LLM classifier if ambiguous
          4. Computes PPR subgraph rankings if GRAPH_MULTIHOP is selected
        """
        # Step 1: Entity Extraction
        entity_info = self.extractor.extract_entities(query)

        # Step 2: Fast-Path Rule Evaluation
        fast_result = self._fast_path_classify(query)
        if fast_result is not None:
            decision = fast_result
        else:
            # Step 3: LLM Fallback
            decision = self._llm_classify(query)

        # Attach extracted entity metadata
        decision["metadata"]["extracted_entities"] = entity_info["entities"]
        decision["metadata"]["entity_types"] = entity_info["entity_types"]

        # Step 4: If GRAPH_MULTIHOP, compute PPR Triples Ranking
        if decision["route"] == RouteType.GRAPH_MULTIHOP:
            ticker = decision["metadata"].get("ticker", entity_info["ticker"])
            rel_types = decision["metadata"].get("relation_types", ["discloses"])
            seed_names = [e["name"] for e in entity_info["entities"]] or [ticker.lower()]
            
            ranked_triples = self.ppr_ranker.rank_triples_ppr(
                seed_entities=seed_names,
                ticker=ticker,
                relation_types=rel_types,
                top_k=12
            )
            decision["metadata"]["ppr_ranked_triples"] = ranked_triples

        return decision


if __name__ == "__main__":
    router = QueryRouter()
    test_queries = [
        "What is the general business description and principal products of Apple Inc?",
        "What risk factors and market conditions affect Apple's supply chain and revenue?",
        "What was Apple's percentage change in net sales from 2021 to 2022?",
        "Which key financial metrics does Microsoft disclose in its SEC reports?"
    ]

    print("=" * 70)
    print("Testing Upgraded Router with Entity Extractor, Graph Connectivity & PPR")
    print("=" * 70)
    for q in test_queries:
        res = router.route(q)
        print(f"\nQuery: {q}")
        print(f"  Route: {res['route'].value} (Confidence: {res['confidence']})")
        print(f"  Entities: {[e['name'] for e in res['metadata'].get('extracted_entities', [])]}")
        print(f"  Reasoning: {res['reasoning']}")
        ppr_sample = res['metadata'].get('ppr_ranked_triples', [])
        if ppr_sample:
            print(f"  Top PPR Triplet: {ppr_sample[0]['triple_str']} (Score: {ppr_sample[0]['ppr_score']})")
