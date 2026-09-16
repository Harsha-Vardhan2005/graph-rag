"""
Hybrid Adaptive Query Router for Financial GraphRAG
Classifies incoming financial queries into 3 specialized execution routes:
  1. SIMPLE_VECTOR     -> Semantic Passage Retrieval (for descriptive, general filing text)
  2. GRAPH_MULTIHOP    -> Structured Cypher Graph Traversal (for entity relationships, regulations, risk factors)
  3. SYMBOLIC_COMPUTE  -> Graph Metric Extraction + Symbolic Arithmetic (for multi-year YoY growth, percentage change)

Architecture:
  - Stage 1: Fast Rule & Regex Intent Filter (0ms latency)
  - Stage 2: Entity & Metric Grammar Extractor
  - Stage 3: LLM Intent Classifier Fallback (for nuanced natural language queries)
"""

import re
import os
from enum import Enum
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv, find_dotenv
from groq import Groq

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

            return {
                "route": RouteType.GRAPH_MULTIHOP,
                "confidence": 0.90,
                "reasoning": f"Detected structured relationship intent: {matched_graph_keys[:3]}.",
                "metadata": {
                    "ticker": ticker,
                    "relation_types": list(set(rel_types)) or ["discloses"]
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
  "reasoning": "brief explanation",
  "ticker": "AAPL" | "MSFT"
}}"""

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=150
            )
            raw = resp.choices[0].message.content.strip()
            # Clean possible markdown wrapping
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0].strip()
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0].strip()

            parsed = eval(raw) if raw.startswith("{") else {}
            route_str = parsed.get("route", "SIMPLE_VECTOR")
            route_enum = RouteType(route_str) if route_str in RouteType.__members__ else RouteType.SIMPLE_VECTOR

            return {
                "route": route_enum,
                "confidence": 0.85,
                "reasoning": parsed.get("reasoning", "LLM classified intent."),
                "metadata": {"ticker": parsed.get("ticker", "AAPL")}
            }
        except Exception as e:
            return {
                "route": RouteType.SIMPLE_VECTOR,
                "confidence": 0.60,
                "reasoning": f"LLM parsing fallback: {e}",
                "metadata": {}
            }

    def route(self, query: str) -> Dict[str, Any]:
        """
        Main entry point for routing a query.
        """
        # Fast path first (0ms)
        fast_result = self._fast_path_classify(query)
        if fast_result:
            return fast_result

        # LLM fallback
        return self._llm_classify(query)


if __name__ == "__main__":
    router = QueryRouter()
    test_queries = [
        "What are Apple's main accounting policies for revenue recognition?",
        "Which regulatory bodies are named in connection with Apple's disclosed net income?",
        "What was Apple's Net sales in 2021 and 2022, and what was the YoY percentage growth?",
        "Which entities or segments does Microsoft hold a stake in?",
        "What financial market conditions does Apple disclose as negatively impacting its financial metrics?",
    ]

    print("=" * 70)
    print("HYBRID ADAPTIVE QUERY ROUTER TEST")
    print("=" * 70)
    for q in test_queries:
        res = router.route(q)
        print(f"\nQuery: {q}")
        print(f" -> Route:      {res['route'].value}")
        print(f" -> Confidence: {res['confidence']}")
        print(f" -> Reason:     {res['reasoning']}")
        print(f" -> Metadata:   {res.get('metadata', {})}")
