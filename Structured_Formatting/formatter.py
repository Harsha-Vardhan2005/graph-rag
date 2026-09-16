"""
Structured Evidence Formatter for Financial GraphRAG
Solves the "Retrieval-vs-Reasoning Gap" by transforming flat, unordered graph triples
into semantically grouped, prioritized Markdown tables for LLM prompt context.

Key Features:
  1. Grouping: Categorizes triples by target entity type (e.g., ORG_REG, FIN_MARKET, LITIGATION, RISK_FACTOR).
  2. Query-Aware Prioritization: Promotes the most query-relevant entity group to the top with a [PRIMARY MATCH] anchor.
  3. 100% Fact Preservation: Reorganizes and structures all facts without dropping any information.
  4. Markdown Table Rendering: Formats facts into clear 3-column tables (Source Entity | Relationship | Target Entity).
"""

import sys
import re
from typing import List, Dict, Any, Union

# Ensure UTF-8 output if possible
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


class StructuredEvidenceFormatter:
    def __init__(self):
        # Category groupings for entity types (using ASCII-compatible headers)
        self.category_map = {
            # Regulatory & Legal
            "ORG_REG": ("[REGULATORY] Regulatory & Oversight Bodies", ["regulatory", "regulator", "sec", "doj", "court", "oversight", "supervise"]),
            "REGULATORY_REQUIREMENT": ("[COMPLIANCE] Regulatory Requirements & Rules", ["rule", "requirement", "compliance", "standard", "mandate"]),
            "LITIGATION": ("[LEGAL] Litigation & Legal Proceedings", ["litigation", "lawsuit", "antitrust", "legal", "court", "case", "investigation"]),
            "LEGAL_DOC": ("[LEGAL_DOC] Disclosed Legal Documents", ["document", "filing", "disclosure"]),
            
            # Market Risk & Macro
            "FIN_MARKET": ("[MARKET] Financial Market Conditions", ["market", "condition", "liquidity", "credit", "rate", "interest"]),
            "MACRO_CONDITION": ("[MACRO] Macroeconomic Conditions", ["macro", "economic", "recession", "inflation", "gdp", "currency"]),
            "RISK_FACTOR": ("[RISK] Disclosed Risk Factors", ["risk", "threat", "uncertainty", "pressure", "vulnerability"]),
            
            # Financial Metrics & Accounting
            "FIN_METRIC": ("[METRICS] Financial & Operational Metrics", ["metric", "sale", "revenue", "income", "margin", "growth", "expense"]),
            "ACCOUNTING_POLICY": ("[ACCOUNTING] Accounting Policies & Methods", ["accounting", "policy", "method", "gaap", "deferred"]),
            
            # Corporate Structure & Segments
            "ORG": ("[ORGANIZATION] Companies & Affiliates", ["organization", "company", "partner", "entity"]),
            "SEGMENT": ("[SEGMENT] Business Units & Segments", ["segment", "unit", "division", "cloud", "service"]),
            "PRODUCT": ("[PRODUCT] Products & Platforms", ["product", "service", "hardware", "software", "device", "produce"]),
            "PERSON": ("[PERSONNEL] Key Personnel & Executives", ["executive", "officer", "director", "person", "ceo", "cfo"]),
            "CONCEPT": ("[CONCEPT] Business Concepts & Events", ["concept", "event", "strategy", "action"])
        }

    def _parse_triple(self, raw_fact: Union[str, Dict[str, Any]]) -> Dict[str, str]:
        """
        Parses a raw fact string or dict into standard components:
        [src_type] src --rel--> [tgt_type] tgt
        """
        if isinstance(raw_fact, dict):
            return {
                "src_type": raw_fact.get("src_type", "ENTITY"),
                "src": raw_fact.get("src", "Unknown"),
                "rel": raw_fact.get("rel", "related_to"),
                "tgt_type": raw_fact.get("tgt_type", "ENTITY"),
                "tgt": raw_fact.get("tgt", "Unknown")
            }

        # Regex parsing for string format: [SRC_TYPE] src --rel--> [TGT_TYPE] tgt
        pattern = r"\[(.*?)\]\s*(.*?)\s*--(.*?)-->\s*\[(.*?)\]\s*(.*)"
        match = re.search(pattern, str(raw_fact))
        if match:
            return {
                "src_type": match.group(1).strip(),
                "src": match.group(2).strip(),
                "rel": match.group(3).strip(),
                "tgt_type": match.group(4).strip(),
                "tgt": match.group(5).strip()
            }
        
        # Fallback for non-standard string formats
        return {
            "src_type": "ENTITY",
            "src": "Entity",
            "rel": "relates_to",
            "tgt_type": "ENTITY",
            "tgt": str(raw_fact)
        }

    def _determine_priority_score(self, entity_type: str, query: str) -> int:
        """
        Calculates relevance score between entity type and user query.
        """
        q_lower = query.lower()
        cat_info = self.category_map.get(entity_type)
        if not cat_info:
            return 0

        keywords = cat_info[1]
        score = sum(2 for kw in keywords if kw in q_lower)
        
        # Direct entity type name mention boost
        if entity_type.lower() in q_lower:
            score += 5

        return score

    def format_structured_evidence(self, graph_facts: List[Union[str, Dict[str, Any]]], query: str = "") -> str:
        """
        Main formatter: Reorganizes facts into categorized, query-prioritized Markdown tables.
        Guarantees 0% dropped facts.
        """
        if not graph_facts:
            return "No structured knowledge graph facts available for this query."

        # 1. Parse all facts
        parsed_facts = [self._parse_triple(f) for f in graph_facts]

        # 2. Group facts by dominant entity type (either src_type or tgt_type)
        grouped: Dict[str, List[Dict[str, str]]] = {}
        for item in parsed_facts:
            # If target is a generic ticker/ORG (e.g. aapl), group by src_type, else tgt_type
            if item["tgt"].lower() in ["aapl", "msft", "apple", "microsoft"] and item["src_type"] != "ORG":
                dominant_type = item["src_type"]
            else:
                dominant_type = item["tgt_type"]

            if dominant_type not in grouped:
                grouped[dominant_type] = []
            grouped[dominant_type].append(item)

        # 3. Score and sort groups by query relevance
        group_scores = []
        for d_type, items in grouped.items():
            score = self._determine_priority_score(d_type, query) if query else 0
            group_scores.append((d_type, items, score))

        # Sort: Highest score first, then by count descending
        group_scores.sort(key=lambda x: (x[2], len(x[1])), reverse=True)

        # 4. Render Markdown Tables
        markdown_sections = []
        total_rendered_facts = 0

        for idx, (d_type, items, score) in enumerate(group_scores):
            cat_header, _ = self.category_map.get(d_type, (f"[{d_type}] Specific Entities", []))
            
            # Badge assignment
            if idx == 0 and score > 0:
                badge = " [PRIMARY QUERY MATCH]"
            elif score > 0:
                badge = " [RELEVANT CONTEXT]"
            else:
                badge = " [SUPPORTING GRAPH FACTS]"

            header_line = f"### {cat_header} ({len(items)} facts){badge}"
            
            table_lines = [
                header_line,
                "| Source Entity | Relationship | Target Entity (Type) |",
                "| :--- | :--- | :--- |"
            ]

            for item in items:
                src_str = f"{item['src']} `[{item['src_type']}]`" if item['src_type'] != "ENTITY" else item['src']
                tgt_str = f"**{item['tgt']}** `[{item['tgt_type']}]`"
                rel_str = f"`{item['rel']}`"
                table_lines.append(f"| {src_str} | {rel_str} | {tgt_str} |")
                total_rendered_facts += 1

            markdown_sections.append("\n".join(table_lines))

        # Integrity Check: Ensure 0 dropped facts
        assert total_rendered_facts == len(graph_facts), f"Formatting dropped facts: {total_rendered_facts} != {len(graph_facts)}"

        return "\n\n".join(markdown_sections)


# Singleton instance for direct imports
formatter = StructuredEvidenceFormatter()
format_structured_evidence = formatter.format_structured_evidence


if __name__ == "__main__":
    sample_facts = [
        "[COMP] channel partner --negatively_impacts--> [ORG] aapl",
        "[CONCEPT] price reduction --negatively_impacts--> [ORG] aapl",
        "[FIN_MARKET] liquidity --negatively_impacts--> [ORG] aapl",
        "[ORG_REG] u.s. department of justice --regulates--> [ORG] aapl",
        "[ORG_REG] sec --regulates--> [ORG] aapl",
        "[LITIGATION] apple ebooks antitrust litigation --subject_to--> [ORG] aapl"
    ]

    test_query = "Which regulatory bodies are named in connection with Apple, and what litigation prompted this?"
    
    print("=" * 80)
    print("DEMO: STRUCTURED EVIDENCE FORMATTER")
    print(f"Query: {test_query}")
    print("=" * 80)
    
    formatted_output = format_structured_evidence(sample_facts, test_query)
    print(formatted_output)
