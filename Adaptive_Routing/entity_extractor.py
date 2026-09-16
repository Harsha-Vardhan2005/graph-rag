"""
Financial Entity Extractor for FinReflectKG
Maps query text to exact FinReflectKG schema entity types:
  - FIN_METRIC, ORG_REG, RISK_FACTOR, MACRO_CONDITION, FIN_MARKET, LITIGATION, ACCOUNTING_POLICY, ORG, GPE
"""

import os
import re
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Set

root_dir = Path(__file__).resolve().parent.parent

class FinancialEntityExtractor:
    def __init__(self, csv_path: str = None):
        if csv_path is None:
            csv_path = os.path.join(root_dir, "finreflectkg_aapl_msft.csv")
        
        self.entities_by_type: Dict[str, Set[str]] = {
            "ORG": {"apple", "aapl", "microsoft", "msft", "ernst & young llp"},
            "FIN_METRIC": {
                "net sales", "net income", "operate income", "operating income", "revenue",
                "gross margin", "select financial data", "financial statement and supplementary data",
                "executive compensation", "total assets", "cash flow", "earnings per share", "eps",
                "research and development", "cost of sales"
            },
            "ORG_REG": {
                "sec", "securities and exchange commission", "u.s. department of justice", "doj",
                "u.s. district court for the southern district of new york",
                "public company accounting oversight board", "european commission", "internal revenue service", "irs"
            },
            "RISK_FACTOR": {
                "single or limited source for component", "industry-wide shortage of component",
                "initial capacity constraint for new technology", "supply delay or constraint",
                "aggressive pricing practice", "frequent product introduction", "rapid technological advance",
                "industry-wide downward pressure on gross margin", "significant competition in digital content service",
                "free peer-to-peer music and video service", "substantial resource of competitor",
                "significant pricing fluctuation", "foreign exchange", "supply chain"
            },
            "FIN_MARKET": {"liquidity", "credit market", "market conditions", "interest rate"},
            "MACRO_CONDITION": {"macroeconomic condition", "global economic turmoil", "inflation"},
            "LITIGATION": {"apple ebooks antitrust litigation", "antitrust litigation", "litigation proceeding", "patent lawsuit"},
            "ACCOUNTING_POLICY": {"stock-based compensation", "revenue recognition", "lease accounting"},
            "GPE": {"ireland", "u.s.", "united states", "china", "europe"}
        }

        # If CSV is accessible, dynamically expand known entity vocabulary
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path, usecols=["entity", "entity_type", "target", "target_type"])
                for _, r in df.iterrows():
                    src_type = str(r["entity_type"]).strip()
                    src_name = str(r["entity"]).lower().strip()
                    tgt_type = str(r["target_type"]).strip()
                    tgt_name = str(r["target"]).lower().strip()

                    if src_type in self.entities_by_type and len(src_name) > 2:
                        self.entities_by_type[src_type].add(src_name)
                    if tgt_type in self.entities_by_type and len(tgt_name) > 2:
                        self.entities_by_type[tgt_type].add(tgt_name)
            except Exception:
                pass

    def extract_entities(self, query: str) -> Dict[str, Any]:
        """
        Extracts entities from query and maps them to exact FinReflectKG types.
        """
        q_low = query.lower()
        extracted: List[Dict[str, str]] = []
        matched_types: Set[str] = set()

        # Identify primary company ticker
        ticker = "AAPL"
        if "microsoft" in q_low or "msft" in q_low:
            ticker = "MSFT"
        elif "apple" in q_low or "aapl" in q_low:
            ticker = "AAPL"

        # Match against known entity type sets
        for etype, enames in self.entities_by_type.items():
            for name in enames:
                # Use word-boundary regex for clean matching
                if re.search(r'\b' + re.escape(name) + r'\b', q_low):
                    extracted.append({
                        "name": name,
                        "type": etype,
                        "matched_text": name
                    })
                    matched_types.add(etype)

        # Fallback check for key financial terms if no direct dictionary hit
        if "net sales" in q_low and not any(e["name"] == "net sales" for e in extracted):
            extracted.append({"name": "net sales", "type": "FIN_METRIC", "matched_text": "net sales"})
            matched_types.add("FIN_METRIC")
        if "risk" in q_low and not any(e["type"] == "RISK_FACTOR" for e in extracted):
            matched_types.add("RISK_FACTOR")
        if "regulatory" in q_low or "oversee" in q_low or "regulat" in q_low:
            matched_types.add("ORG_REG")

        # Deduplicate matches
        unique_extracted = []
        seen = set()
        for e in extracted:
            key = (e["name"], e["type"])
            if key not in seen:
                seen.add(key)
                unique_extracted.append(e)

        return {
            "ticker": ticker,
            "entities": unique_extracted,
            "entity_types": list(matched_types),
            "seed_node_count": len(unique_extracted)
        }

if __name__ == "__main__":
    extractor = FinancialEntityExtractor()
    test_q = "Which regulatory bodies oversee Apple regarding its financial disclosures and net income?"
    print(extractor.extract_entities(test_q))
