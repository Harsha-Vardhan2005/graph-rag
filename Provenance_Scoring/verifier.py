"""
Claim-Level Provenance & Auditability Scorer
Decomposes answers into atomic claims and cross-references them against SEC 10-K chunks and FinReflectKG triples.
"""

import re
import string
from typing import List, Dict, Any

class ClaimVerifier:
    def __init__(self, token_overlap_threshold: float = 0.40):
        self.threshold = token_overlap_threshold
        # Common financial/generic stop words to ignore when measuring overlap
        self.stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "and", "or", "in", "on", 
            "at", "by", "for", "with", "about", "against", "between", "into", "through",
            "during", "before", "after", "above", "below", "to", "from", "up", "down",
            "in", "out", "on", "off", "over", "under", "again", "further", "then", "once",
            "here", "there", "when", "where", "why", "how", "all", "any", "both", "each",
            "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only",
            "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just",
            "don", "should", "now", "also", "including", "based", "according", "report",
            "filing", "stated", "states", "shows", "apple", "microsoft", "company"
        }

    def _tokenize(self, text: str) -> set:
        """Tokenize text into cleaned keywords."""
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        tokens = set(text.split())
        return {t for t in tokens if t not in self.stop_words and len(t) > 2}

    def extract_claims(self, text: str) -> List[str]:
        """
        Splits an LLM generated answer into discrete, verifiable factual claims.
        Filters out formatting headers, empty lines, and conversational filler.
        """
        raw_lines = text.split("\n")
        claims = []
        
        for line in raw_lines:
            line = line.strip()
            if not line:
                continue
            # Remove markdown headers and bullets
            line = re.sub(r'^(#+|\*|-|\d+\.)\s*', '', line)
            # Remove markdown bold/italics
            line = re.sub(r'[*_`]', '', line)
            
            # If line has multiple sentences, split by period
            sentences = re.split(r'(?<=[.!?])\s+', line)
            for s in sentences:
                s = s.strip()
                # Must have meaningful length and contain factual content
                if len(s.split()) >= 4 and not s.lower().startswith("based on") and not s.lower().startswith("in summary"):
                    claims.append(s)
                    
        return claims

    def verify_claim(self, claim: str, evidence_sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Verifies an individual claim against a list of evidence sources.
        """
        claim_tokens = self._tokenize(claim)
        if not claim_tokens:
            return {
                "claim": claim,
                "is_grounded": True,
                "confidence": 1.0,
                "matched_source_id": "General/Structural",
                "source_type": "none",
                "matched_snippet": ""
            }

        best_score = 0.0
        best_source = None

        for src in evidence_sources:
            src_text = src.get("content", "")
            src_tokens = self._tokenize(src_text)
            if not src_tokens and not src_text:
                continue
            
            # 1. Direct token set overlap
            intersection = claim_tokens.intersection(src_tokens)
            overlap_ratio = len(intersection) / len(claim_tokens) if claim_tokens else 1.0

            # 2. Substring matching bonus for named entities (e.g., SEC, DOJ, Ireland)
            clean_claim_low = claim.lower()
            clean_src_low = src_text.lower()
            for t in claim_tokens:
                if t in clean_src_low:
                    overlap_ratio = min(1.0, overlap_ratio + 0.15)

            # 3. Financial number alignment bonus
            claim_nums = set(re.findall(r'\b\d+(?:\.\d+)?%?\b', claim))
            src_nums = set(re.findall(r'\b\d+(?:\.\d+)?%?\b', src_text))
            if claim_nums:
                num_overlap = claim_nums.intersection(src_nums)
                if num_overlap:
                    overlap_ratio = min(1.0, overlap_ratio + 0.35)

            if overlap_ratio > best_score:
                best_score = overlap_ratio
                best_source = src

        is_grounded = best_score >= self.threshold

        return {
            "claim": claim,
            "is_grounded": is_grounded,
            "confidence": round(best_score, 3),
            "matched_source_id": best_source.get("id") if (best_source and is_grounded) else "None (Ungrounded/Hallucination)",
            "source_type": best_source.get("type") if (best_source and is_grounded) else "unsupported",
            "matched_snippet": best_source.get("content", "")[:120] if (best_source and is_grounded) else ""
        }

    def audit_response(self, response_text: str, vector_chunks: List[str] = None, graph_facts: List[str] = None) -> Dict[str, Any]:
        """
        Performs a full claim-level provenance audit of the entire response.
        """
        evidence_sources = []
        
        if vector_chunks:
            for idx, chunk in enumerate(vector_chunks):
                evidence_sources.append({
                    "id": f"SEC_10K_Chunk_{idx+1}",
                    "type": "vector_chunk",
                    "content": chunk
                })

        if graph_facts:
            for idx, fact in enumerate(graph_facts):
                evidence_sources.append({
                    "id": f"KG_Triple_{idx+1}",
                    "type": "kg_triple",
                    "content": fact
                })

        claims = self.extract_claims(response_text)
        if not claims:
            return {
                "total_claims": 0,
                "grounded_claims": 0,
                "hallucinated_claims": 0,
                "faithfulness_score": 100.0,
                "hallucination_rate": 0.0,
                "claims": []
            }

        verified_claims = [self.verify_claim(c, evidence_sources) for c in claims]
        grounded_count = sum(1 for c in verified_claims if c["is_grounded"])
        faithfulness = (grounded_count / len(claims)) * 100.0

        return {
            "total_claims": len(claims),
            "grounded_claims": grounded_count,
            "hallucinated_claims": len(claims) - grounded_count,
            "faithfulness_score": round(faithfulness, 2),
            "hallucination_rate": round(100.0 - faithfulness, 2),
            "claims": verified_claims
        }
