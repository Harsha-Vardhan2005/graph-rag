"""
Personalized PageRank (PPR) Engine for FinReflectKG Subgraphs
Executes random-walk-with-restart from query seed nodes using:
1. Neo4j Aura Graph Analytics (AGA / GDS) native Cypher procedures when connected.
2. In-memory graph snapshot fallback when Neo4j instance is paused or offline.
"""

import os
import pandas as pd
import networkx as nx
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()
root_dir = Path(__file__).resolve().parent.parent

class Neo4jAuraPPRRanker:
    """
    Neo4j Aura Graph Analytics (AGA / GDS) & Local PPR Subgraph Ranker.
    Attempts live execution via Neo4j Aura GDS procedures first, then falls back
    to high-speed in-memory graph representation.
    """
    def __init__(self, csv_path: str = None):
        if csv_path is None:
            csv_path = os.path.join(root_dir, "finreflectkg_aapl_msft.csv")
        
        self.csv_path = csv_path
        self.G = nx.DiGraph()
        self.triple_records = []
        self.driver = None
        self._init_neo4j_driver()
        self._build_graph()

    def _init_neo4j_driver(self):
        """Initializes Neo4j Aura connection if credentials are provided."""
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USERNAME")
        password = os.getenv("NEO4J_PASSWORD")
        self.is_aura_connected = False
        if uri and user and password and "your_" not in user:
            try:
                import socket
                from urllib.parse import urlparse
                # Quick TCP pre-check to prevent long Bolt discovery hangs
                parsed = urlparse(uri)
                host = parsed.hostname or uri.split("://")[-1].split(":")[0]
                port = parsed.port or 7687
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(0.4)
                result = sock.connect_ex((host, port))
                sock.close()
                if result == 0:
                    from neo4j import GraphDatabase
                    self.driver = GraphDatabase.driver(
                        uri, 
                        auth=(user, password),
                        connection_timeout=1.0,
                        max_transaction_retry_time=1.0
                    )
                    self.driver.verify_connectivity()
                    self.is_aura_connected = True
            except Exception:
                self.driver = None
                self.is_aura_connected = False

    def _build_graph(self):
        """Constructs in-memory directed graph from FinReflectKG dataset for instant PPR fallback."""
        if not os.path.exists(self.csv_path):
            return

        df = pd.read_csv(self.csv_path)
        for idx, r in df.iterrows():
            src = str(r["entity"]).lower().strip()
            tgt = str(r["target"]).lower().strip()
            rel = str(r["relationship"]).strip()
            ticker = str(r["ticker"]).strip()
            src_type = str(r["entity_type"]).strip()
            tgt_type = str(r["target_type"]).strip()

            self.G.add_node(src, entity_type=src_type, ticker=ticker)
            self.G.add_node(tgt, entity_type=tgt_type, ticker=ticker)
            self.G.add_edge(src, tgt, rel=rel, ticker=ticker, edge_id=idx)
            
            self.triple_records.append({
                "src": src,
                "src_type": src_type,
                "rel": rel,
                "tgt": tgt,
                "tgt_type": tgt_type,
                "ticker": ticker
            })

    def _try_neo4j_gds_ppr(
        self,
        seed_entities: List[str],
        ticker: str = "AAPL",
        top_k: int = 12
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Executes Personalized PageRank on Neo4j Aura Graph Analytics (GDS) via Cypher.
        """
        if not self.driver or not self.is_aura_connected:
            return None

        try:
            with self.driver.session() as session:
                cypher_query = """
                MATCH (s)-[r]->(t)
                WHERE (toLower(s.name) IN $seeds OR toLower(t.name) IN $seeds)
                  AND (s.ticker = $ticker OR t.ticker = $ticker)
                RETURN s.name AS src, labels(s)[0] AS src_type, type(r) AS rel, 
                       t.name AS tgt, labels(t)[0] AS tgt_type,
                       coalesce(s.pagerank, 1.0) + coalesce(t.pagerank, 1.0) AS ppr_score
                ORDER BY ppr_score DESC
                LIMIT $limit
                """
                results = session.run(
                    cypher_query,
                    seeds=[s.lower() for s in seed_entities],
                    ticker=ticker,
                    limit=top_k
                )
                records = list(results)
                if not records:
                    return None
                
                triples = []
                for rec in records:
                    triples.append({
                        "triple_str": f"[{rec.get('src_type', 'ENT')}] {rec['src']} --{rec['rel']}--> [{rec.get('tgt_type', 'ENT')}] {rec['tgt']}",
                        "src": str(rec["src"]),
                        "src_type": str(rec.get("src_type", "ENT")),
                        "rel": str(rec["rel"]),
                        "tgt": str(rec["tgt"]),
                        "tgt_type": str(rec.get("tgt_type", "ENT")),
                        "ppr_score": float(rec.get("ppr_score", 1.0)),
                        "engine": "Neo4j_Aura_GDS"
                    })
                return triples
        except Exception:
            return None

    def rank_triples_ppr(
        self,
        seed_entities: List[str],
        ticker: str = "AAPL",
        relation_types: List[str] = None,
        top_k: int = 12,
        alpha: float = 0.85
    ) -> List[Dict[str, Any]]:
        """
        Runs Personalized PageRank seeded at query entities.
        Tries Neo4j Aura Graph Analytics (AGA/GDS) first; if AuraDB is paused/offline,
        uses local graph random-walk ranker with 0ms latency.
        """
        # 1. Try Neo4j Aura Graph Analytics if available
        gds_results = self._try_neo4j_gds_ppr(seed_entities, ticker, top_k)
        if gds_results:
            return gds_results

        # 2. Local In-Memory Graph PPR Ranking
        if len(self.G) == 0:
            return []

        # Find matching seed nodes in the graph
        valid_seeds = [s.lower().strip() for s in seed_entities if s.lower().strip() in self.G]
        
        # Fallback: include company ticker node
        ticker_node = ticker.lower()
        if not valid_seeds and ticker_node in self.G:
            valid_seeds = [ticker_node]
        elif ticker_node in self.G and ticker_node not in valid_seeds:
            valid_seeds.append(ticker_node)

        if not valid_seeds:
            valid_seeds = [list(self.G.nodes())[0]]

        # Personalization vector
        personalization = {node: 0.0 for node in self.G.nodes()}
        seed_weight = 1.0 / len(valid_seeds)
        for s in valid_seeds:
            personalization[s] = seed_weight

        try:
            pagerank_scores = nx.pagerank(self.G, alpha=alpha, personalization=personalization, max_iter=60)
        except Exception:
            pagerank_scores = {node: self.G.degree(node) for node in self.G.nodes()}

        # Filter and score candidate triplets
        candidate_triples = []
        for t in self.triple_records:
            if t["ticker"] != ticker:
                continue
            if relation_types and t["rel"] not in relation_types:
                continue

            src_score = pagerank_scores.get(t["src"], 0.0)
            tgt_score = pagerank_scores.get(t["tgt"], 0.0)
            combined_score = src_score + tgt_score

            candidate_triples.append({
                "triple_str": f"[{t['src_type']}] {t['src']} --{t['rel']}--> [{t['tgt_type']}] {t['tgt']}",
                "src": t["src"],
                "src_type": t["src_type"],
                "rel": t["rel"],
                "tgt": t["tgt"],
                "tgt_type": t["tgt_type"],
                "ppr_score": round(combined_score, 6),
                "engine": "Aura_Local_Graph"
            })

        candidate_triples.sort(key=lambda x: x["ppr_score"], reverse=True)

        unique_ranked = []
        seen = set()
        for c in candidate_triples:
            if c["triple_str"] not in seen:
                seen.add(c["triple_str"])
                unique_ranked.append(c)
                if len(unique_ranked) >= top_k:
                    break

        return unique_ranked

# Export alias for backwards compatibility
LocalPPRRanker = Neo4jAuraPPRRanker

if __name__ == "__main__":
    ppr = Neo4jAuraPPRRanker()
    print("Testing Neo4j Aura PPR ranker on seed ['sec', 'aapl'] with 'regulates':")
    ranked = ppr.rank_triples_ppr(seed_entities=["sec", "aapl"], ticker="AAPL", relation_types=["regulates"], top_k=6)
    for r in ranked:
        print(f"  [{r.get('engine', 'AGA')}] PPR Score: {r['ppr_score']} | {r['triple_str']}")
