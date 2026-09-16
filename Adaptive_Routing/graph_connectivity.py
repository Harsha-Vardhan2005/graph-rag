"""
Graph Connectivity Confidence Signal Calculator
Evaluates Knowledge Graph neighborhood density and direct path existence
to compute a mathematical Graph Confidence Score (0.0 to 1.0).
"""

import os
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv, find_dotenv
from neo4j import GraphDatabase

root_dir = Path(__file__).resolve().parent.parent
load_dotenv(find_dotenv(), override=True)

NEO4J_USER = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER") or os.getenv("USERNAME", "24708389")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD") or os.getenv("PASSWORD", "")
NEO4J_URI = os.getenv("NEO4J_URI") or os.getenv("URI") or f"neo4j+ssc://{NEO4J_USER}.databases.neo4j.io"

class GraphConnectivityScorer:
    def __init__(self, csv_path: str = None):
        if csv_path is None:
            csv_path = os.path.join(root_dir, "finreflectkg_aapl_msft.csv")
        
        self.csv_path = csv_path
        self.df_kg = pd.read_csv(csv_path) if os.path.exists(csv_path) else None
        self.driver = None

        # Quick TCP check to avoid Bolt connection delays when Neo4j Aura is paused
        if NEO4J_URI and NEO4J_USER and NEO4J_PASSWORD and "your_" not in NEO4J_USER:
            try:
                import socket
                from urllib.parse import urlparse
                parsed = urlparse(NEO4J_URI)
                host = parsed.hostname or NEO4J_URI.split("://")[-1].split(":")[0]
                port = parsed.port or 7687
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(0.4)
                if sock.connect_ex((host, port)) == 0:
                    self.driver = GraphDatabase.driver(
                        NEO4J_URI,
                        auth=(NEO4J_USER, NEO4J_PASSWORD),
                        connection_timeout=1.0,
                        max_transaction_retry_time=1.0
                    )
                sock.close()
            except Exception:
                self.driver = None

    def evaluate_connectivity(self, ticker: str, relation_types: List[str], entity_names: List[str] = None) -> Dict[str, Any]:
        """
        Computes the structural Graph Connectivity Signal based on matching edges.
        Returns:
          - edge_count: number of matching triples in the neighborhood
          - graph_confidence: score between 0.0 and 1.0
          - is_connected: boolean indicating whether graph traversal is viable
          - source: 'neo4j_live' or 'local_kg_snapshot'
        """
        if not relation_types:
            relation_types = ["discloses", "regulates", "subject_to", "negatively_impacts", "impacted_by", "has_stake_in"]

        matched_count = 0
        source = "local_kg_snapshot"

        # 1. Try Neo4j query first if reachable
        if self.driver:
            try:
                with self.driver.session() as session:
                    cypher = """
                    MATCH (a:Entity)-[r:RELATION]->(b:Entity)
                    WHERE (a.ticker = $ticker OR b.ticker = $ticker) AND r.type IN $rel_types
                    RETURN count(r) AS total_edges
                    """
                    result = session.run(cypher, ticker=ticker, rel_types=relation_types)
                    record = result.single()
                    if record:
                        matched_count = record["total_edges"]
                        source = "neo4j_live"
            except Exception:
                matched_count = 0
                source = "local_kg_snapshot"

        # 2. Resilient fallback to local KG dataset
        if matched_count == 0 and self.df_kg is not None:
            source = "local_kg_snapshot"
            sub = self.df_kg[
                (self.df_kg["ticker"] == ticker) & 
                (self.df_kg["relationship"].isin(relation_types))
            ]
            
            # If entity names provided, check specific entity connections
            if entity_names:
                en_low = [e.lower() for e in entity_names]
                sub_ent = sub[
                    sub["entity"].str.lower().isin(en_low) | 
                    sub["target"].str.lower().isin(en_low)
                ]
                matched_count = len(sub_ent) if not sub_ent.empty else len(sub)
            else:
                matched_count = len(sub)

        # Mathematical confidence scaling:
        # 0 edges -> 0.0 confidence (Graph cannot answer -> fallback to vector)
        # 1-3 edges -> 0.65 to 0.80
        # >= 4 edges -> 0.90 to 0.98
        if matched_count == 0:
            confidence = 0.0
            is_connected = False
        elif matched_count < 3:
            confidence = 0.75
            is_connected = True
        elif matched_count < 8:
            confidence = 0.90
            is_connected = True
        else:
            confidence = 0.98
            is_connected = True

        return {
            "ticker": ticker,
            "matched_edge_count": matched_count,
            "graph_confidence": confidence,
            "is_connected": is_connected,
            "provenance_source": source
        }

if __name__ == "__main__":
    scorer = GraphConnectivityScorer()
    print("Testing Graph Connectivity Signal on AAPL regulates:")
    res = scorer.evaluate_connectivity("AAPL", ["regulates"])
    print(res)
