"""
Extract 2-hop pattern instances for MSFT from Neo4j AuraDB.
Run locally: pip install neo4j
"""

import os
import json
from dotenv import load_dotenv, find_dotenv
from neo4j import GraphDatabase

load_dotenv(find_dotenv(), override=True)

USERNAME = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER") or os.getenv("USERNAME", "24708389")
PASSWORD = os.getenv("NEO4J_PASSWORD") or os.getenv("PASSWORD", "")
URI = os.getenv("NEO4J_URI") or os.getenv("URI") or f"neo4j+ssc://{USERNAME}.databases.neo4j.io"

QUERY = """
MATCH (a:Entity {ticker:'MSFT'})-[r1:RELATION]->(b:Entity)-[r2:RELATION]->(c:Entity)
WHERE a <> c
RETURN a.name AS entity1, a.entity_type AS type1,
       r1.type AS rel1, r1.chunk_text AS chunk1, r1.source_file AS source1, r1.year AS year1,
       b.name AS entity2, b.entity_type AS type2,
       r2.type AS rel2, r2.chunk_text AS chunk2, r2.source_file AS source2, r2.year AS year2,
       c.name AS entity3, c.entity_type AS type3
LIMIT 15
"""

def main():
    driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
    results = []
    with driver.session() as session:
        records = session.run(QUERY)
        for r in records:
            results.append(dict(r))
    driver.close()

    # Print compactly (skip long chunk_text in console, save full to file)
    for i, r in enumerate(results):
        print(f"\n--- Pattern {i+1} ---")
        print(f"{r['entity1']} ({r['type1']}) --[{r['rel1']}]--> {r['entity2']} ({r['type2']}) --[{r['rel2']}]--> {r['entity3']} ({r['type3']})")
        print(f"  hop1 source: {r['source1']} year:{r['year1']}")
        print(f"  hop2 source: {r['source2']} year:{r['year2']}")

    with open("msft_patterns.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n\nSaved full output (with chunk_text) to msft_patterns.json")

if __name__ == "__main__":
    main()