import os
import json
from dotenv import load_dotenv, find_dotenv
from neo4j import GraphDatabase

load_dotenv(find_dotenv(), override=True)

USERNAME = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER") or os.getenv("USERNAME", "24708389")
PASSWORD = os.getenv("NEO4J_PASSWORD") or os.getenv("PASSWORD", "")
URI = os.getenv("NEO4J_URI") or os.getenv("URI") or f"neo4j+ssc://{USERNAME}.databases.neo4j.io"

QUERY = """
MATCH (a:Entity {name:'msft', entity_type:'ORG'})-[r:RELATION {type:'has_stake_in'}]->(b:Entity {entity_type:'SEGMENT'})-[r2:RELATION]->(c:Entity)
WHERE r2.source_file = 'MSFT_10k_2024.pdf' AND r2.chunk_text CONTAINS 'Intelligent Cloud'
RETURN DISTINCT r2.chunk_text AS chunk, r2.chunk_id AS chunk_id
LIMIT 3
"""

def main():
    driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
    with driver.session() as session:
        recs = [dict(r) for r in session.run(QUERY)]
    driver.close()

    for i, r in enumerate(recs):
        print(f"\n--- Chunk {i+1} (id={r['chunk_id']}) ---")
        print(r['chunk'])

    with open("msft_segment_full_chunk.json", "w") as f:
        json.dump(recs, f, indent=2)
    print("\nSaved to msft_segment_full_chunk.json")

if __name__ == "__main__":
    main()