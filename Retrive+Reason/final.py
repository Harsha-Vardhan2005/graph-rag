import os
import json
from dotenv import load_dotenv, find_dotenv
from neo4j import GraphDatabase

load_dotenv(find_dotenv(), override=True)

USERNAME = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER") or os.getenv("USERNAME", "24708389")
PASSWORD = os.getenv("NEO4J_PASSWORD") or os.getenv("PASSWORD", "")
URI = os.getenv("NEO4J_URI") or os.getenv("URI") or f"neo4j+ssc://{USERNAME}.databases.neo4j.io"

# Query A: real 2-hop chains, excluding self-loops
QUERY_A = """
MATCH (a:Entity {name:'msft', entity_type:'ORG'})-[r1:RELATION {type:'has_stake_in'}]->(b:Entity)-[r2:RELATION {type:'discloses'}]->(c:Entity {entity_type:'FIN_METRIC'})
WHERE b.name <> 'msft'
RETURN DISTINCT b.name AS entity2, b.entity_type AS type2,
       c.name AS entity3,
       r2.chunk_text AS chunk2, r2.source_file AS source2, r2.year AS year2
LIMIT 10
"""

# Query B: full text of the revenue table chunk we saw (find distinct chunk_texts for net income/revenue disclosures by msft itself)
QUERY_B = """
MATCH (a:Entity {name:'msft', entity_type:'ORG'})-[r:RELATION {type:'discloses'}]->(c:Entity {entity_type:'FIN_METRIC'})
WHERE r.chunk_text CONTAINS 'Revenue' AND r.chunk_text CONTAINS '|'
RETURN DISTINCT c.name AS metric, r.chunk_text AS chunk, r.source_file AS source, r.year AS year, r.start_date AS start, r.end_date AS end
LIMIT 5
"""

def main():
    driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
    with driver.session() as session:
        print("=== QUERY A: Real 2-hop chains (non-self-loop) ===")
        recs_a = [dict(r) for r in session.run(QUERY_A)]
        for i, r in enumerate(recs_a):
            print(f"\n--- {i+1}: msft -> {r['entity2']} ({r['type2']}) -> {r['entity3']} ---")
            print(f"  chunk: {r['chunk2'][:250]}")
            print(f"  source: {r['source2']} year:{r['year2']}")

        print("\n\n=== QUERY B: Full revenue table chunks disclosed by msft ===")
        recs_b = [dict(r) for r in session.run(QUERY_B)]
        for i, r in enumerate(recs_b):
            print(f"\n--- {i+1}: metric={r['metric']} source={r['source']} year={r['year']} period={r['start']} to {r['end']} ---")
            print(r['chunk'])
    driver.close()

    with open("msft_query_results.json", "w") as f:
        json.dump({"chains": recs_a, "revenue_chunks": recs_b}, f, indent=2)
    print("\n\nSaved to msft_query_results.json")

if __name__ == "__main__":
    main()