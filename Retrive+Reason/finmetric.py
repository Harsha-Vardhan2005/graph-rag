import os
import json
from dotenv import load_dotenv, find_dotenv
from neo4j import GraphDatabase

load_dotenv(find_dotenv(), override=True)

USERNAME = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER") or os.getenv("USERNAME", "24708389")
PASSWORD = os.getenv("NEO4J_PASSWORD") or os.getenv("PASSWORD", "")
URI = os.getenv("NEO4J_URI") or os.getenv("URI") or f"neo4j+ssc://{USERNAME}.databases.neo4j.io"

QUERY = """
MATCH (a:Entity {name:'msft', entity_type:'ORG'})-[r1:RELATION {type:'has_stake_in'}]->(b:Entity)-[r2:RELATION {type:'discloses'}]->(c:Entity {entity_type:'FIN_METRIC'})
RETURN a.name AS entity1,
       b.name AS entity2, b.entity_type AS type2,
       c.name AS entity3, c.entity_type AS type3,
       r1.chunk_text AS chunk1, r1.source_file AS source1, r1.year AS year1,
       r2.chunk_text AS chunk2, r2.source_file AS source2, r2.year AS year2,
       r2.start_date AS start2, r2.end_date AS end2
LIMIT 10
"""

def main():
    driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
    results = []
    with driver.session() as session:
        records = session.run(QUERY)
        for r in records:
            results.append(dict(r))
    driver.close()

    for i, r in enumerate(results):
        print(f"\n--- Instance {i+1} ---")
        print(f"msft --[has_stake_in]--> {r['entity2']} ({r['type2']}) --[discloses]--> {r['entity3']} ({r['type3']})")
        print(f"  hop2 (the fact): {r['chunk2'][:300]}...")
        print(f"  source: {r['source2']}  year: {r['year2']}  period: {r['start2']} to {r['end2']}")

    with open("msft_finmetric_instances.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved full detail to msft_finmetric_instances.json")

if __name__ == "__main__":
    main()