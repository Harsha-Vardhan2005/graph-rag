import os
import json
from dotenv import load_dotenv, find_dotenv
from neo4j import GraphDatabase

load_dotenv(find_dotenv(), override=True)

USERNAME = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER") or os.getenv("USERNAME", "24708389")
PASSWORD = os.getenv("NEO4J_PASSWORD") or os.getenv("PASSWORD", "")
URI = os.getenv("NEO4J_URI") or os.getenv("URI") or f"neo4j+ssc://{USERNAME}.databases.neo4j.io"

# Survey distinct pattern SHAPES (not instances) so we can pick ones
# that terminate in concrete, checkable entity types (numbers, named orgs, metrics)
QUERY = """
MATCH (a:Entity {ticker:'MSFT'})-[r1:RELATION]->(b:Entity)-[r2:RELATION]->(c:Entity)
WHERE a <> c
RETURN a.entity_type AS type1, r1.type AS rel1,
       b.entity_type AS type2, r2.type AS rel2,
       c.entity_type AS type3,
       count(*) AS freq
ORDER BY freq DESC
LIMIT 30
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
        print(f"{i+1}. ({r['type1']}) -[{r['rel1']}]-> ({r['type2']}) -[{r['rel2']}]-> ({r['type3']})   x{r['freq']}")

    with open("msft_pattern_shapes.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to msft_pattern_shapes.json")

if __name__ == "__main__":
    main()