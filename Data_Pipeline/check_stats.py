import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv(override=True)

USERNAME = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER") or os.getenv("USERNAME", "24708389")
PASSWORD = os.getenv("NEO4J_PASSWORD") or os.getenv("PASSWORD", "")
URI = os.getenv("NEO4J_URI") or os.getenv("URI") or f"neo4j+ssc://{USERNAME}.databases.neo4j.io"

driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))

def get_stats(tx):
    q1 = "MATCH (n) RETURN count(n) AS node_count"
    q2 = "MATCH ()-[r]->() RETURN count(r) AS rel_count"
    q3 = "MATCH (n) RETURN DISTINCT n.entity_type AS type, count(*) AS count ORDER BY count(*) DESC"
    
    nodes = tx.run(q1).single()["node_count"]
    rels = tx.run(q2).single()["rel_count"]
    types = list(tx.run(q3))
    
    return nodes, rels, types

def main():
    with driver.session() as session:
        nodes, rels, types = session.execute_read(get_stats)
        print(f"Total Nodes: {nodes}")
        print(f"Total Relationships: {rels}")
        print("\nNode Counts by Entity Type:")
        for r in types:
            print(f"  {r['type'] or '<blank>'}: {r['count']}")

if __name__ == "__main__":
    main()
    driver.close()
