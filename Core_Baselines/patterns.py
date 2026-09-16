import os
import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv(override=True)

USERNAME = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER") or os.getenv("USERNAME", "24708389")
PASSWORD = os.getenv("NEO4J_PASSWORD") or os.getenv("PASSWORD", "")
URI = os.getenv("NEO4J_URI") or os.getenv("URI") or f"neo4j+ssc://{USERNAME}.databases.neo4j.io"

driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))

def find_2hop_patterns(tx):
    query = """
    MATCH (a:Entity)-[r1:RELATION]->(b:Entity)-[r2:RELATION]->(c:Entity)
    WHERE a.entity_type <> b.entity_type AND b.entity_type <> c.entity_type
    RETURN a.entity_type AS type1, r1.type AS rel1,
           b.entity_type AS type2, r2.type AS rel2,
           c.entity_type AS type3,
           count(DISTINCT [a.name, b.name, c.name]) AS freq
    ORDER BY freq DESC
    LIMIT 20
    """
    return list(tx.run(query))

def find_3hop_patterns(tx, type1, rel1, type2, rel2, type3):
    """Extend a specific known 2-hop pattern by one more hop, instead of scanning all 3-hop paths blindly."""
    query = """
    MATCH (a:Entity {entity_type:$type1})-[r1:RELATION {type:$rel1}]->(b:Entity {entity_type:$type2})
          -[r2:RELATION {type:$rel2}]->(c:Entity {entity_type:$type3})
    MATCH (c)-[r3:RELATION]->(d:Entity)
    WHERE c.entity_type <> d.entity_type
    RETURN r3.type AS rel3, d.entity_type AS type4,
           count(DISTINCT [a.name, b.name, c.name, d.name]) AS freq
    ORDER BY freq DESC
    LIMIT 5
    """
    return list(tx.run(query, type1=type1, rel1=rel1, type2=type2, rel2=rel2, type3=type3))

def get_sample_instance(tx, type1, rel1, type2, rel2, type3):
    query = """
    MATCH (a:Entity {entity_type:$type1})-[r1:RELATION {type:$rel1}]->(b:Entity {entity_type:$type2})
          -[r2:RELATION {type:$rel2}]->(c:Entity {entity_type:$type3})
    RETURN a.name AS entity1, b.name AS entity2, c.name AS entity3,
           r1.chunk_text AS context1, r2.chunk_text AS context2
    LIMIT 1
    """
    return tx.run(query, type1=type1, rel1=rel1, type2=type2, rel2=rel2, type3=type3).single()

def main():
    with driver.session() as session:
        print("=== Frequent 2-hop patterns in your graph ===\n")
        patterns_2hop = session.execute_read(find_2hop_patterns)
        for p in patterns_2hop:
            print(f"({p['type1']}) -[{p['rel1']}]-> ({p['type2']}) -[{p['rel2']}]-> ({p['type3']})  | freq={p['freq']}")

        print("\n=== 3-hop extensions of top 5 2-hop patterns ===\n")
        for p in patterns_2hop[:5]:
            extensions = session.execute_read(
                find_3hop_patterns, p['type1'], p['rel1'], p['type2'], p['rel2'], p['type3']
            )
            print(f"\nBase: ({p['type1']}) -[{p['rel1']}]-> ({p['type2']}) -[{p['rel2']}]-> ({p['type3']})")
            for e in extensions:
                print(f"  + -[{e['rel3']}]-> ({e['type4']})  | freq={e['freq']}")

        # Show one concrete example for the top 2-hop pattern
        if patterns_2hop:
            top = patterns_2hop[0]
            print(f"\n=== Example instance of top 2-hop pattern ===")
            sample = session.execute_read(
                get_sample_instance, top['type1'], top['rel1'], top['type2'], top['rel2'], top['type3']
            )
            if sample:
                print(f"Entity1: {sample['entity1']}")
                print(f"Entity2: {sample['entity2']}")
                print(f"Entity3: {sample['entity3']}")
                print(f"Context1: {sample['context1'][:200]}...")
                print(f"Context2: {sample['context2'][:200]}...")

if __name__ == "__main__":
    main()
    driver.close()