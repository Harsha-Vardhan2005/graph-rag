import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv(override=True)

USERNAME = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER") or os.getenv("USERNAME", "24708389")
PASSWORD = os.getenv("NEO4J_PASSWORD") or os.getenv("PASSWORD", "")
URI = os.getenv("NEO4J_URI") or os.getenv("URI") or f"neo4j+ssc://{USERNAME}.databases.neo4j.io"

driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))

# Pick 5 patterns worth turning into questions (mix of 2-hop and 3-hop, avoid pure ownership trivia)
PATTERNS = [
    ("PERSON", "has_stake_in", "ORG", "discloses", "FIN_METRIC", "positively_impacts", "PRODUCT"),
    ("PERSON", "discloses", "ORG", "discloses", "FIN_METRIC", "impacted_by", "MACRO_CONDITION"),
    ("ORG_REG", "regulates", "ORG", "discloses", "FIN_METRIC", None, None),
    ("FIN_MARKET", "negatively_impacts", "ORG", "discloses", "FIN_METRIC", None, None),
    ("PERSON", "announces", "ORG", "discloses", "FIN_METRIC", "impacted_by", "RISK_FACTOR"),
]

def get_instance_2hop(tx, type1, rel1, type2, rel2, type3):
    query = """
    MATCH (a:Entity {entity_type:$type1})-[r1:RELATION {type:$rel1}]->(b:Entity {entity_type:$type2})
          -[r2:RELATION {type:$rel2}]->(c:Entity {entity_type:$type3})
    RETURN a.name AS e1, b.name AS e2, c.name AS e3,
           b.ticker AS ticker, r1.year AS year1, r2.year AS year2,
           r1.chunk_text AS ctx1, r2.chunk_text AS ctx2
    LIMIT 3
    """
    return list(tx.run(query, type1=type1, rel1=rel1, type2=type2, rel2=rel2, type3=type3))

def get_instance_3hop(tx, type1, rel1, type2, rel2, type3, rel3, type4):
    query = """
    MATCH (a:Entity {entity_type:$type1})-[r1:RELATION {type:$rel1}]->(b:Entity {entity_type:$type2})
          -[r2:RELATION {type:$rel2}]->(c:Entity {entity_type:$type3})
          -[r3:RELATION {type:$rel3}]->(d:Entity {entity_type:$type4})
    RETURN a.name AS e1, b.name AS e2, c.name AS e3, d.name AS e4,
           b.ticker AS ticker, r1.year AS year1, r2.year AS year2, r3.year AS year3,
           r1.chunk_text AS ctx1, r2.chunk_text AS ctx2, r3.chunk_text AS ctx3
    LIMIT 3
    """
    return list(tx.run(query, type1=type1, rel1=rel1, type2=type2, rel2=rel2, type3=type3, rel3=rel3, type4=type4))

def main():
    with driver.session() as session:
        for i, pat in enumerate(PATTERNS, 1):
            type1, rel1, type2, rel2, type3, rel3, type4 = pat
            print(f"\n{'='*70}")
            if rel3 is None:
                print(f"PATTERN {i} (2-hop): ({type1}) -[{rel1}]-> ({type2}) -[{rel2}]-> ({type3})")
                results = session.execute_read(get_instance_2hop, type1, rel1, type2, rel2, type3)
                for r in results:
                    print(f"\n  {r['e1']}  --{rel1}-->  {r['e2']} ({r['ticker']}, {r['year1']})  --{rel2}-->  {r['e3']} ({r['year2']})")
                    print(f"  Context1: {r['ctx1'][:150]}...")
                    print(f"  Context2: {r['ctx2'][:150]}...")
            else:
                print(f"PATTERN {i} (3-hop): ({type1}) -[{rel1}]-> ({type2}) -[{rel2}]-> ({type3}) -[{rel3}]-> ({type4})")
                results = session.execute_read(get_instance_3hop, type1, rel1, type2, rel2, type3, rel3, type4)
                for r in results:
                    print(f"\n  {r['e1']}  --{rel1}-->  {r['e2']} ({r['ticker']}, {r['year1']})  --{rel2}-->  {r['e3']} ({r['year2']})  --{rel3}-->  {r['e4']} ({r['year3']})")
                    print(f"  Context1: {r['ctx1'][:150]}...")
                    print(f"  Context2: {r['ctx2'][:150]}...")
                    print(f"  Context3: {r['ctx3'][:150]}...")

if __name__ == "__main__":
    main()
    driver.close()