import os
import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv(override=True)

# ---- AuraDB connection details ----
USERNAME = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER") or os.getenv("USERNAME", "24708389")
PASSWORD = os.getenv("NEO4J_PASSWORD") or os.getenv("PASSWORD", "")
URI = os.getenv("NEO4J_URI") or os.getenv("URI") or f"neo4j+ssc://{USERNAME}.databases.neo4j.io"

driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))

def load_triplets(tx, rows):
    query = """
    UNWIND $rows AS row
    MERGE (e:Entity {name: row.entity, ticker: row.ticker})
    SET e.entity_type = row.entity_type

    MERGE (t:Entity {name: row.target, ticker: row.ticker})
    SET t.entity_type = row.target_type

    MERGE (e)-[r:RELATION {
        type: row.relationship,
        triplet_id: row.triplet_id
    }]->(t)
    SET r.start_date = row.start_date,
        r.end_date = row.end_date,
        r.year = row.year,
        r.source_file = row.source_file,
        r.page_id = row.page_id,
        r.chunk_id = row.chunk_id,
        r.chunk_text = row.chunk_text,
        r.extraction_type = row.extraction_type
    """
    tx.run(query, rows=rows)

def main():
    df = pd.read_csv("finreflectkg_aapl_msft.csv")
    print(f"Loaded {len(df):,} rows from CSV")

    # Replace NaNs to avoid Neo4j type errors
    df = df.fillna("")

    records = df.to_dict("records")

    batch_size = 500  # AuraDB free tier: keep batches modest
    with driver.session() as session:
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            session.execute_write(load_triplets, batch)
            print(f"Loaded {i + len(batch):,} / {len(records):,}")

    print("Done loading into Neo4j AuraDB.")

if __name__ == "__main__":
    main()
    driver.close()