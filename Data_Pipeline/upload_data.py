import os
import time
import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase

# 1. Load environment variables with override=True so Windows system %USERNAME% is overridden
load_dotenv(override=True)

USERNAME = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER") or os.getenv("USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD") or os.getenv("PASSWORD")

if not USERNAME or not PASSWORD:
    raise ValueError("USERNAME and PASSWORD must be set in your .env file!")

# Construct URI based on the username/instance ID if not explicitly provided
URI = os.getenv("NEO4J_URI") or os.getenv("URI") or f"neo4j+ssc://{USERNAME}.databases.neo4j.io"

print(f"Connecting to Neo4j AuraDB at: {URI}")
print(f"Authenticated as user: {USERNAME}")

driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))


def init_database(session):
    """
    Creates indexes and constraints before inserting data.
    This speeds up UNWIND + MERGE dramatically on AuraDB free tier.
    """
    print("\n[1/3] Setting up schema indexes & constraints...")
    
    # Create composite index on (name, ticker) for faster entity lookups
    session.run("CREATE INDEX entity_name_ticker IF NOT EXISTS FOR (e:Entity) ON (e.name, e.ticker)")
    
    # Create index on entity_type for downstream query filtering
    session.run("CREATE INDEX entity_type_idx IF NOT EXISTS FOR (e:Entity) ON (e.entity_type)")
    
    # Create index on triplet_id for relationship tracking
    session.run("CREATE INDEX rel_triplet_idx IF NOT EXISTS FOR ()-[r:RELATION]-() ON (r.triplet_id)")
    
    print("Schema indexes initialized successfully.")


def load_triplets(tx, rows):
    """
    Inserts a batch of rows into Neo4j using UNWIND and MERGE.
    """
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


def verify_upload(session):
    """
    Verifies and prints summary statistics after upload.
    """
    print("\n[3/3] Verifying uploaded data...")
    res_nodes = session.run("MATCH (n:Entity) RETURN count(n) AS node_count").single()
    res_rels = session.run("MATCH ()-[r:RELATION]->() RETURN count(r) AS rel_count").single()
    
    print(f"Total Nodes created: {res_nodes['node_count']:,}")
    print(f"Total Relationships created: {res_rels['rel_count']:,}")


def main():
    csv_file = "finreflectkg_aapl_msft.csv"
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Cannot find '{csv_file}' in the current directory!")

    print(f"Reading CSV file '{csv_file}'...")
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df):,} rows from CSV.")

    # Fill NaNs to prevent Neo4j data type conversion errors
    df = df.fillna("")
    records = df.to_dict("records")

    total_rows = len(records)
    batch_size = 500  # Optimal batch size for AuraDB free tier

    start_time = time.time()
    
    with driver.session() as session:
        # Step 1: Create indexes
        init_database(session)

        # Step 2: Upload batches
        print(f"\n[2/3] Uploading {total_rows:,} triplets in batches of {batch_size}...")
        for i in range(0, total_rows, batch_size):
            batch = records[i:i + batch_size]
            session.execute_write(load_triplets, batch)
            
            elapsed = time.time() - start_time
            processed = min(i + batch_size, total_rows)
            pct = (processed / total_rows) * 100
            print(f"  Processed {processed:,} / {total_rows:,} ({pct:.1f}%) | Elapsed: {elapsed:.1f}s")

        # Step 3: Verification
        verify_upload(session)

    print(f"\nAll data successfully uploaded into AuraDB in {time.time() - start_time:.1f}s.")


if __name__ == "__main__":
    try:
        main()
    finally:
        driver.close()
