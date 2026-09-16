import os
import re
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv(override=True)

USERNAME = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER") or os.getenv("USERNAME", "24708389")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD") or os.getenv("PASSWORD", "")
URI = os.getenv("NEO4J_URI") or os.getenv("URI") or f"neo4j+ssc://{USERNAME}.databases.neo4j.io"

driver = GraphDatabase.driver(URI, auth=(USERNAME, NEO4J_PASSWORD))


def fetch_metric_across_years(tx, ticker, metric_name):
    query = """
    MATCH (org:Entity {ticker:$ticker})-[r:RELATION {type:"discloses"}]->(m:Entity {entity_type:"FIN_METRIC"})
    WHERE toLower(m.name) CONTAINS toLower($metric_name)
    RETURN m.name AS metric_name, r.year AS year, r.triplet_id AS triplet_id,
           r.chunk_text AS chunk_text, r.source_file AS source_file
    ORDER BY r.year
    """
    return list(tx.run(query, ticker=ticker, metric_name=metric_name))


def extract_numeric_value(text, metric_hint=""):
    """
    Looks for a number specifically following the metric name (e.g., 'Net sales')
    in table-style text, rather than the first number anywhere in the chunk.
    """
    if not text:
        return None

    search_text = text
    if metric_hint:
        # Find the metric label in the text, then only look at what follows it
        idx = text.lower().find(metric_hint.lower())
        if idx != -1:
            search_text = text[idx: idx + 150]  # look just after the label

    # Prefer numbers with a $ sign or comma-separated thousands (real financial figures)
    matches = re.findall(r'\$\s?[\d,]+\.?\d*|\b\d{1,3}(?:,\d{3})+\.?\d*\b', search_text)
    numbers = []
    for m in matches:
        cleaned = m.replace("$", "").replace(",", "").strip()
        try:
            numbers.append(float(cleaned))
        except ValueError:
            continue
    return numbers[0] if numbers else None


def compute_growth_rate(value_start, value_end):
    if value_start is None or value_end is None or value_start == 0:
        return None
    return round(((value_end - value_start) / value_start) * 100, 2)


def symbolic_compute_yoy_growth(ticker, metric_name, year_start, year_end):
    """
    Main tool function: computes year-over-year growth for a metric,
    returns the computed value PLUS the exact triplets/chunks it came from.
    """
    with driver.session() as session:
        records = session.execute_read(fetch_metric_across_years, ticker, metric_name)

    year_start_record = next((r for r in records if r["year"] == year_start), None)
    year_end_record = next((r for r in records if r["year"] == year_end), None)

    if not year_start_record or not year_end_record:
        return {
            "success": False,
            "reason": f"Could not find both {year_start} and {year_end} data for '{metric_name}' ({ticker}).",
            "available_years": sorted(set(r["year"] for r in records)) if records else []
        }

    val_start = extract_numeric_value(year_start_record["chunk_text"], metric_hint="net sales")
    val_end = extract_numeric_value(year_end_record["chunk_text"], metric_hint="net sales")

    growth = compute_growth_rate(val_start, val_end)

    return {
        "success": growth is not None,
        "metric": metric_name,
        "ticker": ticker,
        "year_start": year_start,
        "value_start": val_start,
        "year_end": year_end,
        "value_end": val_end,
        "growth_rate_pct": growth,
        "provenance": {
            "year_start_triplet_id": year_start_record["triplet_id"],
            "year_start_source": year_start_record["source_file"],
            "year_start_evidence": year_start_record["chunk_text"][:300],
            "year_end_triplet_id": year_end_record["triplet_id"],
            "year_end_source": year_end_record["source_file"],
            "year_end_evidence": year_end_record["chunk_text"][:300],
        }
    }


if __name__ == "__main__":
    result = symbolic_compute_yoy_growth("AAPL", "net sale", 2014, 2015)
    print("=== Symbolic Computation Result ===")
    for k, v in result.items():
        if k == "provenance":
            print("\nProvenance:")
            for pk, pv in v.items():
                print(f"  {pk}: {pv}")
        else:
            print(f"{k}: {v}")

    driver.close()