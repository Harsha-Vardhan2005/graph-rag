from datasets import load_dataset
import pandas as pd

print("Loading dataset...")
dataset = load_dataset("domyn/FinReflectKG")

print("Filtering...")
subset = dataset["train"].filter(lambda x: x["ticker"] in ["AAPL", "MSFT"])

print(f"\nTotal rows (AAPL + MSFT): {len(subset):,}")

df = subset.to_pandas()

print(f"\nBreakdown by ticker:")
print(df["ticker"].value_counts())

print(f"\nBreakdown by year:")
print(df.groupby(["ticker", "year"]).size().reset_index(name="count").to_string())

print(f"\nColumns available: {df.columns.tolist()}")

print(f"\nEntity types present:")
print(df["entity_type"].value_counts().to_string())

print(f"\nTop 10 relationship types:")
print(df["relationship"].value_counts().head(10).to_string())

# Preview
print(f"\nSample row:")
print(df.iloc[0])

df.to_csv("finreflectkg_aapl_msft.csv", index=False)
print("\nSaved filtered subset to finreflectkg_aapl_msft.csv")