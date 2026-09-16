import pandas as pd
import re

df = pd.read_csv("finreflectkg_aapl_msft.csv")
sub = df[(df['ticker'] == 'AAPL') & (df['relationship'] == 'discloses') & (df['target_type'] == 'FIN_METRIC')]

print("Available 2021 & 2022 metrics for AAPL:")
for target in ["net sale", "total net sale", "net income", "revenue", "gross margin", "operate income"]:
    t_rows = sub[sub['target'] == target]
    y21 = t_rows[t_rows['year'] == 2021]
    y22 = t_rows[t_rows['year'] == 2022]
    print(f" - '{target}': 2021 rows = {len(y21)}, 2022 rows = {len(y22)}")

print("\nInspecting 'total net sale' and 'net income' chunk texts for 2021 & 2022:")
for target in ["total net sale", "net income"]:
    for y in [2021, 2022]:
        rows = sub[(sub['target'] == target) & (sub['year'] == y)]
        for _, r in rows.iterrows():
            nums = re.findall(r'\$\s*([0-9]{1,3}(?:,[0-9]{3})+|\b[0-9]{1,3}(?:,[0-9]{3})+)', r['chunk_text'])
            if nums:
                print(f"Target '{target}' | Year {y} | Extracted: {nums[:4]} | Text: {r['chunk_text'][:80]}...")
                break
