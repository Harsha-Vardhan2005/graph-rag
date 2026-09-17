# FinReflectKG: Interactive Demo Query Guide


## 🚀 How to Launch the Demo
In terminal, run:
```powershell
python run_ui.py
```
*(Wait 4-5 seconds for models to load; the browser will open automatically).*

---

## 📍 Tab 1: Unified Query Explorer
*Tests end-to-end routing, Knowledge Graph evidence tables, and answer synthesis.*

### Query 1: Open-Ended Overview (Fast-Path Vector Search)
* **Copy Query:** `What is the general business description and principal products of Apple Inc?`
* **Expected Route:** `SIMPLE_VECTOR`
* **What to Notice:** Router chooses the fast dense embedding path, retrieving text passages with low latency (~0.3s).

### Query 2: Multi-Hop Relational Risks (Knowledge Graph Traversal)
* **Copy Query:** `What risk factors and market conditions affect Apple's supply chain and revenue?`
* **Expected Route:** `GRAPH_MULTIHOP`
* **What to Notice:** Knowledge Graph retrieves relational triples and renders them as categorized Markdown tables (Supply Chain Risks, Revenue Risks).

### Query 3: Multi-Year Math Calculation (Deterministic Symbolic Engine)
* **Copy Query:** `What was Apple's percentage change in net sales from 2021 to 2022?`
* **Expected Route:** `SYMBOLIC_COMPUTE`
* **What to Notice:** Bypasses LLM arithmetic guesswork and executes exact math: `FY2021 ($365,817M) -> FY2022 ($394,328M) = +7.79% YoY`.

### Query 4: Multi-Company Disclosures (Microsoft SEC Metrics)
* **Copy Query:** `Which key financial metrics does Microsoft disclose in its SEC reports?`
* **Expected Route:** `GRAPH_MULTIHOP`
* **What to Notice:** Pulls exact `discloses` triples for Microsoft (`net income`, `select financial data`, `financial statement data`).

### Query 5: Guardrail & Scope Verification (Out-of-Scope Test)
* **Copy Query:** `What are Tesla's vehicle delivery numbers for the current year?`
* **Expected Route:** `OUT_OF_SCOPE`
* **What to Notice:** Detects that Tesla is outside the active AAPL/MSFT index and returns a polite scope notice without hallucinating.

---

## ⚡ Tab 2: Adaptive Routing Inspector
*Tests side-by-side execution to prove the **~64% latency savings** on simple queries.*

### Query 1: Descriptive Fast-Path
* **Copy Query:** `What is the general business description and principal products of Apple Inc?`
* **What to Notice:** Compares Fast-Path Vector (0.29s) vs Full Graph Traversal (0.82s) $\rightarrow$ displays **-64% latency reduction**.

### Query :
`which key financial metrics does microsoft disclose in SEC reports?`

### Query 2: Hardware & Software Portfolio
* **Copy Query:** `What are Apple's primary hardware and software product lines?`
* **What to Notice:** Direct semantic passage retrieval answers in sub-second time without querying the database graph.

### Query 3: Regulatory Compliance Context
* **Copy Query:** `Which regulatory bodies oversee Apple regarding its financial disclosures?`
* **What to Notice:** Router recommends `GRAPH_MULTIHOP` because structured relationships (`[ORG_REG] --regulates--> [ORG]`) are necessary.

---

## 📊 Tab 3: Structured Evidence Inspector
*Tests Raw Flat Triples vs Typed Markdown Tables to prove **Distractor Resistance**.*

### Query 1: Metric Disclosures vs Accounting Policy Distractors
* **Copy Query:** `Which key financial metrics does Microsoft disclose in its SEC reports?`
* **What to Notice:**
  * **Raw Baseline:** Mixes `[ACCOUNTING_POLICY] stock-based compensation` with metrics, causing LLM distraction.
  * **Proposed Tables:** Isolates `[FIN_METRIC]` in its own table, achieving **100% Distractor Resistance**.

### Query 2: Regulatory Bodies vs Company Affiliates
* **Copy Query:** `Which regulatory bodies oversee Apple regarding its financial disclosures?`
* **What to Notice:** Structured tables categorize oversight bodies under `[REGULATORY]` (SEC, DOJ) separate from general companies.

### Query 3: Market Risk Conditions
* **Copy Query:** `What financial market conditions does Apple disclose as negatively impacting its financial metrics?`
* **What to Notice:** Cleanly groups `[FIN_MARKET]` entities (`liquidity`, `credit market`) linked via `negatively_impacts`.
