# Oracle Retrieval Experiment: Isolating Reasoning Quality from Retrieval Quality

## Motivation

Our ablation pipeline (Tiers 1 to 3) measures end-to-end accuracy, where errors can come from two different sources: the retrieval step failing to surface the right evidence, or the LLM failing to reason correctly even when it has the right evidence. These two failure modes need different fixes, so we designed an oracle retrieval experiment to isolate them.

The idea: remove retrieval entirely. Hand the LLM the correct, gold-standard evidence directly, and measure whether it can reach the correct final answer through reasoning alone. If accuracy is still imperfect under this "perfect retrieval" condition, that tells us reasoning itself is a bottleneck, not just retrieval.

## Methodology

**Model tested:** llama-3.3-70b-versatile (our production LLM), via the Groq API, temperature 0.

**Benchmark used:** FinQA (Chen et al., EMNLP 2021), a peer-reviewed dataset of expert-annotated question-answer pairs over financial report excerpts. Each example includes the source text and table (used as gold evidence), the question, the gold numeric answer, and a gold "program", the exact sequence of arithmetic operations (add, subtract, divide, multiply, etc.) required to reach that answer.

We chose FinQA over a hand-constructed question set because early hand-crafted tests (built from our own MSFT subgraph) were too easy. Simple single-fact lookup questions were answered correctly 100% of the time, which does not reflect the harder multi-hop reasoning our agent actually needs to perform. FinQA provides expert-validated, genuinely difficult multi-step questions, so results are not a product of question selection bias.

**Sampling:** From the FinQA dev set (883 examples), we filtered to questions requiring 2 or more reasoning steps (360 such questions found), then sampled a stratified set of 20 questions spanning 2-step, 3-step, 4-step, and 5-step gold programs (5 questions per step count).

**Evidence formatting:** Each question's gold text and table were handed directly to the model as context, exactly as FinQA provides them. No retrieval, no agent, no tool calls. The model was asked to show its calculation and state a final numeric answer.

**Scoring:** Model answers were extracted and compared against the gold execution answer (exe_ans) with a small numeric tolerance to account for rounding.

## Results

**Overall accuracy: 12/20 (60%)**

| Reasoning steps required | Accuracy |
|---|---|
| 2 steps | 4/5 (80%) |
| 3 steps | 5/5 (100%) |
| 4 steps | 1/5 (20%) |
| 5 steps | 2/5 (40%) |

Accuracy holds up on simpler (2 to 3 step) questions but drops sharply on harder (4 to 5 step) questions, even though the model was given the correct evidence in every case. This is the core finding: reasoning failures increase with reasoning complexity, independent of retrieval.

Note: on manual review, 3 of the 8 incorrect answers (CB/2010, LMT/2013, MSI/2006) appear to stem from a likely annotation inconsistency in FinQA's gold programs for averages (an extra addition of "3" before dividing by 2 instead of 3). The model's answers in these cases were arithmetically sound as simple averages. Excluding these 3 as probable dataset noise, accuracy on the remaining 17 validated questions is 12/17 (70.6%), still showing a clear decline concentrated in the 4 to 5 step range.

## Qualitative failure analysis

Five genuine reasoning failures were identified and categorized by failure type:

1. **Self-doubt override (AMAT/2013):** the model computed the correct answer ($7.22 billion) but then second-guessed itself over an unrelated detail in the question, discarding its own correct work and answering "Cannot be determined."

2. **Goal substitution (APD/2018):** the model set up the correct compound-growth formula, then abandoned the computation partway through and reported an intermediate value (a rate) instead of the requested final quantity (a return amount).

3. **Arithmetic drift (FRT/2005):** the model misread table values partway through a multi-step calculation, producing a final answer over 10x larger than the gold answer, with visible confusion in its own reasoning trace about which quantity was being asked for.

4. **Dropped term (ALXN/2007):** in a 5-year average, the model omitted one year's value from the sum entirely, despite that value being present and unambiguous in the evidence.

5. **Sign error (C/2017/page_328-3):** the model reached the correct magnitude (37.4%) but the wrong sign, missing that the comparison should have been negative.

These are distinct, well-characterized reasoning failure modes, not random noise. None of them involved missing or ambiguous evidence. All the necessary facts were directly present in the context handed to the model.

## Implication for contribution 2

This experiment provides direct, benchmark-validated evidence for our retrieval-vs-reasoning gap finding. Even under ideal, oracle-level retrieval, our production LLM's accuracy degrades substantially as the number of required reasoning steps increases (80 to 100% on 2 to 3 step questions, down to 20 to 40% on 4 to 5 step questions). This confirms that structured evidence formatting and reasoning-support mechanisms (our proposed fix) are addressing a real, measurable component of pipeline error, not just a retrieval shortfall.

## Suggested next steps

- Scale the sample size beyond 20 questions (FinQA's dev set has 360 multi-step questions available) for a more statistically robust accuracy estimate per step count.
- Repeat this test after Groq's model deprecation is resolved, to confirm results hold on the replacement production model.
- Consider running the same 20 questions through our full agentic pipeline (Tier 3) for direct comparison against this oracle ceiling, to quantify exactly how much of our end-to-end error is retrieval-driven versus reasoning-driven.
