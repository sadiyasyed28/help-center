🔎 Help Center Search

«Leakage-safe, typo-tolerant information retrieval system for mapping natural-language support queries to the most relevant Help Center articles.»

A complete retrieval experimentation pipeline designed to determine how effectively different lexical retrieval strategies can map noisy, conversational user queries to the correct support article.

The project evaluates progressively stronger retrieval approaches—from simple word overlap and Jaccard similarity to character n-gram TF-IDF, query cleaning, training-query document expansion, fusion experiments, leakage auditing, error analysis, and final test-set prediction.

---

🏆 Final Result

The final selected model is Document Expansion + Character N-Gram TF-IDF.

Final Group-Aware 5-Fold CV Performance

Metric| Score
🎯 Recall@1| 78.28%
🎯 Recall@3| 89.37%
📈 MRR| 0.8466
📊 Recall@3 Std. Dev.| 0.0346
📉 Recall@3 Min| 86.05%
📈 Recall@3 Max| 93.02%

Why this result matters

The evaluation uses GroupKFold, rather than ordinary random cross-validation.

The dataset contains exact duplicates and highly similar/paraphrased queries. Randomly splitting these queries can place nearly identical queries in both training and validation sets, causing severe leakage.

The final evaluation prevents this.

«Final realistic validation performance: 89.37% Recall@3.»

---

📌 Problem Statement

Given a user's natural-language support query:

"how do i switch where deposits go"

the system must retrieve the most relevant Help Center article from a fixed collection of articles.

The challenge is that users do not necessarily use the same terminology as the article.

For example:

User query:
"stop fake orders with stolen cards"

Target article:
"Preventing fraudulent charges"

The concepts are related, but the vocabulary is very different.

Queries may also contain:

- conversational filler
- punctuation
- urgency markers
- spelling mistakes
- abbreviations
- paraphrases
- colloquial terminology
- incomplete descriptions
- ambiguous wording

The project investigates how far traditional lexical retrieval can go under these constraints.

---

🎯 Objectives

The project was designed around several goals:

- Build a reproducible Help Center retrieval pipeline.
- Establish simple lexical baselines.
- Measure progressively stronger retrieval approaches.
- Handle spelling variation and noisy query language.
- Learn from training queries without leaking validation information.
- Detect duplicate and near-duplicate queries.
- Use leakage-safe cross-validation.
- Compare query-learning and fusion approaches.
- Analyze retrieval failures.
- Select the simplest model that provides strong performance.
- Generate deterministic predictions for the unseen test set.
- Package the complete experiment for reproducibility.

---

📊 Dataset

The project uses three datasets:

data/
├── articles.csv
├── queries_train.csv
└── queries_test.csv

Articles

24 Help Center articles

Each article contains:

- "article_id"
- title
- article body
- category information

Training queries

216 labeled queries

Each query has a known relevant article.

The dataset is perfectly balanced:

24 articles
×
9 training queries per article
=
216 training queries

Test queries

96 unlabeled queries

These queries are used only for the final prediction stage.

Their labels are never used during model development or tuning.

---

🧪 Experiment Architecture

The project was developed through a structured multi-phase experiment.

                         ┌──────────────────────┐
                         │      Raw Dataset     │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Phase 1: Exploration │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Phase 2: Baselines   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Phase 3: Lexical IR  │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Phase 4: Leakage-    │
                         │ Safe Query Learning  │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Phase 5: Fusion &    │
                         │ Light Tuning         │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Phase 6: Error       │
                         │ Analysis             │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Phase 7: Final Fit   │
                         │ & Test Predictions   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Phase 8/9: Docs &    │
                         │ Final Audit          │
                         └──────────────────────┘

---

🔬 Phase 1 — Data Exploration

The first phase examined the structure and difficulty of the retrieval problem.

Key findings

216 training queries

24 articles

9 queries per article

Query length:

Dataset| Avg Characters| Avg Tokens
Train| 39.7| 7.5
Test| 41.1| 7.9

The analysis also revealed substantial conversational filler.

Examples:

hi
help
need help
how do i
can you tell
please
asap

A major finding was the vocabulary gap between queries and articles.

Average query/article token overlap was only approximately:

«32.1%»

This immediately suggested that strict exact-word matching would be insufficient.

---

📏 Phase 2 — Baseline Retrieval

Two simple baselines were established.

Baseline A — Shared Word Count

Ranks articles according to the number of words shared with the query.

Metric| Score
Recall@1| 32.87%
Recall@3| 56.48%
MRR| 0.4956

Baseline B — Jaccard Similarity

Jaccard similarity normalizes the intersection by the union of query and document vocabulary.

Metric| Score
Recall@1| 44.44%
Recall@3| 65.28%
MRR| 0.5839

Jaccard became the initial strong baseline.

---

🧠 Phase 3 — Lexical Retrieval

Several traditional information-retrieval approaches were evaluated.

Model| Recall@1| Recall@3| MRR
Shared Word Count| 32.87%| 56.48%| 0.4956
Jaccard| 44.44%| 65.28%| 0.5839
TF-IDF| 53.70%| 70.37%| 0.6399
TF-IDF + Bigrams| 53.24%| 71.30%| 0.6374
Title-weighted TF-IDF| 49.07%| 68.52%| 0.6097
BM25| 56.94%| 75.00%| 0.6833
BM25 + Cleaning| 59.72%| 75.46%| 0.7023
Character TF-IDF| 68.98%| 81.48%| 0.7678

🏆 Character TF-IDF

Character n-grams dramatically outperformed word-based approaches.

Configuration:

Character n-grams: 3–5
Similarity: cosine

This was especially effective for:

- spelling mistakes
- morphological differences
- partial word matches
- punctuation variation
- small terminology changes

For example:

acocunt

can still share character patterns with:

account

without requiring an explicit spell-checking dictionary.

---

🧹 Query Cleaning

The experiments also tested deterministic removal of conversational filler.

Examples:

hi
help
need help
how do i
can you tell me
please
asap

Cleaning improved BM25 performance:

BM25
R@1: 56.94%

BM25 + Cleaning
R@1: 59.72%

The improvement was useful but relatively small compared with character n-grams.

---

⚠️ Phase 4 — Leakage Investigation

This was one of the most important parts of the project.

The training set contains:

- exact duplicate queries
- punctuation variants
- highly similar paraphrases
- near-duplicate queries

Character TF-IDF similarity revealed:

Similarity Threshold| Pairs
≥ 0.95| 13
≥ 0.90| 29
≥ 0.85| 47
≥ 0.80| 133

Using a ≥0.85 similarity threshold produced:

«49 similarity groups containing 123 queries»

This demonstrated that ordinary random cross-validation could produce misleadingly high results.

---

🔐 Leakage-Safe Cross-Validation

Instead of relying on random K-fold splitting, the project uses:

GroupKFold(n_splits=5)

Similar and duplicate queries are kept within the same group.

Therefore:

Training Fold
      │
      ├── Query A
      ├── Query B
      └── Query C

        ╳

Validation Fold
      │
      ├── Near-duplicate A'
      └── Paraphrase B'

The same query family cannot cross the train/validation boundary.

This produces a much more realistic estimate of generalization.

---

🚀 Phase 4 — Learning From Training Queries

Two approaches were tested.

v7 — Document Expansion

Training queries associated with each article were appended to that article before creating the TF-IDF representation.

Conceptually:

Article
+
Known training queries
        ↓
Expanded article representation
        ↓
Character TF-IDF
        ↓
Cosine similarity
        ↓
Ranked articles

This allows the model to learn the vocabulary users actually use when referring to an article.

Result

89.37% Group-aware Recall@3

---

v8 — Query-to-Query Retrieval

Instead of ranking articles directly, the system indexed training queries.

A new query was matched against known training queries and the corresponding article label was retrieved.

This looked extremely strong under random CV.

However, once duplicate leakage was removed:

«Query-to-query retrieval achieved only approximately 66.2% Recall@3.»

This was an important experimental result.

It demonstrated why leakage-safe evaluation matters.

---

🏆 Document Expansion Wins

Group-aware results:

Model| Recall@3
Article-only Character TF-IDF| 81.46%
Query-to-Query| ~66.2%
Document Expansion| 89.37%

Document expansion improved the realistic baseline by approximately:

«+7.91 percentage points»

or:

«+9.71% relative improvement»

---

🔀 Phase 5 — Fusion Experiments

Several fusion approaches were tested.

Model| Group R@3
v7 Reference| 89.3658%
Raw Fusion α=0.25| 85.1586%
Raw Fusion α=0.50| 89.3658%
Raw Fusion α=0.75| 88.9006%
RRF Fusion| 88.4355%
Expansion + BM25| 88.9112%

Fusion did not produce a meaningful improvement over v7.

Therefore, the project deliberately selected the simpler model.

Final model selection rule

«If a more complex model does not improve Recall@3 by at least 1 percentage point, prefer the simpler model.»

Result:

v7 Document Expansion was selected.

---

🔍 Phase 6 — Error Analysis

The final validation errors were analyzed rather than simply reporting a single metric.

There were:

«23 Recall@3 misses»

The major failure categories were:

Failure Type| Cases| Percentage
Vocabulary Gap| 14| 60.9%
Semantic Ambiguity| 8| 34.8%
Same-category Confusion| 1| 4.3%

Main conclusion

The remaining errors are primarily caused by the model's lack of true semantic understanding.

For example:

"stop fake orders with stolen cards"

versus:

"Preventing fraudulent charges"

The two expressions are semantically related, but their lexical forms differ substantially.

---

📉 Weakest Areas

The weakest categories were:

Payouts

Recall@3 ≈ 77.8%

Payments

Recall@3 ≈ 80.6%

The hardest individual article was:

A103 — When will I get paid?
Recall@3 ≈ 33.3%

This article frequently confused with other payout-related articles.

This highlights a key retrieval challenge:

«Distinguishing between articles that share a large amount of domain vocabulary but represent different user intents.»

---

🧪 Phase 7 — Final Test Prediction

After model selection, the final model was fit once using:

24 base articles
+
216 labeled training queries

The test set remained completely isolated from model development.

Test set

96 queries

Each query receives:

query_id
rank_1
rank_2
rank_3

The final file is:

predictions.csv

Output format

query_id,rank_1,rank_2,rank_3
Q...,A...,A...,A...
Q...,A...,A...,A...
...

---

✅ Prediction Validation

The final prediction validator checks:

- exactly 96 rows
- correct column names
- no duplicate query IDs
- no missing query IDs
- valid article IDs
- distinct top-3 predictions
- no null values
- referential integrity

Final result:

All validation checks passed successfully.

---

🔁 Reproducibility

The prediction pipeline was executed twice.

The resulting prediction matrices were compared programmatically.

Result:

Identical outputs

This confirms deterministic ranking and tie-breaking.

---

📁 Project Structure

experiment/
│
├── data/
│   ├── articles.csv
│   ├── queries_train.csv
│   └── queries_test.csv
│
├── src/
│   ├── evaluation.py
│   ├── phase1_analysis.py
│   ├── phase2_baseline.py
│   ├── phase3_lexical.py
│   ├── phase4_pre_audit.py
│   ├── phase4_learning.py
│   ├── phase4_verification.py
│   ├── phase5_fusion.py
│   ├── phase5_report.py
│   ├── phase6_error_analysis.py
│   ├── phase7_final_predictions.py
│   ├── phase8_documentation.py
│   └── validate_predictions.py
│
├── tests/
│   └── test_evaluation.py
│
├── reports/
│   ├── phase1/
│   ├── phase3/
│   ├── phase4/
│   ├── phase5/
│   ├── phase6/
│   ├── phase7/
│   ├── final_project_report.md
│   ├── final_experiment_summary.md
│   ├── final_pre_submission_audit.md
│   └── final_packaging_audit.md
│
├── predictions.csv
├── requirements.txt
└── .gitignore

---

⚙️ Installation

Clone the repository and enter the project directory.

Create a virtual environment:

python -m venv .venv

Activate it.

Windows

.venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

---

▶️ Running the Project

Run unit tests

python -m unittest discover tests

Expected result:

Ran 8 tests

OK

Validate predictions

python src/validate_predictions.py

Expected:

All validation checks passed successfully.

Generate final predictions

python src/phase7_final_predictions.py

This generates:

predictions.csv

---

📈 Evaluation Metrics

The project uses three primary retrieval metrics.

Recall@1

Measures whether the correct article is the first result.

Correct article in rank 1
-------------------------
       All queries

Recall@3

Measures whether the correct article appears anywhere in the top three results.

This is the primary metric used for model selection.

Mean Reciprocal Rank — MRR

Rewards models that place the correct article higher in the ranking.

Rank 1 → 1.00
Rank 2 → 0.50
Rank 3 → 0.33
Rank 4 → 0.25
...

---

🔐 Data Leakage Policy

The project follows strict leakage controls.

Never used during model development

Test labels
External APIs
External synonym dictionaries
External datasets
Pre-trained embedding models
Manual article mappings
Hard-coded query → article mappings

Training queries are used only where explicitly documented

The final model uses all 216 labeled training queries for document expansion only after model selection and cross-validation are complete.

The test queries are used exclusively for final prediction.

---

🚫 What This Project Does NOT Use

This project intentionally remains a traditional lexical information-retrieval system.

It does not use:

- ❌ LLMs
- ❌ OpenAI APIs
- ❌ sentence embeddings
- ❌ vector databases
- ❌ external synonym dictionaries
- ❌ external spell-checking APIs
- ❌ manually hard-coded mappings
- ❌ test-set labels

This makes the experiment useful for understanding the capabilities and limitations of classical retrieval methods.

---

💡 Key Technical Insights

1. Character n-grams are surprisingly powerful

They provide natural tolerance to:

- spelling errors
- word variants
- partial matches
- punctuation changes

without requiring external language resources.

2. Query cleaning helps—but only modestly

Removing conversational filler improves lexical retrieval, but it cannot solve semantic vocabulary gaps.

3. Query-to-query retrieval is vulnerable to leakage

It can look nearly perfect under random CV when duplicate queries cross folds.

Group-aware evaluation exposes the true performance.

4. Document expansion is much more robust

Adding real training-query language to article representations allows the system to bridge some of the vocabulary mismatch while retaining the article itself as the retrieval target.

5. More complexity did not automatically mean better performance

Fusion models failed to improve Recall@3 meaningfully.

Therefore:

«The simplest model meeting the performance target was selected.»

---

🚧 Limitations

The final model still has important limitations.

Vocabulary gaps

The system cannot reliably understand that:

"fake orders with stolen cards"

means:

"fraudulent charges"

unless the lexical representation provides enough evidence.

Semantic ambiguity

Queries such as:

"where is my money?"

may correspond to several payout-related intents.

Same-category confusion

Articles discussing similar processes naturally share vocabulary.

Limited dataset size

The experiment contains only:

24 articles
216 training queries
96 test queries

Therefore, results should not automatically be generalized to larger production-scale search systems.

---

🚀 Future Improvements

The error analysis suggests several potential next steps.

1. Dense Semantic Retrieval

Use sentence embeddings to capture semantic relationships between different expressions.

For example:

"fake orders with stolen cards"
        ↓
semantic representation
        ↓
"fraudulent charges"

2. Cross-Encoder Reranking

Retrieve a larger candidate set using lexical retrieval, then use a cross-encoder to rank the most promising candidates.

Possible architecture:

Query
  │
  ▼
Character TF-IDF
  │
  ▼
Top 10–20 candidates
  │
  ▼
Semantic Reranker
  │
  ▼
Top 3 articles

3. Hybrid Retrieval

Combine:

Lexical retrieval
+
Dense semantic retrieval
+
Reranking

This could potentially address both exact terminology and semantic vocabulary gaps.

---

🏁 Final Conclusion

This project demonstrates that a carefully designed traditional retrieval pipeline can achieve strong performance even on noisy natural-language support queries.

The final system achieved:

«89.37% Group-aware Recall@3»

without relying on:

- LLMs
- embeddings
- external APIs
- external dictionaries
- test labels
- manually constructed mappings

The most important contribution is not simply the final score.

The project demonstrates a complete experimental methodology:

Explore
   ↓
Establish Baselines
   ↓
Improve Retrieval
   ↓
Detect Leakage
   ↓
Use Group-aware CV
   ↓
Learn From Training Queries
   ↓
Compare Fusion
   ↓
Analyze Errors
   ↓
Lock Model
   ↓
Generate Test Predictions
   ↓
Validate
   ↓
Package Reproducibly

The final selected architecture is intentionally simple:

«Character TF-IDF + Training Query Document Expansion»

It provided the strongest leakage-safe performance while avoiding unnecessary model complexity.

---

📜 Final Project Status

Component| Status
Dataset integrity| ✅
Data exploration| ✅
Baselines| ✅
Lexical retrieval| ✅
Character n-gram retrieval| ✅
Duplicate audit| ✅
Group-aware CV| ✅
Query learning| ✅
Fusion experiments| ✅
Error analysis| ✅
Final model selection| ✅
Test prediction generation| ✅
Prediction validation| ✅
Reproducibility check| ✅
Unit tests| ✅
Documentation| ✅
Portability audit| ✅
Git-ready| ✅

Final Score

🎯 Recall@3: 89.37%

🏆 Selected Model: Document Expansion + Character TF-IDF

🔒 Evaluation: Leakage-safe GroupKFold

📦 Status: Ready for submission
