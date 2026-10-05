# Phase 7 — Final Fit and Predictions

## 1. Locked model

The final locked model is `v7_document_expansion`. It is a lexical retrieval model utilizing a single `TfidfVectorizer` configured with `analyzer="char_wb"`, `ngram_range=(3,5)`, and `lowercase=True`. Documents are scored using cosine similarity against the query.

## 2. Training data

- **24 articles:** Used as the base retrieval corpus.
- **216 labeled queries:** Used to expand the base articles.

## 3. Final fitting procedure

The final model was fit by appending all 216 labeled training queries to their respective true articles. Specifically, for each of the 24 articles, the final expanded document consists of: `[Article Title] + [Article Body] + [All associated training queries]`. The TF-IDF vectorizer was fit exclusively on these 24 expanded documents.

## 4. Test set

The test set consists of **96 unlabeled queries** from `queries_test.csv`.

## 5. Prediction procedure

The 96 test queries were transformed using the already-fitted character TF-IDF vectorizer. We calculated the cosine similarity between each test query and all 24 expanded article documents. For each test query, candidates were sorted descending by similarity. Exact ties were broken deterministically using ascending `article_id`. The top 3 distinct articles were selected as the final predictions.

## 6. Validation

A robust validation script (`src/validate_predictions.py`) was executed on the output file. It successfully asserted:
1. `predictions.csv` exists.
2. It contains exactly 96 rows.
3. Column names are exactly `query_id`, `rank_1`, `rank_2`, `rank_3`.
4. Every `query_id` appears exactly once.
5. Every predicted `query_id` exists in the original `queries_test.csv`.
6. No test queries are missing.
7. No extra `query_id`s were generated.
8. Every rank contains a valid `article_id` from the 24-article corpus.
9. `rank_1`, `rank_2`, and `rank_3` are distinct for every single row.
10. There are no null values.
11. There are no duplicate rows.
12. There are exactly three predictions per query.
13. Predictions are fully deterministic.

## 7. Test distribution

**Prediction Distribution (Top-1):**
- A113: 14.6%
- A110: 11.5%
- A108: 9.4%
- (6 articles received 0 top-1 predictions, which is expected due to the long-tail nature of support queries).

**Similarity Diagnostics:**
- Average top-1 similarity: 0.2430
- Average top-3 similarity: 0.1835
- Min top-1 similarity: 0.1092
- Max top-1 similarity: 0.5508
- Number of very-low-similarity queries (< 0.1): 0

## 8. Distribution shift

Based purely on unlabeled characteristics, the test queries appear very similar to the training queries.
- **Test avg length:** 41.1 chars (7.9 tokens)
- **Train avg length:** 39.7 chars (7.5 tokens)

There is no evidence of a substantial distribution shift. Furthermore, the minimum top-1 similarity score of 0.1092 indicates that there were no entirely alien queries that the model failed to map to the vocabulary.

## 9. Reproducibility

The final prediction script was run twice consecutively. A programmatic equality check confirmed that the two resulting `predictions.csv` files were identical row-for-row, proving the pipeline is completely deterministic.

## 10. Final artifact

The final artifact has been saved exactly as requested:
`.\predictions.csv`

It contains:
- 96 rows
- 4 columns (`query_id`, `rank_1`, `rank_2`, `rank_3`)
- Exactly 3 distinct predictions per query.
