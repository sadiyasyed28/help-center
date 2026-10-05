# Help Center Search — Final Project Report

## 1. Problem Statement
The objective is to build a robust retrieval system for a Help Center search application. Given an unlabeled test query, the system must rank the correct article in the top 3 (target Recall@3 >= 60%). The corpus consists of 24 articles. We are provided with 216 labeled training queries and 96 unlabeled test queries. The primary challenge is generalization to unseen phrasings, noisy conversational language, filler words, and typos, without relying on massive external language models or pre-trained dense embeddings.

## 2. Dataset Summary
- **Number of articles:** 24
- **Number of training queries:** 216
- **Number of test queries:** 96
- **Queries per article (training):** 9
The training dataset is perfectly balanced, with exactly 9 labeled queries per article. Relevant columns include `query_id`, `query`, and `relevant_article_id` for the training data, and `article_id`, `title`, and `body` for the articles.

## 3. Phase 1 — Data Exploration
Initial analysis revealed:
- **Query length:** Train queries average ~40 characters and ~7.5 tokens.
- **Filler patterns:** High presence of conversational prefixes like 'how do i', 'can you', 'please'.
- **Unknown-token ratio & Typographical errors:** Numerous misspellings (e.g. 'acount', 'paymnt') and abbreviations not found in the formal article text.
- **Vocabulary overlap:** Many queries share minimal strict vocabulary overlap with the target article, relying on synonyms.
- **Difficult semantic areas:** Substantial topical overlap in areas like Payouts and Payments, making lexical discrimination challenging.

## 4. Phase 2 — Baselines
We implemented two ultra-simple, article-only baselines:
- **v0_count (Shared Word Count):** R@1=32.87%, R@3=66.20%, MRR=0.4727
- **v0_jaccard (Jaccard Similarity):** R@1=44.44%, R@3=73.61%, MRR=0.5694
Jaccard Similarity performed better because it normalizes for article length, preventing extremely long articles from dominating simply by having more words.

## 5. Phase 3 — Lexical Retrieval
We systematically improved the TF-IDF configuration on the article-only corpus:
| Version | Description | R@1 | R@3 | MRR |
|---|---|---|---|---|
| v1_tfidf | Model v1_tfidf | 53.70% | 70.37% | 0.6399 |
| v2_tfidf_bigrams | Model v2_tfidf_bigrams | 53.24% | 71.30% | 0.6374 |
| v3_title_weighted | Model v3_title_weighted | 49.07% | 68.52% | 0.6097 |
| v4_bm25 | Model v4_bm25 | 56.94% | 75.00% | 0.6833 |
| v5_bm25_cleaned | Model v5_bm25_cleaned | 59.72% | 75.46% | 0.7023 |
| v6_char_tfidf | Model v6_char_tfidf | 68.98% | 81.48% | 0.7678 |

**Why character n-grams (`v6_char_tfidf`) were effective:**
By splitting text into character n-grams (3-5 chars), the model inherently accommodates typos ('acount' matches 'account' highly), spelling variations, and morphological differences (e.g., 'paying' vs 'payment'). It bridges minor vocabulary gaps without requiring a formal stemmer or spellchecker.

## 6. Phase 4 — Learning From Training Queries
We explored Document Expansion (appending queries to articles) and Query-to-Query Retrieval. Crucially, we identified a **severe data leakage problem**: the training queries contain numerous near-duplicates and paraphrases. Random 5-fold cross-validation allowed identical concepts to leak across the train/validation split, artificially inflating scores.

To fix this, we conducted a near-duplicate audit and utilized **GroupKFold**, grouping similar queries so they strictly remain in the same fold. Group-aware CV provides a trustworthy metric.

- **v7 Random CV Recall@3:** 99.5349%
- **v7 Group-aware CV Recall@3:** 89.3658%

We clearly state: **v7_document_expansion achieved approximately 89.37% Group-aware Recall@3.**

## 7. Phase 5 — Fusion and Tuning
We attempted to fuse the expanded `v7` model with the base `v6` model and BM25 using raw-score normalization (alphas 0.25, 0.50, 0.75), and Rank Reciprocal Fusion (RRF).

Fusion was completely rejected because:
1. It did not meaningfully beat the v7 standalone model (best fusion tied at 89.37%).
2. The added architectural and hyperparameter complexity was not justified.
3. The simpler `v7` model is strictly preferred.

## 8. Final Model
The final selected model is **`v7_document_expansion`**.
1. We start with each article's title + body.
2. We append all labeled training queries corresponding to that article to form an expanded document.
3. We build a character TF-IDF representation (analyzer='char_wb').
4. We utilize n-gram range (3, 5).
5. We rank the 24 expanded articles by cosine similarity to the query.
6. During final fitting, we use all 216 labeled training queries to expand the articles.
7. Test queries are strictly ranked *after* the vectorizer is locked. They are never appended to articles.

*(Note: This is purely a lexical frequency model. It is not a semantic embedding model and does not natively understand synonyms in the neural/LLM sense.)*

## 9. Cross-Validation Methodology
GroupKFold was mandatory due to near-duplicate queries. An audit identified clusters of similar queries using TF-IDF and character similarity. Ultimately, the 216 queries were mapped to 140 distinct groups (including singletons). Validation queries and their grouped near-duplicates were kept entirely out of the training fold. Consequently, the Group-aware score is the primary, trustworthy evaluation metric.

## 10. Final Performance
Exact metrics for the locked final model (`v7_document_expansion` / `v7_reference`):
- **Group-aware Recall@1:** 78.2770%
- **Group-aware Recall@3:** 89.3658%
- **Group-aware MRR:** 0.8466
- **Recall@3 Std Dev:** 0.0346
- **Recall@3 Min/Max:** 0.8605 / 0.9302

This **~89.37% Recall@3** vastly outperforms the v0 Jaccard baseline (~73.61%), the v6 article-only character TF-IDF (~81.46%), and the v8 query-to-query method (~87.05%).

## 11. Error Analysis
In GroupKFold CV, there were exactly **23 Recall@3 misses**.
Failure categories:
- vocabulary_gap: 14 misses
- semantic_ambiguity: 8 misses
- same_category_confusion: 1 misses

**Three Representative Failures:**
1. **vocabulary_gap**: Query `stop fake orders with stolen cards` failed to retrieve A107. Reason: Query words {'stolen', 'with', 'orders', 'fake', 'stop', 'cards'} missing from true article but found in predicted.
2. **vocabulary_gap**: Query `can you tell me stop fake orders with stolen cards asap` failed to retrieve A107. Reason: Query words {'stolen', 'with', 'orders', 'fake', 'stop', 'me', 'cards', 'can', 'tell', 'asap'} missing from true article but found in predicted.
3. **semantic_ambiguity**: Query `need help: use a different checking account` failed to retrieve A103. Reason: Query is vague or could apply to multiple intents.

Errors are largely driven by strict vocabulary gaps and semantic ambiguities that lexical character n-grams cannot bridge.

## 12. Final Test Predictions
The final model was fit on the 24 articles expanded by all 216 training queries. The 96 test queries were then ranked against these 24 expanded articles, outputting the Top 3 to `predictions.csv` with columns `query_id`, `rank_1`, `rank_2`, `rank_3`. Validation scripts confirmed exact dimensional, referential, and uniqueness constraints.

## 13. Limitations
- **Lexical nature:** The model relies entirely on character n-gram overlaps. It has no external semantic embeddings or synonym data.
- **Vocabulary gaps:** Synonyms or entirely different phrasings for the same intent will still fail.
- **Ambiguity:** Articles in overlapping domains (e.g. Payments vs Payouts) remain difficult to distinguish based purely on frequency counts.
- **No Test Labels:** CV is a proxy for test performance. We do not claim an 89.37% test accuracy, as the true test labels are unknown locally.

## 14. Reproducibility
To reproduce this pipeline locally from the experiment root (`.`):
```bash
# 1. Ensure dependencies
.venv\Scripts\pip.exe install -r requirements.txt
# 2. Execute pipeline sequentially (if regenerating)
.venv\Scripts\python.exe src/phase4_learning.py
.venv\Scripts\python.exe src/phase5_fusion.py
.venv\Scripts\python.exe src/phase6_error_analysis.py
# 3. Generate Final Predictions
.venv\Scripts\python.exe src/phase7_final_predictions.py
# 4. Validate Predictions
.venv\Scripts\python.exe src/validate_predictions.py
# 5. Run Unit Tests
.venv\Scripts\python.exe -m unittest discover tests
```

## 15. Project Conclusion
Document Expansion combined with Character TF-IDF (v7) proved to be an elegant, robust, and highly effective solution for Help Center retrieval. By rigorously addressing data leakage through GroupKFold cross-validation, we ensured that the reported ~89.37% Recall@3 is a trustworthy estimate of generalization capability. The final model requires no external dependencies or massive LLM deployments, perfectly balancing high accuracy with operational simplicity.