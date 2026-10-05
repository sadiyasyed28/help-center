# Final Experiment Summary

**Problem:** Rank exactly 24 help center articles for highly noisy, conversational user queries. We had 216 training queries and 96 test queries.

**Methodology & Leakage:** Standard random CV was discarded due to near-duplicate query leakage. A rigorous GroupKFold evaluation protocol was implemented to group similar queries, ensuring trustworthy evaluation metrics.

**Best Model:** `v7_document_expansion` (Character TF-IDF on articles expanded by training queries).

**Final CV Result:** ~89.37% Group-aware Recall@3.

**Why v7 won:** It comfortably outperformed base lexical models and query-to-query methods. Complex fusion pipelines (Phase 5) were rejected as they failed to meaningfully improve over v7, favoring v7's operational simplicity.

**Major Errors:** The remaining 10.6% error rate is heavily driven by fundamental vocabulary gaps (missing synonyms) and same-category semantic ambiguity (e.g. Payments vs Payouts).

**Final Prediction Status:** The locked v7 model was fit on the fully expanded 24 articles (using all 216 training queries), and successfully generated exactly 96 Top-3 rankings in the strictly validated `predictions.csv`.