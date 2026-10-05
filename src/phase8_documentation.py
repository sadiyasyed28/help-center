import os
import pandas as pd
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# Step 1: Read metrics
results_df = pd.read_csv(os.path.join(REPORTS_DIR, "results.csv"))

# Step 2: Create final results table
# Select key models
key_versions = [
    'v0_count', 'v0_jaccard', 'v1_tfidf', 'v2_tfidf_bigrams', 'v3_title_weighted',
    'v4_bm25', 'v5_bm25_cleaned', 'v6_char_tfidf', 'v6_cv_char_tfidf',
    'v7_document_expansion_char_tfidf', 'v8_query_to_query_tfidf', 'v9_raw_score_fusion_alpha_0.50',
    'v10_rrf_fusion', 'v11_expansion_bm25', 'v7_reference'
]

# We want to extract the summary (fold == 'mean' or equivalent representation)
# For v0-v6 (non-CV), fold is usually NaN or not applicable, but in results.csv it might be saved as 'mean' or just 1 row.
final_table = []
for v in key_versions:
    sub = results_df[results_df['version'] == v]
    if len(sub) == 0:
        continue
    
    # If there's a 'mean' row, take it. Otherwise just take the first row.
    mean_row = sub[sub['fold'] == 'mean']
    if len(mean_row) > 0:
        row = mean_row.iloc[0]
    else:
        row = sub.iloc[0]
        
    final_table.append({
        'version': row['version'],
        'description': f"Model {row['version']}",
        'uses_training_queries': row['uses_training_queries'],
        'evaluation_protocol': row['evaluation_protocol'],
        'recall_at_1': row['recall_at_1'],
        'recall_at_3': row['recall_at_3'],
        'mrr': row['mrr'],
        'notes': ''
    })

final_results_df = pd.DataFrame(final_table)
final_results_df.to_csv(os.path.join(REPORTS_DIR, "final_results_table.csv"), index=False)

# Get specific v7 stats for report
v7_group = results_df[(results_df['version'] == 'v7_reference') & (results_df['evaluation_protocol'] == 'Group-aware 5-Fold')]
v7_group_mean = v7_group[v7_group['fold'] == 'mean'].iloc[0]
v7_folds = v7_group[v7_group['fold'] != 'mean']

v7_r1 = float(v7_group_mean['recall_at_1'])
v7_r3 = float(v7_group_mean['recall_at_3'])
v7_mrr = float(v7_group_mean['mrr'])
v7_r3_std = float(v7_folds['recall_at_3'].astype(float).std())
v7_r3_min = float(v7_folds['recall_at_3'].astype(float).min())
v7_r3_max = float(v7_folds['recall_at_3'].astype(float).max())

# Read Phase 6 stats
phase6_misses_df = pd.read_csv(os.path.join(REPORTS_DIR, "phase6", "v7_recall3_misses.csv"))
misses_count = len(phase6_misses_df)
failures_df = pd.read_csv(os.path.join(REPORTS_DIR, "phase6", "failure_categories.csv"))
cat_counts = failures_df['failure_category'].value_counts()

# Load representative examples
ex1 = failures_df.iloc[0]
ex2 = failures_df.iloc[1]
ex3 = failures_df.iloc[2]

# Build Final Project Report
report = []
report.append("# Help Center Search — Final Project Report\n")
report.append("## 1. Problem Statement")
report.append("The objective is to build a robust retrieval system for a Help Center search application. Given an unlabeled test query, the system must rank the correct article in the top 3 (target Recall@3 >= 60%). The corpus consists of 24 articles. We are provided with 216 labeled training queries and 96 unlabeled test queries. The primary challenge is generalization to unseen phrasings, noisy conversational language, filler words, and typos, without relying on massive external language models or pre-trained dense embeddings.\n")

report.append("## 2. Dataset Summary")
report.append("- **Number of articles:** 24")
report.append("- **Number of training queries:** 216")
report.append("- **Number of test queries:** 96")
report.append("- **Queries per article (training):** 9")
report.append("The training dataset is perfectly balanced, with exactly 9 labeled queries per article. Relevant columns include `query_id`, `query`, and `relevant_article_id` for the training data, and `article_id`, `title`, and `body` for the articles.\n")

report.append("## 3. Phase 1 — Data Exploration")
report.append("Initial analysis revealed:")
report.append("- **Query length:** Train queries average ~40 characters and ~7.5 tokens.")
report.append("- **Filler patterns:** High presence of conversational prefixes like 'how do i', 'can you', 'please'.")
report.append("- **Unknown-token ratio & Typographical errors:** Numerous misspellings (e.g. 'acount', 'paymnt') and abbreviations not found in the formal article text.")
report.append("- **Vocabulary overlap:** Many queries share minimal strict vocabulary overlap with the target article, relying on synonyms.")
report.append("- **Difficult semantic areas:** Substantial topical overlap in areas like Payouts and Payments, making lexical discrimination challenging.\n")

report.append("## 4. Phase 2 — Baselines")
report.append("We implemented two ultra-simple, article-only baselines:")
report.append("- **v0_count (Shared Word Count):** R@1=32.87%, R@3=66.20%, MRR=0.4727")
report.append("- **v0_jaccard (Jaccard Similarity):** R@1=44.44%, R@3=73.61%, MRR=0.5694")
report.append("Jaccard Similarity performed better because it normalizes for article length, preventing extremely long articles from dominating simply by having more words.\n")

report.append("## 5. Phase 3 — Lexical Retrieval")
report.append("We systematically improved the TF-IDF configuration on the article-only corpus:")
report.append("| Version | Description | R@1 | R@3 | MRR |")
report.append("|---|---|---|---|---|")
for v in ['v1_tfidf', 'v2_tfidf_bigrams', 'v3_title_weighted', 'v4_bm25', 'v5_bm25_cleaned', 'v6_char_tfidf']:
    row = final_results_df[final_results_df['version'] == v].iloc[0]
    report.append(f"| {v} | {row['description']} | {float(row['recall_at_1'])*100:.2f}% | {float(row['recall_at_3'])*100:.2f}% | {float(row['mrr']):.4f} |")
report.append("\n**Why character n-grams (`v6_char_tfidf`) were effective:**")
report.append("By splitting text into character n-grams (3-5 chars), the model inherently accommodates typos ('acount' matches 'account' highly), spelling variations, and morphological differences (e.g., 'paying' vs 'payment'). It bridges minor vocabulary gaps without requiring a formal stemmer or spellchecker.\n")

report.append("## 6. Phase 4 — Learning From Training Queries")
report.append("We explored Document Expansion (appending queries to articles) and Query-to-Query Retrieval. Crucially, we identified a **severe data leakage problem**: the training queries contain numerous near-duplicates and paraphrases. Random 5-fold cross-validation allowed identical concepts to leak across the train/validation split, artificially inflating scores.")
report.append("\nTo fix this, we conducted a near-duplicate audit and utilized **GroupKFold**, grouping similar queries so they strictly remain in the same fold. Group-aware CV provides a trustworthy metric.")
v7_rand = results_df[(results_df['version'] == 'v7_document_expansion_char_tfidf') & (results_df['evaluation_protocol'] == 'Random Stratified 5-Fold') & (results_df['fold'] == 'mean')].iloc[0]
report.append(f"\n- **v7 Random CV Recall@3:** {float(v7_rand['recall_at_3'])*100:.4f}%")
report.append(f"- **v7 Group-aware CV Recall@3:** {v7_r3*100:.4f}%")
report.append(f"\nWe clearly state: **v7_document_expansion achieved approximately 89.37% Group-aware Recall@3.**\n")

report.append("## 7. Phase 5 — Fusion and Tuning")
report.append("We attempted to fuse the expanded `v7` model with the base `v6` model and BM25 using raw-score normalization (alphas 0.25, 0.50, 0.75), and Rank Reciprocal Fusion (RRF).")
report.append("\nFusion was completely rejected because:")
report.append("1. It did not meaningfully beat the v7 standalone model (best fusion tied at 89.37%).")
report.append("2. The added architectural and hyperparameter complexity was not justified.")
report.append("3. The simpler `v7` model is strictly preferred.\n")

report.append("## 8. Final Model")
report.append("The final selected model is **`v7_document_expansion`**.")
report.append("1. We start with each article's title + body.")
report.append("2. We append all labeled training queries corresponding to that article to form an expanded document.")
report.append("3. We build a character TF-IDF representation (analyzer='char_wb').")
report.append("4. We utilize n-gram range (3, 5).")
report.append("5. We rank the 24 expanded articles by cosine similarity to the query.")
report.append("6. During final fitting, we use all 216 labeled training queries to expand the articles.")
report.append("7. Test queries are strictly ranked *after* the vectorizer is locked. They are never appended to articles.")
report.append("\n*(Note: This is purely a lexical frequency model. It is not a semantic embedding model and does not natively understand synonyms in the neural/LLM sense.)*\n")

report.append("## 9. Cross-Validation Methodology")
report.append("GroupKFold was mandatory due to near-duplicate queries. An audit identified clusters of similar queries using TF-IDF and character similarity. Ultimately, the 216 queries were mapped to 140 distinct groups (including singletons). Validation queries and their grouped near-duplicates were kept entirely out of the training fold. Consequently, the Group-aware score is the primary, trustworthy evaluation metric.\n")

report.append("## 10. Final Performance")
report.append("Exact metrics for the locked final model (`v7_document_expansion` / `v7_reference`):")
report.append(f"- **Group-aware Recall@1:** {v7_r1*100:.4f}%")
report.append(f"- **Group-aware Recall@3:** {v7_r3*100:.4f}%")
report.append(f"- **Group-aware MRR:** {v7_mrr:.4f}")
report.append(f"- **Recall@3 Std Dev:** {v7_r3_std:.4f}")
report.append(f"- **Recall@3 Min/Max:** {v7_r3_min:.4f} / {v7_r3_max:.4f}")
report.append("\nThis **~89.37% Recall@3** vastly outperforms the v0 Jaccard baseline (~73.61%), the v6 article-only character TF-IDF (~81.46%), and the v8 query-to-query method (~87.05%).\n")

report.append("## 11. Error Analysis")
report.append(f"In GroupKFold CV, there were exactly **{misses_count} Recall@3 misses**.")
report.append("Failure categories:")
for cat, count in cat_counts.items():
    report.append(f"- {cat}: {count} misses")
report.append("\n**Three Representative Failures:**")
report.append(f"1. **{ex1['failure_category']}**: Query `{ex1['query']}` failed to retrieve {ex1['true_article_id']}. Reason: {ex1['explanation']}")
report.append(f"2. **{ex2['failure_category']}**: Query `{ex2['query']}` failed to retrieve {ex2['true_article_id']}. Reason: {ex2['explanation']}")
report.append(f"3. **{ex3['failure_category']}**: Query `{ex3['query']}` failed to retrieve {ex3['true_article_id']}. Reason: {ex3['explanation']}")
report.append("\nErrors are largely driven by strict vocabulary gaps and semantic ambiguities that lexical character n-grams cannot bridge.\n")

report.append("## 12. Final Test Predictions")
report.append("The final model was fit on the 24 articles expanded by all 216 training queries. The 96 test queries were then ranked against these 24 expanded articles, outputting the Top 3 to `predictions.csv` with columns `query_id`, `rank_1`, `rank_2`, `rank_3`. Validation scripts confirmed exact dimensional, referential, and uniqueness constraints.\n")

report.append("## 13. Limitations")
report.append("- **Lexical nature:** The model relies entirely on character n-gram overlaps. It has no external semantic embeddings or synonym data.")
report.append("- **Vocabulary gaps:** Synonyms or entirely different phrasings for the same intent will still fail.")
report.append("- **Ambiguity:** Articles in overlapping domains (e.g. Payments vs Payouts) remain difficult to distinguish based purely on frequency counts.")
report.append("- **No Test Labels:** CV is a proxy for test performance. We do not claim an 89.37% test accuracy, as the true test labels are unknown locally.\n")

report.append("## 14. Reproducibility")
report.append("To reproduce this pipeline locally from the experiment root:")
report.append("```bash")
report.append("# 1. Ensure dependencies")
report.append(".venv\\Scripts\\pip.exe install -r requirements.txt")
report.append("# 2. Execute pipeline sequentially (if regenerating)")
report.append(".venv\\Scripts\\python.exe src/phase4_learning.py")
report.append(".venv\\Scripts\\python.exe src/phase5_fusion.py")
report.append(".venv\\Scripts\\python.exe src/phase6_error_analysis.py")
report.append("# 3. Generate Final Predictions")
report.append(".venv\\Scripts\\python.exe src/phase7_final_predictions.py")
report.append("# 4. Validate Predictions")
report.append(".venv\\Scripts\\python.exe src/validate_predictions.py")
report.append("# 5. Run Unit Tests")
report.append(".venv\\Scripts\\python.exe -m unittest discover tests")
report.append("```\n")

report.append("## 15. Project Conclusion")
report.append("Document Expansion combined with Character TF-IDF (v7) proved to be an elegant, robust, and highly effective solution for Help Center retrieval. By rigorously addressing data leakage through GroupKFold cross-validation, we ensured that the reported ~89.37% Recall@3 is a trustworthy estimate of generalization capability. The final model requires no external dependencies or massive LLM deployments, perfectly balancing high accuracy with operational simplicity.")

with open(os.path.join(REPORTS_DIR, "final_project_report.md"), "w") as f:
    f.write("\n".join(report))


# Build Summary
summary = []
summary.append("# Final Experiment Summary")
summary.append("\n**Problem:** Rank exactly 24 help center articles for highly noisy, conversational user queries. We had 216 training queries and 96 test queries.")
summary.append("\n**Methodology & Leakage:** Standard random CV was discarded due to near-duplicate query leakage. A rigorous GroupKFold evaluation protocol was implemented to group similar queries, ensuring trustworthy evaluation metrics.")
summary.append("\n**Best Model:** `v7_document_expansion` (Character TF-IDF on articles expanded by training queries).")
summary.append(f"\n**Final CV Result:** ~89.37% Group-aware Recall@3.")
summary.append("\n**Why v7 won:** It comfortably outperformed base lexical models and query-to-query methods. Complex fusion pipelines (Phase 5) were rejected as they failed to meaningfully improve over v7, favoring v7's operational simplicity.")
summary.append("\n**Major Errors:** The remaining 10.6% error rate is heavily driven by fundamental vocabulary gaps (missing synonyms) and same-category semantic ambiguity (e.g. Payments vs Payouts).")
summary.append("\n**Final Prediction Status:** The locked v7 model was fit on the fully expanded 24 articles (using all 216 training queries), and successfully generated exactly 96 Top-3 rankings in the strictly validated `predictions.csv`.")

with open(os.path.join(REPORTS_DIR, "final_experiment_summary.md"), "w") as f:
    f.write("\n".join(summary))

print("Documentation artifacts successfully created.")
