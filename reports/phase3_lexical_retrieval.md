# Phase 3 — Classic Lexical Retrieval

## Baseline
Reference v0 results on the 216-query training set:
- **v0_count**: Recall@1 = 0.3287, Recall@3 = 0.5648, MRR = 0.4956
- **v0_jaccard**: Recall@1 = 0.4444, Recall@3 = 0.6528, MRR = 0.5839

## v1 — TF-IDF
**Method**: Classic TF-IDF using Scikit-Learn `TfidfVectorizer` (lowercase, english stop words, unigrams). Fit purely on the 24 articles.
**Results**: Recall@1 = 0.5370, Recall@3 = 0.7037, MRR = 0.6399
**Change from baseline**: Significant improvement over Jaccard. Recall@3 increased by ~0.05.

## v2 — Bigrams
**Method**: TF-IDF expanded to use `ngram_range=(1,2)`.
**Results**: Recall@1 = 0.5324, Recall@3 = 0.7130, MRR = 0.6374
**Change from v1**: Minor improvement in Recall@3 (+0.0093), but a tiny decrease in Recall@1 and MRR. The inclusion of bigrams was largely neutral to slightly positive.

## v3 — Title weighting
**Method**: Split article text into Title and Body matrices. Applied a linear weighting `w * title_sim + 1.0 * body_sim`. Tested $w \in [0.5, 0.7, 1.0]$.
**Selected weight**: 0.5 (or 0.7, both achieved the same highest Recall@3 of 0.6852). We recorded the best diagnostic run.
**Results**: Recall@1 = 0.4907, Recall@3 = 0.6852, MRR = 0.6097
**Change from v2**: Noticeable drop in performance. Simple static title weighting failed to outperform the baseline TF-IDF. This suggests that for this dataset, queries often match the body rather than the title explicitly.

## v4 — BM25
**Method**: Standard BM25 (Okapi) ranking using $k1=1.5$ and $b=0.75$ with basic word tokenization.
**Results**: Recall@1 = 0.5694, Recall@3 = 0.7500, MRR = 0.6833
**Change from v3**: Substantial improvement. Recall@3 jumped to 75%, and Recall@1 improved by over 0.07. BM25 proves much stronger than TF-IDF here.

## v5 — Query cleaning
**Method**: Deterministic removal of filler (e.g. "hi", "how do i", "asap", "please") from the query before running BM25.
**Cleaning rules**: See `reports/phase3_cleaning_rules.md`
**Before/after comparison**: BM25 R@3=0.7500 vs Cleaned BM25 R@3=0.7546.
**Results**: Recall@1 = 0.5972, Recall@3 = 0.7546, MRR = 0.7023
**Whether cleaning helped**: Yes. Removing the conversational filler yielded a steady boost across all metrics (R@1 improved by ~0.03).

## v6 — Character n-grams
**Method**: Character-level TF-IDF using `analyzer="char_wb"`, `ngram_range=(3,5)`, and `lowercase=True`. 
**Results**: Recall@1 = 0.6898, Recall@3 = 0.8148, MRR = 0.7678
**Change from v5**: Massive improvement. R@3 crossed 80%, and R@1 improved by nearly 0.10.

## Overall comparison table

| Version | Description | Uses Training Queries | R@1 | R@3 | MRR |
|---------|-------------|-----------------------|-----|-----|-----|
| v0_count | Shared Word Count | False | 0.3287 | 0.5648 | 0.4956 |
| v0_jaccard | Jaccard Similarity | False | 0.4444 | 0.6528 | 0.5839 |
| v1_tfidf | TF-IDF (unigrams, stop words) | False | 0.5370 | 0.7037 | 0.6399 |
| v2_tfidf_bigrams | TF-IDF (bigrams, stop words) | False | 0.5324 | 0.7130 | 0.6374 |
| v3_title_weighted | Title weighted TF-IDF (0.5) | False | 0.4907 | 0.6852 | 0.6097 |
| v4_bm25 | BM25 default params | False | 0.5694 | 0.7500 | 0.6833 |
| v5_bm25_cleaned | BM25 with query filler cleaning | False | 0.5972 | 0.7546 | 0.7023 |
| v6_char_tfidf | Char n-grams (3-5) TF-IDF | False | 0.6898 | 0.8148 | 0.7678 |

## What worked
- **BM25 Over TF-IDF**: BM25's saturation function handled document length differences and term frequencies far better than plain TF-IDF.
- **Query Cleaning**: Stripping conversational noise ("hi", "asap") directly reduced false positive matches and brought the query vocabulary closer to the article text.
- **Character N-grams**: This was the absolute standout feature. By operating on 3-5 character substrings, this method natively bypassed typos, inflection differences (e.g. "account" vs "accounts"), and spelling variants that crippled word-level BM25/TF-IDF.

## What did not work
- **Title Weighting**: Hard-coding a boost to article titles resulted in degraded performance. Users often query specific symptoms/processes that are buried in the article body rather than summarizing the title.

## Interpretation
Phase 1 revealed a dismal 32.1% average token overlap between a query and its correct article, largely due to a 40.3% unknown vocabulary ratio driven by conversational filler, synonyms, and typos. 
- Traditional word-based models (TF-IDF, BM25) struggled because they demand exact word matches.
- Query cleaning (v5) addressed the filler portion of the vocabulary mismatch, yielding moderate gains.
- Character n-grams (v6) addressed the typo and morphological mismatch issues natively, driving the biggest leap in performance. When a user types "acocunt" or "subscripsions," character n-grams still capture high similarity to the correct terms, bridging the remaining lexical gap that full-word models miss.
