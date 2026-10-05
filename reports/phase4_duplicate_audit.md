# Phase 4 Pre-Audit: Duplicate and Paraphrase Leakage

## 1. Query Similarity Audit Summary
We computed pairwise similarity across all 216 training queries using Character N-gram (3-5) TF-IDF. This method was chosen because it naturally captures typographic errors, morphological variations, and overlap in core phrasing, ensuring robust detection of near-duplicates regardless of typos.

## 2. Near-Duplicate Thresholds & Classifications
Based on the pairwise similarities, we found the following distribution of high-similarity pairs:
- **>= 0.95 similarity**: 13 pairs
- **>= 0.90 similarity**: 29 pairs
- **>= 0.85 similarity**: 47 pairs
- **>= 0.80 similarity**: 44 pairs

*(Note: These are cumulative counts of distinct pairs crossing the threshold).*

### Classification Examples:
**A. Exact Duplicate (sim = 1.00)**
- Q1056: `can you tell me cancel everything and leave`
- Q1184: `can you tell me cancel everything and leave`

**B. Punctuation-only / trivial variation (sim >= 0.95)**
- Q1004: `hi, dunning settings for subscriptions`
- Q1029: `dunning settings for subscriptions`
- Q1113: `help fake card numbers for testing?`
- Q1118: `fake card numbers for testing?`

**C. Near-duplicate wording (sim >= 0.85)**
- Q1024: `need help: put a hold on a card`
- Q1125: `need help: put a hold on a card asap`

**D. Strong paraphrase with typos (sim ~ 0.60 - 0.75)**
- Q1032: `why return a charge to the customer`
- Q1190: `how do i return a charge to the cusotmer` (sim=0.61)

**E. Merely related query (sim < 0.65, distinct intent)**
- Q1037: `question: charged a fee for a chargeback` (Article A106)
- Q1206: `question: how do i fight a chargeback for my shop` (Article A105) (sim=0.64)

## 3. Article-Label Distribution
**Critical finding:** Among all pairs with a similarity score $\ge 0.80$, **100%** of them share the exact same target article. 
The only pairs that share high lexical overlap but map to *different* articles occurred below a similarity score of 0.65 (e.g., questions about *fighting* a chargeback vs *fees* for a chargeback).

## 4. Proposed Groups
Using a graph connected-components approach with a highly conservative edge threshold of **0.85 similarity**, we identified:
- **49 distinct groups** containing a total of **123 queries**.
These 123 queries represent the "danger zone" of data leakage. Because they all share the exact same target article as their near-duplicate peers, they must be forced into the same fold during any cross-validation split.

## 5. Random CV Risk
If we apply standard, random 5-fold cross-validation without grouping these queries, we introduce massive data leakage into our evaluation metrics.

**A. Document Expansion:**
If we train a model to expand the original article text using the training queries mapped to it, random CV will allow the model to ingest "fake card numbers for testing?" into the article representation during the training phase. During the validation phase, the model will be evaluated on the near-duplicate "help fake card numbers for testing?". The evaluation will show a high success rate, but it is a false positive driven by memorization of the validation set's phrasing.

**B. Query-to-Query Retrieval:**
A k-NN or semantic query-to-query retriever compares a test query against all known training queries. If exact duplicates (like Q1056 and Q1184: "can you tell me cancel everything and leave") are split across folds, the model will achieve a perfect similarity score during validation by simply matching the exact string it saw in training. It will appear to have "solved" the problem, but in reality, it is just overfitting to the leaked overlapping data. 

## Recommendation for Phase 4
We must abandon random 5-fold cross-validation. Instead, we must implement **GroupKFold** (or a similar group-aware splitting strategy), using the proposed 49 groups (and assigning unique group IDs to the remaining isolated queries) to ensure that no near-duplicate pairs ever cross the train/validation boundary.
