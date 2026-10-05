# Phase 4 Final Verification

## 1. Exact Phase 4 Results (Mean Folds)
```
v6_cv_char_tfidf | Random Stratified 5-Fold | R@1: 0.690169 | R@3: 0.814905 | MRR: 0.768026 | Uses Train: False
v7_document_expansion_char_tfidf | Random Stratified 5-Fold | R@1: 0.935307 | R@3: 0.995349 | MRR: 0.964708 | Uses Train: True
v8_query_to_query_all | Random Stratified 5-Fold | R@1: 0.879704 | R@3: 0.990803 | MRR: 0.935227 | Uses Train: True
v8_query_to_query_top3 | Random Stratified 5-Fold | R@1: 0.939746 | R@3: 1.000000 | MRR: 0.969873 | Uses Train: True
v8_query_to_query_top5 | Random Stratified 5-Fold | R@1: 0.897992 | R@3: 1.000000 | MRR: 0.948996 | Uses Train: True
v8_query_to_query_top10 | Random Stratified 5-Fold | R@1: 0.865539 | R@3: 0.995349 | MRR: 0.931607 | Uses Train: True
v6_cv_char_tfidf | Group-aware 5-Fold | R@1: 0.689958 | R@3: 0.814588 | MRR: 0.767801 | Uses Train: False
v7_document_expansion_char_tfidf | Group-aware 5-Fold | R@1: 0.782770 | R@3: 0.893658 | MRR: 0.846578 | Uses Train: True
v8_query_to_query_all | Group-aware 5-Fold | R@1: 0.453277 | R@3: 0.592812 | MRR: 0.556094 | Uses Train: True
v8_query_to_query_top3 | Group-aware 5-Fold | R@1: 0.550951 | R@3: 0.648414 | MRR: 0.628716 | Uses Train: True
v8_query_to_query_top5 | Group-aware 5-Fold | R@1: 0.495349 | R@3: 0.662368 | MRR: 0.604016 | Uses Train: True
v8_query_to_query_top10 | Group-aware 5-Fold | R@1: 0.490486 | R@3: 0.643869 | MRR: 0.596751 | Uses Train: True
```

## 2. Fold Results Validation
```
Model: v6_cv_char_tfidf | Protocol: Random Stratified 5-Fold
  R@1 Mean: 0.690169 Std: 0.103048
  R@3 Mean: 0.814905 Std: 0.078542
  MRR Mean: 0.768026 Std: 0.074067

Model: v6_cv_char_tfidf | Protocol: Group-aware 5-Fold
  R@1 Mean: 0.689958 Std: 0.057204
  R@3 Mean: 0.814588 Std: 0.055230
  MRR Mean: 0.767801 Std: 0.034968

Model: v7_document_expansion_char_tfidf | Protocol: Random Stratified 5-Fold
  R@1 Mean: 0.935307 Std: 0.044413
  R@3 Mean: 0.995349 Std: 0.010400
  MRR Mean: 0.964708 Std: 0.023753

Model: v7_document_expansion_char_tfidf | Protocol: Group-aware 5-Fold
  R@1 Mean: 0.782770 Std: 0.064890
  R@3 Mean: 0.893658 Std: 0.034559
  MRR Mean: 0.846578 Std: 0.043277

Model: v8_query_to_query_top5 | Protocol: Random Stratified 5-Fold
  R@1 Mean: 0.897992 Std: 0.051209
  R@3 Mean: 1.000000 Std: 0.000000
  MRR Mean: 0.948996 Std: 0.025604

Model: v8_query_to_query_top5 | Protocol: Group-aware 5-Fold
  R@1 Mean: 0.495349 Std: 0.034494
  R@3 Mean: 0.662368 Std: 0.068350
  MRR Mean: 0.604016 Std: 0.041340

```

## 3 & 4. Leakage Validation (v7 & v8)
```
V7 Document Expansion Leakage Checks Passed: True
V8 Query-to-Query Leakage Checks Passed: True
```

## 5. Group Integrity
```
Number of queries: 216
Number of groups: 140
Queries per group mean: 1.54
Queries per group median: 1.00
Largest group size: 3
Smallest group size: 1
Group-Train/Validation intersection is logically prevented by GroupKFold.
```

## 6. Group Class Distribution
```
Fold 0: 44 queries, 18 unique articles
Fold 1: 43 queries, 16 unique articles
Fold 2: 43 queries, 15 unique articles
Fold 3: 43 queries, 17 unique articles
Fold 4: 43 queries, 18 unique articles

article_id  fold_0_count  fold_1_count  fold_2_count  fold_3_count  fold_4_count
      A101             1             1             7             0             0
      A102             3             0             2             1             3
      A103             3             4             0             2             0
      A104             0             6             0             0             3
      A105             2             0             4             0             3
      A106             1             4             1             1             2
      A107             4             0             0             0             5
      A108             3             1             0             4             1
      A109             3             1             1             0             4
      A110             0             3             4             1             1
      A111             3             2             0             3             1
      A112             2             0             4             3             0
      A113             1             1             3             4             0
      A114             3             1             1             2             2
      A115             3             0             0             3             3
      A116             2             3             0             3             1
      A117             0             0             6             3             0
      A118             2             1             1             1             4
      A119             0             4             0             2             3
      A120             0             0             3             4             2
      A121             2             7             0             0             0
      A122             0             1             2             4             2
      A123             5             0             3             0             1
      A124             1             3             1             2             2
```

## 7. v7 vs v6 Improvement
```
v6 Group R@3: 0.814588
v7 Group R@3: 0.893658
Absolute Improvement: 0.079070
Relative Improvement: 9.71%
```

## 8. Random vs Group Gap
```
Model: v7_document_expansion_char_tfidf
Random Mean R@3: 0.995349
Group Mean R@3: 0.893658
Absolute Gap: 0.101691
Relative Gap: 11.38%

Model: v8_query_to_query_top5
Random Mean R@3: 1.000000
Group Mean R@3: 0.662368
Absolute Gap: 0.337632
Relative Gap: 50.97%

Explanation: The massive gap demonstrates data leakage. The Random CV incorrectly included near-duplicate variants of the validation queries in the training set. The Group CV prevents this, revealing the true, non-inflated performance on unseen variations.
```

## 9. Check the 89.3% Claim
```
YES
Exact Value: 0.893658
```

## Conclusion
Phase 4 is fully verified. Document expansion provides a proven, leakage-free ~89.3% Recall@3 which is an absolute gain over the ~81.4% article-only baseline.