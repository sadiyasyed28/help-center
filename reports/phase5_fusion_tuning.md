# Phase 5 — Fusion and Light Tuning

## Starting benchmark
Our starting benchmark is `v7_document_expansion` (from Phase 4, reproduced here as `v7_reference`), which achieved a robust group-aware Recall@3 of **89.3658%**.
The purely lexical article-only baseline (`v6_cv_char_tfidf`) sits at **81.4588%**.

## Fusion methodology
We tested fusing the strong `v7` document expansion signal with the original `v6` article signal and BM25. Before combining, raw scores were normalized per-query using min-max scaling to project them into the [0, 1] range. We also tested Rank Reciprocal Fusion (RRF) using the standard `1 / (rank + 60)` formula.

## Raw-score fusion
- **Alpha = 0.25** (Expansion weight): Recall@3 = **85.1586%** (MRR: 0.8051)
- **Alpha = 0.50** (Expansion weight): Recall@3 = **89.3658%** (MRR: 0.8525)
- **Alpha = 0.75** (Expansion weight): Recall@3 = **88.9006%** (MRR: 0.8688)

## RRF fusion
**v10_rrf_fusion** (Equal-weight RRF): Recall@3 = **88.4355%** (MRR: 0.8074)

## BM25 fusion
**v11_expansion_bm25** (Equal-weight raw score): Recall@3 = **88.9112%** (MRR: 0.8212)

## Fold stability (Group-aware CV)
| Model | R@1 Mean | R@1 Std | R@3 Mean | R@3 Std | R@3 Min | R@3 Max | MRR Mean | MRR Std |
|---|---|---|---|---|---|---|---|---|
| v7_reference | 0.7828 | 0.0649 | 0.8937 | 0.0346 | 0.8605 | 0.9302 | 0.8466 | 0.0433 |
| v9_raw_score_fusion_alpha_0.25 | 0.7271 | 0.0563 | 0.8516 | 0.0589 | 0.7674 | 0.9091 | 0.8051 | 0.0297 |
| v9_raw_score_fusion_alpha_0.50 | 0.7916 | 0.0494 | 0.8937 | 0.0579 | 0.8372 | 0.9767 | 0.8525 | 0.0273 |
| v9_raw_score_fusion_alpha_0.75 | 0.8196 | 0.0681 | 0.8890 | 0.0411 | 0.8372 | 0.9302 | 0.8688 | 0.0379 |
| v10_rrf_fusion | 0.7177 | 0.0597 | 0.8844 | 0.0431 | 0.8372 | 0.9535 | 0.8074 | 0.0336 |
| v11_expansion_bm25 | 0.7363 | 0.0335 | 0.8891 | 0.0737 | 0.7907 | 0.9767 | 0.8212 | 0.0368 |

## Random vs Group CV
| Model | Random R@3 | Group R@3 | Diff |
|---|---|---|---|
| v7_reference | 99.5349% | 89.3658% | 10.1691% |
| v9_raw_score_fusion_alpha_0.50 | 97.6956% | 89.3658% | 8.3298% |
| v10_rrf_fusion | 89.8203% | 88.4355% | 1.3848% |
| v11_expansion_bm25 | 96.3002% | 88.9112% | 7.3890% |

## Final selection
The selected model is **v7_reference**.
- **Group-aware Recall@3:** 89.3658%
- **Group-aware MRR:** 0.8466
- **Improvement over v7:** +0.0000% absolute (+0.00% relative)
- **Stability:** The selected model demonstrated a tight standard deviation, proving robustness.

Fusion combinations provided little to no substantial improvement over the standalone document expansion (`v7`) approach. Because the improvement threshold (0.01) was not convincingly broken by the more complex models, `v7_reference` (or the equivalent best simple model) is preferred for its operational simplicity.

## Overfitting discussion
A tiny dataset of only 216 training queries is highly susceptible to hyperparameter overfitting. A large grid search over alpha weights or bm25 parameters would have yielded an artificially inflated combination that memorizes this specific validation set. We intentionally restricted the search space to a few discrete pre-declared states (e.g. 0.25, 0.50, 0.75, and equal-weight RRF) to ensure any observed gain is a true structural improvement rather than noise fitting.