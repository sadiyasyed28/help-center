# Phase 2 — Baseline Retrieval

## Method
Two simple lexical retrieval baselines were implemented without any data cleaning, training query usage, or advanced techniques.
1. **v0_count (Shared Word Count)**: Scores articles by the number of unique normalized query tokens present in the article.
2. **v0_jaccard (Jaccard Similarity)**: Scores articles by the intersection size divided by the union size of query and article tokens.

## Results
| Method | Recall@1 | Recall@3 | MRR |
|---|---|---|---|
| v0_count | 0.3287 | 0.5648 | 0.4956 |
| v0_jaccard | 0.4444 | 0.6528 | 0.5839 |

## Comparison
The stronger baseline is **v0_jaccard** based on Recall@3 and MRR.

## Examples
### Successful Retrievals
- **Query**: `question: switch from test to live` (True: A123) - Retrieved at Rank 1
- **Query**: `how do i store customer card on file` (True: A110) - Retrieved at Rank 1
- **Query**: `hi, payout schedule timing please` (True: A101) - Retrieved at Rank 1

### Failed Retrievals
- **Query**: `why set trial lentgh to 14 days asap` (True: A121) - Ranked at 6
- **Query**: `how do i fight a chargeback?` (True: A105) - Ranked at 6
- **Query**: `hi, dunning settings for subscriptions` (True: A122) - Ranked at 22

## Limitations
Simple lexical matching (like count and Jaccard) struggles significantly because Phase 1 showed query-article vocabulary overlap is only ~32%. Queries often contain filler ('hi', 'please'), typos ('acocunt'), or different terminology ('1099 form' vs 'tax document') that simple token matching fails to capture.