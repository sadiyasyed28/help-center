# Phase 7 - Test Sanity Report

## 1. Number of predictions per article (Top-1)

- A113: 14 (14.6%)
- A110: 11 (11.5%)
- A108: 9 (9.4%)
- A106: 8 (8.3%)
- A123: 7 (7.3%)
- A114: 5 (5.2%)
- A102: 5 (5.2%)
- A103: 5 (5.2%)
- A119: 5 (5.2%)
- A121: 4 (4.2%)
- A101: 4 (4.2%)
- A107: 4 (4.2%)
- A115: 4 (4.2%)
- A109: 4 (4.2%)
- A124: 3 (3.1%)
- A117: 2 (2.1%)
- A118: 1 (1.0%)
- A104: 1 (1.0%)
- A105: 0 (0.0%)
- A111: 0 (0.0%)
- A116: 0 (0.0%)
- A112: 0 (0.0%)
- A120: 0 (0.0%)
- A122: 0 (0.0%)

## 2. Similarity Scores
- Average top-1 similarity: 0.2430
- Average top-3 similarity: 0.1835
- Min top-1 similarity: 0.1092
- Max top-1 similarity: 0.5508
- Number of very-low-similarity queries (< 0.1): 0

## 3. Query Length Distribution
**Test Queries:**
- Avg char length: 41.1
- Avg token length: 7.9

**Train Queries:**
- Avg char length: 39.7
- Avg token length: 7.5

## 4. Train vs Test Distribution Shift
Based on unlabeled query characteristics (character and token length), the test distribution appears very similar to the training distribution. The average lengths are comparable. Minimum top-1 similarities suggest no completely alien queries were observed.

## 5. Prediction Skew
We observe some concentration of top-1 predictions, but no single article dominates absolutely. Given real-world help centers usually have a long-tail distribution of queries focusing on a few key articles (like refunds or payments), this skew is expected and does not warrant automatic correction.