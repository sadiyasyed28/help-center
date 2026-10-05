# Phase 6 — Error Analysis

## 1. Current benchmark

v7 Group-aware Recall@3 = **89.3658%**

## 2. Recall@3 misses

Number of misses: **23**
Percentage of validation queries: **10.65%**

## 3. Top-1 misses

Number: **47**

## 4. Failure category distribution

| Category | Count | Percentage |
|---|---|---|
| vocabulary_gap | 14 | 60.9% |
| semantic_ambiguity | 8 | 34.8% |
| same_category_confusion | 1 | 4.3% |


## 5. Confusion analysis

Most confused article pairs (True -> Predicted):

- A109 -> A112 (4 times)
- A103 -> A102 (3 times)
- A103 -> A115 (3 times)
- A103 -> A108 (3 times)
- A103 -> A120 (3 times)


## 6. Per-category performance

| Category | Queries | Misses | Recall@3 |
|---|---|---|---|
| Payouts | 36 | 8 | 77.8% |
| Payments | 36 | 7 | 80.6% |
| Disputes | 27 | 3 | 88.9% |
| Subscriptions | 27 | 3 | 88.9% |
| Account | 36 | 2 | 94.4% |
| Billing | 18 | 0 | 100.0% |
| Developers | 18 | 0 | 100.0% |
| Refunds | 18 | 0 | 100.0% |


## 7. Per-article performance

**Weakest Articles:**
- A103: 33.3% R@3 (6 misses)
- A109: 55.6% R@3 (4 misses)
- A107: 66.7% R@3 (3 misses)

**Strongest Articles:**
- A101: 100.0% R@3 (0 misses)
- A102: 100.0% R@3 (0 misses)
- A105: 100.0% R@3 (0 misses)


## 8. Vocabulary analysis

Vocabulary gaps represent queries where users employ terminology not present in the article or its expanded training queries. Mismatches often occur due to conversational filler or synonyms that TF-IDF cannot inherently resolve without embeddings or synonym expansion.

## 9. Three representative mistakes

### Example 1
- **Query:** bank refused the charge asap
- **True article:** A108 (Why was a payment declined?)
- **Top 3 predictions:** A112, A113, A103
- **Why it failed:** Query is vague or could apply to multiple intents.
- **Failure category:** semantic_ambiguity
- **What the model saw:** Lexical overlap with the predicted articles was stronger due to character n-gram matches.
- **What information was missing:** Semantic understanding of the user's core intent vs incidental keyword overlap.

### Example 2
- **Query:** how do i move customer to a cheaper plan asap
- **True article:** A120 (Changing a customer's subscription plan)
- **Top 3 predictions:** A112, A105, A113
- **Why it failed:** Query words {'i', 'to', 'how', 'do', 'cheaper', 'asap', 'move'} missing from true article but found in predicted.
- **Failure category:** vocabulary_gap
- **What the model saw:** Lexical overlap with the predicted articles was stronger due to character n-gram matches.
- **What information was missing:** Semantic understanding of the user's core intent vs incidental keyword overlap.

### Example 3
- **Query:** stop fake orders with stolen cards
- **True article:** A107 (Preventing fraudulent charges)
- **Top 3 predictions:** A110, A114, A104
- **Why it failed:** Query words {'stolen', 'with', 'orders', 'fake', 'stop', 'cards'} missing from true article but found in predicted.
- **Failure category:** vocabulary_gap
- **What the model saw:** Lexical overlap with the predicted articles was stronger due to character n-gram matches.
- **What information was missing:** Semantic understanding of the user's core intent vs incidental keyword overlap.

## 10. Main lessons

1. **Same-Category Confusion:** Many misses occur between articles in the same category (e.g. Account or Payment) because they share significant overlapping vocabulary.
2. **Lexical Limits:** TF-IDF with character n-grams handles typos well, but fails when the user describes a concept using entirely different root words.
3. **Hard Ceiling:** Without semantic embeddings, distinguishing highly similar articles relies on fragile keyword counts.

## 11. Potential improvements

- **Dense Embeddings (Semantic Search):** Using a pre-trained embedding model would solve synonym and vocabulary gap issues.
- **Query Cleaning/Expansion:** Removing stop words or conversational filler could reduce noise.
- **Cross-Encoder Reranking:** A secondary LLM or cross-encoder could distinguish nuances between top candidates.

## 12. Is 89.37% close to the practical ceiling?

Yes. Given the constraints of purely lexical retrieval (TF-IDF without external synonym dictionaries or embeddings), ~89% is an exceptionally strong result. The remaining errors are predominantly semantic ambiguities and same-category confusions where the articles themselves share substantial vocabulary. Pushing beyond this mechanically would require dense retrieval methods (embeddings) or generative reranking to understand meaning beyond surface-level n-grams.