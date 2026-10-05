# Phase 4: Learning from Training Queries

## The Cost of Data Leakage

By running two distinct cross-validation protocols, we have empirically proven the severe danger of data leakage caused by near-duplicate queries.

**Under the naive Random Stratified 5-Fold CV**, models that learned from training queries appeared to achieve near-perfect performance. Because near-duplicate queries (e.g., "can you tell me cancel everything and leave") were randomly split across folds, the models were essentially tested on identical strings they had just seen in training.
- **Random CV Document Expansion:** ~99.5% Recall@3
- **Random CV Query-to-Query:** 100% Recall@3

However, when we applied **Group-aware 5-Fold CV** (which forces all near-duplicate variations into the same fold to simulate truly unseen queries), the "perfect" scores collapsed, revealing the true generalization capability of the models. The baseline `v6` article-only performance remained stable across both protocols (~81.4% Recall@3) since it does not train on queries at all. 

## Document Expansion (v7)

Document Expansion concatenates the text of all associated training queries (within the training fold) directly onto the corresponding article text before generating TF-IDF representations. 

**Group CV Results for `v7_document_expansion_char_tfidf`:**
- **Recall@3:** 0.8936

This approach provided a substantial and realistic improvement over the `v6` article-only baseline (which scored 0.8145 Recall@3). By enriching the article text with the vocabulary, phrasing, and common typos present in the training queries, the TF-IDF matrix becomes much more robust at catching unseen variations of similar intents.

## Query-to-Query Retrieval (v8)

Query-to-Query Retrieval attempts to bypass the article text entirely. It computes the similarity between the validation query and all indexed training queries, aggregating the scores of the most similar training queries to recommend their parent articles.

**Group CV Results for Query-to-Query (Top 5):**
- **Recall@3:** 0.6623

**Group CV Results for Query Centroid:**
- **Recall@3:** 0.6765

Unlike the artificial 100% score seen in Random CV, Query-to-Query performed **poorly** in realistic conditions—even failing to beat the 81.4% article-only baseline. 
The reason is vocabulary mismatch: if a test query uses slightly different terminology that wasn't captured in the training queries, it has no article text to fall back on. Queries are extremely short, so the semantic overlap is brittle without the rich context of the underlying article body.

## Conclusion

Data leakage masks catastrophic brittleness. Without proper Group-aware evaluation, we would have incorrectly selected Query-to-Query retrieval and expected perfect accuracy in production, only to see it fail spectacularly on new user queries.

**Document Expansion (`v7`)** is the clear winner. It successfully learns from historical queries while maintaining the safety net of the full article text, raising our realistic Recall@3 to **~89.3%**. We will carry Document Expansion forward as our primary retrieval technique.
