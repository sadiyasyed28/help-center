import numpy as np
import pandas as pd

def recall_at_k(ranked_ids, true_id, k=3):
    """
    1 if true_id occurs within the first k ranked IDs.
    0 otherwise.
    """
    if not isinstance(ranked_ids, list):
        return 0
    if k <= 0:
        return 0
    
    # Consider only up to rank k
    top_k = ranked_ids[:k]
    
    if true_id in top_k:
        return 1
    return 0

def mrr(ranked_ids, true_id):
    """
    1 / rank of true_id.
    If true_id is absent: 0
    """
    if not isinstance(ranked_ids, list):
        return 0
    
    try:
        rank = ranked_ids.index(true_id) + 1
        return 1.0 / rank
    except ValueError:
        return 0.0

def evaluate(search_fn, queries_df):
    """
    search_fn(query) -> list of article_ids
    queries_df expects columns: 'query', 'article_id' (true label)
    """
    results = []
    
    for _, row in queries_df.iterrows():
        query_text = row['query']
        true_id = row.get('relevant_article_id', row.get('article_id'))
        
        ranked_ids = search_fn(query_text)
        
        rec_at_1 = recall_at_k(ranked_ids, true_id, k=1)
        rec_at_3 = recall_at_k(ranked_ids, true_id, k=3)
        mrr_score = mrr(ranked_ids, true_id)
        
        results.append({
            'query': query_text,
            'true_id': true_id,
            'ranked_ids': ranked_ids,
            'recall@1': rec_at_1,
            'recall@3': rec_at_3,
            'mrr': mrr_score
        })
        
    results_df = pd.DataFrame(results)
    
    agg_recall_1 = results_df['recall@1'].mean() if not results_df.empty else 0.0
    agg_recall_3 = results_df['recall@3'].mean() if not results_df.empty else 0.0
    agg_mrr = results_df['mrr'].mean() if not results_df.empty else 0.0
    
    return agg_recall_3, agg_mrr, agg_recall_1, results_df
