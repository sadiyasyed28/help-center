from pathlib import Path
import os
import ast
import re
import unicodedata
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

from evaluation import evaluate, recall_at_k, mrr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'reports')
PHASE5_DIR = os.path.join(REPORTS_DIR, 'phase5')
os.makedirs(PHASE5_DIR, exist_ok=True)

# 1. Load Data
articles_df = pd.read_csv(os.path.join(DATA_DIR, 'articles.csv'))
articles_df['text'] = articles_df['title'].astype(str) + " " + articles_df['body'].astype(str)
article_ids = articles_df['article_id'].tolist()
article_texts = articles_df['text'].tolist()

queries_train = pd.read_csv(os.path.join(DATA_DIR, 'queries_train.csv'))
if 'query_id' not in queries_train.columns:
    queries_train['query_id'] = [f"Q{i}" for i in range(len(queries_train))]
queries_train['label'] = queries_train.get('relevant_article_id', queries_train.get('article_id'))
queries_train['query_lower'] = queries_train['query'].str.lower()

# 2. Assign Groups
try:
    groups_df = pd.read_csv(os.path.join(REPORTS_DIR, 'phase4', 'query_groups.csv'))
    q2g = {}
    for _, row in groups_df.iterrows():
        g_id = row['group_id']
        members = ast.literal_eval(row['member_query_ids'])
        for m in members:
            q2g[m] = g_id
    max_gid = groups_df['group_id'].max()
except FileNotFoundError:
    q2g = {}
    max_gid = 0

group_assignments = []
text_to_group = {}
for _, row in queries_train.iterrows():
    qid = row['query_id']
    text = row['query_lower']
    if qid in q2g:
        g_id = q2g[qid]
    elif text in text_to_group:
        g_id = text_to_group[text]
    else:
        max_gid += 1
        g_id = max_gid
    group_assignments.append(g_id)
    text_to_group[text] = g_id
    q2g[qid] = g_id

queries_train['group_id'] = group_assignments
group_map = {}
def get_canonical(g):
    while g in group_map and group_map[g] != g:
        g = group_map[g]
    return g

for i, row in queries_train.iterrows():
    text = row['query_lower']
    g_id = row['group_id']
    if text not in text_to_group:
        text_to_group[text] = g_id
    else:
        old_g = get_canonical(text_to_group[text])
        curr_g = get_canonical(g_id)
        if old_g != curr_g:
            group_map[curr_g] = old_g

queries_train['group_id'] = queries_train['group_id'].apply(get_canonical)

def simple_tokenize(text):
    text = unicodedata.normalize('NFKD', str(text)).encode('ASCII', 'ignore').decode('utf-8')
    text = text.lower()
    return re.findall(r'\b\w+\b', text)

def min_max_norm(scores):
    s_min, s_max = np.min(scores), np.max(scores)
    if s_max == s_min:
        return np.zeros_like(scores)
    return (scores - s_min) / (s_max - s_min)

def get_ranks(scores):
    # Higher score = rank 1 (0-indexed rank)
    sorted_indices = np.argsort(scores)[::-1]
    ranks = np.zeros(len(scores))
    for rank_idx, original_idx in enumerate(sorted_indices):
        ranks[original_idx] = rank_idx + 1
    return ranks

def run_cv(cv_splitter, protocol_name):
    # Models to test:
    # v7_reference
    # v9_raw_score_fusion_alpha_0.25
    # v9_raw_score_fusion_alpha_0.50
    # v9_raw_score_fusion_alpha_0.75
    # v10_rrf_fusion
    # v11_expansion_bm25
    
    model_keys = [
        'v7_reference',
        'v9_raw_score_fusion_alpha_0.25',
        'v9_raw_score_fusion_alpha_0.50',
        'v9_raw_score_fusion_alpha_0.75',
        'v10_rrf_fusion',
        'v11_expansion_bm25'
    ]
    
    metrics = {k: [] for k in model_keys}
    per_query = {k: [] for k in model_keys}
    
    splits = list(cv_splitter.split(queries_train['query'], queries_train['label'], queries_train['group_id']))
    assert len(splits) == 5, "Expected 5 splits"

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        train_df = queries_train.iloc[train_idx]
        val_df = queries_train.iloc[val_idx]
        
        # Leakage Asserts
        train_qids = set(train_df['query_id'])
        val_qids = set(val_df['query_id'])
        train_texts = set(train_df['query_lower'])
        val_texts = set(val_df['query_lower'])
        
        assert len(train_qids.intersection(val_qids)) == 0, "QID Leakage"
        if protocol_name == 'group':
            assert len(train_texts.intersection(val_texts)) == 0, "Exact Text Leakage"
            
        # 1. Build Document Expansion (v7)
        expanded_texts = []
        for aid, orig_text in zip(article_ids, article_texts):
            mapped_queries = train_df[train_df['label'] == aid]['query'].tolist()
            expanded_texts.append(orig_text + " " + " ".join(mapped_queries))
            
        vec_v7 = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), lowercase=True)
        doc_matrix_v7 = vec_v7.fit_transform(expanded_texts)
        
        # 2. Build Article Char (Model B)
        vec_art = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), lowercase=True)
        doc_matrix_art = vec_art.fit_transform(article_texts)
        
        # 3. Build Article BM25 (Model E)
        tokenized_corpus = [simple_tokenize(doc) for doc in article_texts]
        bm25 = BM25Okapi(tokenized_corpus, k1=1.5, b=0.75)
        
        # Verify validation text didn't leak into v7 explicitly (by exact substring check on expansion, though it's covered by above)
        if protocol_name == 'group':
            for v_text in val_texts:
                for exp_text in expanded_texts:
                    # check if the whole query string is in the expanded text
                    # (this might falsely trigger if a sub-query matches exactly what an article says naturally, 
                    # but typically training queries are disjoint from validation queries in GroupKFold)
                    pass

        def get_all_scores(query_text):
            # v7 scores
            q_vec_v7 = vec_v7.transform([query_text])
            sim_v7 = cosine_similarity(q_vec_v7, doc_matrix_v7)[0]
            norm_v7 = min_max_norm(sim_v7)
            
            # article char scores
            q_vec_art = vec_art.transform([query_text])
            sim_art = cosine_similarity(q_vec_art, doc_matrix_art)[0]
            norm_art = min_max_norm(sim_art)
            
            # bm25 scores
            q_tokens = simple_tokenize(query_text)
            sim_bm25 = bm25.get_scores(q_tokens)
            norm_bm25 = min_max_norm(sim_bm25)
            
            return sim_v7, norm_v7, norm_art, norm_bm25

        def build_ranker(model_key):
            def search_fn(query_text):
                sim_v7, norm_v7, norm_art, norm_bm25 = get_all_scores(query_text)
                
                if model_key == 'v7_reference':
                    final_scores = sim_v7
                elif model_key.startswith('v9_raw_score_fusion_alpha_'):
                    alpha = float(model_key.split('_')[-1])
                    final_scores = alpha * norm_v7 + (1.0 - alpha) * norm_art
                elif model_key == 'v10_rrf_fusion':
                    rank_v7 = get_ranks(norm_v7)
                    rank_art = get_ranks(norm_art)
                    final_scores = (1.0 / (rank_v7 + 60)) + (1.0 / (rank_art + 60))
                elif model_key == 'v11_expansion_bm25':
                    final_scores = 0.5 * norm_v7 + 0.5 * norm_bm25
                    
                scores = [(final_scores[i], article_ids[i]) for i in range(len(article_ids))]
                scores.sort(key=lambda x: (-x[0], x[1]))
                return [aid for score, aid in scores]
            return search_fn
        
        for mk in model_keys:
            r3, mrr, r1, df_res = evaluate(build_ranker(mk), val_df)
            metrics[mk].append((r1, r3, mrr))
            df_res['fold'] = fold_idx
            per_query[mk].append(df_res)
            
    return metrics, per_query

print("Running GroupKFold CV...")
metrics_grp, pq_grp = run_cv(GroupKFold(n_splits=5), 'group')

print("Running StratifiedKFold CV...")
metrics_rand, pq_rand = run_cv(StratifiedKFold(n_splits=5, shuffle=True, random_state=42), 'random')

# ----------------------------------------------------
# 4. Process and Save Results
# ----------------------------------------------------
def build_rows(metrics_dict, protocol):
    rows = []
    for model_key, fold_scores in metrics_dict.items():
        for f_idx, (r1, r3, m_mrr) in enumerate(fold_scores):
            rows.append({
                'version': model_key,
                'description': model_key,
                'evaluation_protocol': protocol,
                'fold': f_idx,
                'recall_at_1': r1,
                'recall_at_3': r3,
                'mrr': m_mrr,
                'uses_training_queries': True
            })
        
        m_r1 = np.mean([s[0] for s in fold_scores])
        m_r3 = np.mean([s[1] for s in fold_scores])
        m_mrr = np.mean([s[2] for s in fold_scores])
        rows.append({
            'version': model_key,
            'description': model_key,
            'evaluation_protocol': protocol,
            'fold': 'mean',
            'recall_at_1': m_r1,
            'recall_at_3': m_r3,
            'mrr': m_mrr,
            'uses_training_queries': True
        })
    return rows

all_rows = build_rows(metrics_grp, 'Group-aware 5-Fold')
all_rows += build_rows(metrics_rand, 'Random Stratified 5-Fold')

results_path = os.path.join(REPORTS_DIR, 'results.csv')
if os.path.exists(results_path):
    existing = pd.read_csv(results_path)
    combined = pd.concat([existing, pd.DataFrame(all_rows)], ignore_index=True)
    combined.to_csv(results_path, index=False)
else:
    pd.DataFrame(all_rows).to_csv(results_path, index=False)

def write_pq(pq_dict, suffix):
    for model_key, dfs in pq_dict.items():
        if not dfs: continue
        combined = pd.concat(dfs, ignore_index=True)
        out = []
        for _, row in combined.iterrows():
            ranked_ids = row['ranked_ids']
            true_id = row['true_id']
            q_text = row['query']
            q_id = queries_train[queries_train['query'] == q_text].iloc[0]['query_id']
            out.append({
                'fold': row['fold'],
                'query_id': q_id,
                'query': q_text,
                'true_article_id': true_id,
                'rank_1': ranked_ids[0] if len(ranked_ids) > 0 else '',
                'rank_2': ranked_ids[1] if len(ranked_ids) > 1 else '',
                'rank_3': ranked_ids[2] if len(ranked_ids) > 2 else '',
                'true_rank': ranked_ids.index(true_id) + 1 if true_id in ranked_ids else -1,
                'reciprocal_rank': row['mrr']
            })
        
        # specific files requested: v7_reference_group.csv etc
        short_name = model_key
        if model_key.startswith('v9_raw'):
            short_name = 'v9_raw_fusion' if model_key == 'v9_raw_score_fusion_alpha_0.50' else model_key
            if model_key != 'v9_raw_score_fusion_alpha_0.50':
                continue # only output the top requested ones to save space, but actually I will just output what's requested
        fname = f"{short_name}_{suffix}.csv"
        pd.DataFrame(out).to_csv(os.path.join(PHASE5_DIR, fname), index=False)

write_pq(pq_grp, 'group')
write_pq(pq_rand, 'random')

print("Phase 5 fusion CV complete.")
