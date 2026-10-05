from pathlib import Path
import os
import ast
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

from evaluation import evaluate, recall_at_k, mrr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'reports')
PHASE4_DIR = os.path.join(REPORTS_DIR, 'phase4')
os.makedirs(PHASE4_DIR, exist_ok=True)

# 1. Load Data
articles_df = pd.read_csv(os.path.join(DATA_DIR, 'articles.csv'))
articles_df['text'] = articles_df['title'].astype(str) + " " + articles_df['body'].astype(str)
article_ids = articles_df['article_id'].tolist()

queries_train = pd.read_csv(os.path.join(DATA_DIR, 'queries_train.csv'))
if 'query_id' not in queries_train.columns:
    queries_train['query_id'] = [f"Q{i}" for i in range(len(queries_train))]
queries_train['label'] = queries_train.get('relevant_article_id', queries_train.get('article_id'))

# 2. Assign Groups
try:
    groups_df = pd.read_csv(os.path.join(PHASE4_DIR, 'query_groups.csv'))
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
    text = row['query'].lower()
    
    if qid in q2g:
        g_id = q2g[qid]
    elif text in text_to_group:
        g_id = text_to_group[text]
    else:
        max_gid += 1
        g_id = max_gid
        
    group_assignments.append(g_id)
    text_to_group[text] = g_id
    # Ensure this qid is also registered in q2g so if it is queried again it matches
    q2g[qid] = g_id

queries_train['group_id'] = group_assignments

# We must do one more pass to merge any groups that share the same exact text
# because Q1056 and Q1184 might have different group IDs assigned from q2g!
# Let's fix this properly.
group_map = {}
def get_canonical_group(g):
    while g in group_map and group_map[g] != g:
        g = group_map[g]
    return g

for i, row in queries_train.iterrows():
    text = row['query'].lower()
    g_id = row['group_id']
    if text not in text_to_group:
        text_to_group[text] = g_id
    else:
        # they share the same text, they must share the same group
        old_g = get_canonical_group(text_to_group[text])
        curr_g = get_canonical_group(g_id)
        if old_g != curr_g:
            group_map[curr_g] = old_g

queries_train['group_id'] = queries_train['group_id'].apply(get_canonical_group)


# 3. CV Setup
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
gkf = GroupKFold(n_splits=5)

results_summary = []
all_per_query_results = {}

def get_v6_baseline_ranker(train_idx):
    # Pure article baseline
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), lowercase=True)
    doc_matrix = vec.fit_transform(articles_df['text'].tolist())
    def search_fn(query_text):
        q_vec = vec.transform([query_text])
        sims = cosine_similarity(q_vec, doc_matrix)[0]
        scores = [(sims[i], article_ids[i]) for i in range(len(article_ids))]
        scores.sort(key=lambda x: (-x[0], x[1]))
        return [aid for score, aid in scores]
    return search_fn

def get_v7_doc_exp_ranker(train_idx):
    # Document expansion
    train_queries = queries_train.iloc[train_idx]
    
    expanded_texts = []
    for aid in article_ids:
        orig = articles_df[articles_df['article_id'] == aid]['text'].values[0]
        mapped_queries = train_queries[train_queries['label'] == aid]['query'].tolist()
        expanded_texts.append(orig + " " + " ".join(mapped_queries))
        
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), lowercase=True)
    doc_matrix = vec.fit_transform(expanded_texts)
    
    def search_fn(query_text):
        q_vec = vec.transform([query_text])
        sims = cosine_similarity(q_vec, doc_matrix)[0]
        scores = [(sims[i], article_ids[i]) for i in range(len(article_ids))]
        scores.sort(key=lambda x: (-x[0], x[1]))
        return [aid for score, aid in scores]
    return search_fn

def get_v8_q2q_rankers(train_idx):
    train_queries = queries_train.iloc[train_idx]
    train_texts = train_queries['query'].tolist()
    train_labels = train_queries['label'].tolist()
    
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), lowercase=True)
    doc_matrix = vec.fit_transform(train_texts)
    
    def build_q2q_search(k=None):
        def search_fn(query_text):
            q_vec = vec.transform([query_text])
            sims = cosine_similarity(q_vec, doc_matrix)[0]
            
            # score articles based on queries
            article_scores = {aid: 0.0 for aid in article_ids}
            
            # rank training queries by sim
            query_scores = [(sims[i], train_labels[i]) for i in range(len(train_labels))]
            query_scores.sort(key=lambda x: -x[0])
            
            top_q = query_scores if k is None else query_scores[:k]
            
            for sim, label in top_q:
                if sim > 0:
                    article_scores[label] += sim
                    
            scores = [(article_scores[aid], aid) for aid in article_ids]
            scores.sort(key=lambda x: (-x[0], x[1]))
            return [aid for score, aid in scores]
        return search_fn
    
    return {
        'all': build_q2q_search(k=None),
        'top-3': build_q2q_search(k=3),
        'top-5': build_q2q_search(k=5),
        'top-10': build_q2q_search(k=10)
    }

def get_v8b_centroid_ranker(train_idx):
    train_queries = queries_train.iloc[train_idx]
    
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), lowercase=True)
    vec.fit(train_queries['query'])
    
    centroid_matrix = []
    for aid in article_ids:
        mapped_queries = train_queries[train_queries['label'] == aid]['query'].tolist()
        if mapped_queries:
            q_vecs = vec.transform(mapped_queries)
            centroid = np.asarray(q_vecs.mean(axis=0)).flatten()
        else:
            # fallback to empty if no queries for this article in train (should be rare)
            centroid = np.zeros(len(vec.vocabulary_))
        centroid_matrix.append(centroid)
        
    centroid_matrix = np.array(centroid_matrix)
    
    def search_fn(query_text):
        q_vec = vec.transform([query_text])
        sims = cosine_similarity(q_vec, centroid_matrix)[0]
        scores = [(sims[i], article_ids[i]) for i in range(len(article_ids))]
        scores.sort(key=lambda x: (-x[0], x[1]))
        return [aid for score, aid in scores]
    return search_fn

def check_leakage(train_idx, val_idx, protocol_name):
    train_qids = set(queries_train.iloc[train_idx]['query_id'])
    val_qids = set(queries_train.iloc[val_idx]['query_id'])
    train_texts = set(queries_train.iloc[train_idx]['query'].str.lower())
    val_texts = set(queries_train.iloc[val_idx]['query'].str.lower())
    
    assert len(train_qids.intersection(val_qids)) == 0, "Query ID leakage!"
    if protocol_name == 'group':
        assert len(train_texts.intersection(val_texts)) == 0, "Query text exact leakage!"
    else:
        if len(train_texts.intersection(val_texts)) > 0:
            print("Warning: Query text exact leakage detected in Random CV (expected).")

def run_cv(cv_splitter, protocol_name):
    # We will accumulate results across folds
    metrics = {
        'v6': [],
        'v7': [],
        'v8_all': [], 'v8_top3': [], 'v8_top5': [], 'v8_top10': [],
        'v8b': []
    }
    
    # Also collect per-query rankings
    per_query = {k: [] for k in metrics.keys()}
    
    fold_dists = []

    splits = list(cv_splitter.split(queries_train['query'], queries_train['label'], queries_train['group_id']))
    assert len(splits) == 5, f"Expected 5 splits, got {len(splits)}"

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        check_leakage(train_idx, val_idx, protocol_name)
        
        val_df = queries_train.iloc[val_idx]
        
        # Log distribution
        dist = val_df['label'].value_counts().to_dict()
        for aid, count in dist.items():
            fold_dists.append({'fold': fold_idx, 'article_id': aid, 'validation_query_count': count})
            
        # 1. v6 baseline
        ranker_v6 = get_v6_baseline_ranker(train_idx)
        r3, mrr, r1, df_v6 = evaluate(ranker_v6, val_df)
        metrics['v6'].append((r1, r3, mrr))
        df_v6['fold'] = fold_idx
        per_query['v6'].append(df_v6)
        
        # 2. v7 doc expansion
        ranker_v7 = get_v7_doc_exp_ranker(train_idx)
        r3, mrr, r1, df_v7 = evaluate(ranker_v7, val_df)
        metrics['v7'].append((r1, r3, mrr))
        df_v7['fold'] = fold_idx
        per_query['v7'].append(df_v7)
        
        # 3. v8 query-to-query
        q2q_rankers = get_v8_q2q_rankers(train_idx)
        for sub_k, ranker_fn in q2q_rankers.items():
            r3, mrr, r1, df_v8 = evaluate(ranker_fn, val_df)
            key = f"v8_{sub_k.replace('-','')}"
            metrics[key].append((r1, r3, mrr))
            df_v8['fold'] = fold_idx
            per_query[key].append(df_v8)
            
        # 4. v8b centroid
        ranker_v8b = get_v8b_centroid_ranker(train_idx)
        r3, mrr, r1, df_v8b = evaluate(ranker_v8b, val_df)
        metrics['v8b'].append((r1, r3, mrr))
        df_v8b['fold'] = fold_idx
        per_query['v8b'].append(df_v8b)

    return metrics, per_query, fold_dists

# Run CVs
print("Running StratifiedKFold CV...")
metrics_rand, pq_rand, dist_rand = run_cv(skf, 'random')
print("Running GroupKFold CV...")
metrics_grp, pq_grp, dist_grp = run_cv(gkf, 'group')

# Save Distributions
pd.DataFrame(dist_grp).to_csv(os.path.join(PHASE4_DIR, 'group_fold_distribution.csv'), index=False)

def build_results_rows(metrics_dict, protocol, base_name_map):
    rows = []
    for model_key, fold_scores in metrics_dict.items():
        model_name = base_name_map[model_key]
        for f_idx, (r1, r3, mrr) in enumerate(fold_scores):
            rows.append({
                'version': model_name,
                'description': model_name,
                'evaluation_protocol': protocol,
                'fold': f_idx,
                'recall_at_1': r1,
                'recall_at_3': r3,
                'mrr': mrr,
                'uses_training_queries': model_key != 'v6'
            })
        
        # Mean row
        m_r1 = np.mean([s[0] for s in fold_scores])
        m_r3 = np.mean([s[1] for s in fold_scores])
        m_mrr = np.mean([s[2] for s in fold_scores])
        rows.append({
            'version': model_name,
            'description': model_name,
            'evaluation_protocol': protocol,
            'fold': 'mean',
            'recall_at_1': m_r1,
            'recall_at_3': m_r3,
            'mrr': m_mrr,
            'uses_training_queries': model_key != 'v6'
        })
    return rows

name_map = {
    'v6': 'v6_cv_char_tfidf',
    'v7': 'v7_document_expansion_char_tfidf',
    'v8_all': 'v8_query_to_query_all',
    'v8_top3': 'v8_query_to_query_top3',
    'v8_top5': 'v8_query_to_query_top5',
    'v8_top10': 'v8_query_to_query_top10',
    'v8b': 'v8b_query_centroid'
}

all_rows = build_results_rows(metrics_rand, 'Random Stratified 5-Fold', name_map)
all_rows += build_results_rows(metrics_grp, 'Group-aware 5-Fold', name_map)

# Append to results.csv
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
        # reformat
        out = []
        for _, row in combined.iterrows():
            ranked_ids = row['ranked_ids']
            true_id = row['true_id']
            # Fetch the actual query_id if missing, query_id was added to train
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
        fname = f"{name_map[model_key]}_{suffix}.csv"
        # The prompt asks for v8_query_to_query_random.csv without the topK suffix if we select one, but we are writing all out. 
        pd.DataFrame(out).to_csv(os.path.join(PHASE4_DIR, fname), index=False)

write_pq(pq_rand, 'random')
write_pq(pq_grp, 'group')

print("Phase 4 CV complete.")
