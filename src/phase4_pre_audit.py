from pathlib import Path
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'reports')
PHASE4_DIR = os.path.join(REPORTS_DIR, 'phase4')
os.makedirs(PHASE4_DIR, exist_ok=True)

# Load data
queries_train = pd.read_csv(os.path.join(DATA_DIR, 'queries_train.csv'))

# Use character n-grams to catch typo-based near-duplicates
vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), lowercase=True)
doc_matrix = vectorizer.fit_transform(queries_train['query'])

sim_matrix = cosine_similarity(doc_matrix)

# 1. QUERY SIMILARITY AUDIT
nearest_neighbors = []
pairs = []

n = len(queries_train)
for i in range(n):
    # zero out self-similarity
    sim_matrix[i, i] = -1.0
    
    best_idx = np.argmax(sim_matrix[i])
    best_sim = sim_matrix[i, best_idx]
    
    nearest_neighbors.append({
        'query_id': queries_train.iloc[i].get('query_id', i),
        'query': queries_train.iloc[i]['query'],
        'nearest_query_id': queries_train.iloc[best_idx].get('query_id', best_idx),
        'nearest_query': queries_train.iloc[best_idx]['query'],
        'similarity': best_sim
    })
    
    # Store all unique pairs with sim >= 0.5 for detailed analysis
    for j in range(i + 1, n):
        if sim_matrix[i, j] >= 0.5:
            q1_label = queries_train.iloc[i].get('relevant_article_id', queries_train.iloc[i].get('article_id'))
            q2_label = queries_train.iloc[j].get('relevant_article_id', queries_train.iloc[j].get('article_id'))
            pairs.append({
                'query_id_1': queries_train.iloc[i].get('query_id', i),
                'query_1': queries_train.iloc[i]['query'],
                'query_id_2': queries_train.iloc[j].get('query_id', j),
                'query_2': queries_train.iloc[j]['query'],
                'similarity': sim_matrix[i, j],
                'article_1': q1_label,
                'article_2': q2_label,
                'same_article': (q1_label == q2_label)
            })

nn_df = pd.DataFrame(nearest_neighbors).sort_values('similarity', ascending=False)
nn_df.to_csv(os.path.join(PHASE4_DIR, 'query_nearest_neighbors.csv'), index=False)

pairs_df = pd.DataFrame(pairs).sort_values('similarity', ascending=False)
pairs_df.to_csv(os.path.join(PHASE4_DIR, 'query_similarity_pairs.csv'), index=False)

# Analyze bands
bands = {
    ">= 0.95": len(pairs_df[pairs_df['similarity'] >= 0.95]),
    ">= 0.90": len(pairs_df[(pairs_df['similarity'] >= 0.90) & (pairs_df['similarity'] < 0.95)]),
    ">= 0.85": len(pairs_df[(pairs_df['similarity'] >= 0.85) & (pairs_df['similarity'] < 0.90)]),
    ">= 0.80": len(pairs_df[(pairs_df['similarity'] >= 0.80) & (pairs_df['similarity'] < 0.85)])
}
print("Similarity Bands:", bands)

same_label_rate = pairs_df[pairs_df['similarity'] >= 0.80]['same_article'].mean()
print(f"Same article rate for >= 0.80: {same_label_rate:.4f}")

# 4. PROPOSE GROUPS
# Group queries if their similarity is >= 0.85
G = nx.Graph()
for i in range(n):
    G.add_node(i)

for _, row in pairs_df[pairs_df['similarity'] >= 0.85].iterrows():
    idx1 = queries_train[queries_train['query'] == row['query_1']].index[0]
    idx2 = queries_train[queries_train['query'] == row['query_2']].index[0]
    G.add_edge(idx1, idx2)

groups = []
group_id = 1
for component in nx.connected_components(G):
    if len(component) > 1:
        member_queries = [queries_train.iloc[idx]['query'] for idx in component]
        member_ids = [queries_train.iloc[idx].get('query_id', idx) for idx in component]
        labels = [queries_train.iloc[idx].get('relevant_article_id', queries_train.iloc[idx].get('article_id')) for idx in component]
        groups.append({
            'group_id': group_id,
            'member_query_ids': member_ids,
            'member_queries': member_queries,
            'article_labels': labels,
            'reason_for_grouping': 'Char TF-IDF Similarity >= 0.85'
        })
        group_id += 1

groups_df = pd.DataFrame(groups)
groups_df.to_csv(os.path.join(PHASE4_DIR, 'query_groups.csv'), index=False)

print(f"Proposed {len(groups)} groups containing {sum([len(g) for g in groups_df['member_query_ids']])} queries.")
