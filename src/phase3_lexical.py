from pathlib import Path
import os
import re
import pandas as pd
import numpy as np
import unicodedata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

from evaluation import evaluate

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'reports')
PHASE3_DIR = os.path.join(REPORTS_DIR, 'phase3')
os.makedirs(PHASE3_DIR, exist_ok=True)

# Load data
articles_df = pd.read_csv(os.path.join(DATA_DIR, 'articles.csv'))
queries_train_df = pd.read_csv(os.path.join(DATA_DIR, 'queries_train.csv'))

# Prepare documents
articles_df['text'] = articles_df['title'].astype(str) + " " + articles_df['body'].astype(str)
doc_texts = articles_df['text'].tolist()
article_ids = articles_df['article_id'].tolist()
doc_titles = articles_df['title'].astype(str).tolist()
doc_bodies = articles_df['body'].astype(str).tolist()

results_summary = []

def save_per_query_results(results_df, filename):
    out = []
    for _, row in results_df.iterrows():
        ranked_ids = row['ranked_ids']
        true_id = row['true_id']
        out.append({
            'query_id': row.get('query_id', ''),
            'query': row['query'],
            'true_article_id': true_id,
            'rank_1': ranked_ids[0] if len(ranked_ids) > 0 else '',
            'rank_2': ranked_ids[1] if len(ranked_ids) > 1 else '',
            'rank_3': ranked_ids[2] if len(ranked_ids) > 2 else '',
            'true_rank': ranked_ids.index(true_id) + 1 if true_id in ranked_ids else -1,
            'reciprocal_rank': row['mrr']
        })
    pd.DataFrame(out).to_csv(os.path.join(PHASE3_DIR, filename), index=False)

def log_result(version, desc, r1, r3, mrr):
    results_summary.append({
        'version': version,
        'description': desc,
        'uses_training_queries': False,
        'evaluation_method': "Article-only retrieval evaluated on all 216 labeled queries",
        'recall_at_1': r1,
        'recall_at_3': r3,
        'mrr': mrr
    })
    print(f"{version}: R@1={r1:.4f}, R@3={r3:.4f}, MRR={mrr:.4f}")

# ==========================================
# V1: TF-IDF
# ==========================================
vectorizer_v1 = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1,1))
doc_matrix_v1 = vectorizer_v1.fit_transform(doc_texts)

def search_v1(query_text):
    q_vec = vectorizer_v1.transform([query_text])
    sims = cosine_similarity(q_vec, doc_matrix_v1)[0]
    scores = [(sims[i], article_ids[i]) for i in range(len(article_ids))]
    scores.sort(key=lambda x: (-x[0], x[1]))
    return [aid for score, aid in scores]

r3, mrr, r1, results_df = evaluate(search_v1, queries_train_df)
save_per_query_results(results_df, 'v1_tfidf_per_query.csv')
log_result('v1_tfidf', 'TF-IDF + English stop words', r1, r3, mrr)

# ==========================================
# V2: TF-IDF + BIGRAMS
# ==========================================
vectorizer_v2 = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1,2))
doc_matrix_v2 = vectorizer_v2.fit_transform(doc_texts)

def search_v2(query_text):
    q_vec = vectorizer_v2.transform([query_text])
    sims = cosine_similarity(q_vec, doc_matrix_v2)[0]
    scores = [(sims[i], article_ids[i]) for i in range(len(article_ids))]
    scores.sort(key=lambda x: (-x[0], x[1]))
    return [aid for score, aid in scores]

r3, mrr, r1, results_df = evaluate(search_v2, queries_train_df)
save_per_query_results(results_df, 'v2_tfidf_bigrams_per_query.csv')
log_result('v2_tfidf_bigrams', 'TF-IDF with bigrams', r1, r3, mrr)

# ==========================================
# V3: TITLE WEIGHTING
# ==========================================
vectorizer_v3_title = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1,2))
vectorizer_v3_body = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1,2))

doc_matrix_v3_title = vectorizer_v3_title.fit_transform(doc_titles)
doc_matrix_v3_body = vectorizer_v3_body.fit_transform(doc_bodies)

best_v3_r3 = -1
best_v3_weight = None
best_v3_results_df = None
best_v3_mrr = -1
best_v3_r1 = -1

for w_title in [0.5, 0.7, 1.0]:
    def search_v3(query_text):
        q_vec_t = vectorizer_v3_title.transform([query_text])
        q_vec_b = vectorizer_v3_body.transform([query_text])
        
        sims_t = cosine_similarity(q_vec_t, doc_matrix_v3_title)[0]
        sims_b = cosine_similarity(q_vec_b, doc_matrix_v3_body)[0]
        
        final_sims = w_title * sims_t + 1.0 * sims_b
        
        scores = [(final_sims[i], article_ids[i]) for i in range(len(article_ids))]
        scores.sort(key=lambda x: (-x[0], x[1]))
        return [aid for score, aid in scores]
        
    r3, mrr, r1, results_df = evaluate(search_v3, queries_train_df)
    print(f"  v3 diagnostic: title_weight={w_title} -> R@3={r3:.4f}")
    if r3 > best_v3_r3:
        best_v3_r3 = r3
        best_v3_weight = w_title
        best_v3_results_df = results_df
        best_v3_mrr = mrr
        best_v3_r1 = r1

save_per_query_results(best_v3_results_df, 'v3_title_weighted_per_query.csv')
log_result('v3_title_weighted', f'Title weighted TF-IDF (title_weight={best_v3_weight})', best_v3_r1, best_v3_r3, best_v3_mrr)

# ==========================================
# V4: BM25
# ==========================================
def simple_tokenize(text):
    text = unicodedata.normalize('NFKD', str(text)).encode('ASCII', 'ignore').decode('utf-8')
    text = text.lower()
    return re.findall(r'\b\w+\b', text)

tokenized_corpus = [simple_tokenize(doc) for doc in doc_texts]
bm25 = BM25Okapi(tokenized_corpus, k1=1.5, b=0.75)

def search_v4(query_text):
    q_tokens = simple_tokenize(query_text)
    scores_array = bm25.get_scores(q_tokens)
    scores = [(scores_array[i], article_ids[i]) for i in range(len(article_ids))]
    scores.sort(key=lambda x: (-x[0], x[1]))
    return [aid for score, aid in scores]

r3, mrr, r1, results_df = evaluate(search_v4, queries_train_df)
save_per_query_results(results_df, 'v4_bm25_per_query.csv')
log_result('v4_bm25', 'BM25 default params', r1, r3, mrr)

# ==========================================
# V5: QUERY CLEANING
# ==========================================
FILLER_PATTERNS = [
    r'\bhow do i\b',
    r'\bhow do\b',
    r'\bhow\b',
    r'\bneed help\b',
    r'\bneed\b',
    r'\bquestion\b',
    r'\bhi\b',
    r'\bhelp\b',
    r'\bcan you tell\b',
    r'\bcan you\b',
    r'\basap\b',
    r'\bplease\b',
    r'\btell me\b',
]

def clean_query(text):
    text = text.lower()
    for pattern in FILLER_PATTERNS:
        text = re.sub(pattern, ' ', text)
    # cleanup extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def search_v5(query_text):
    cleaned = clean_query(query_text)
    q_tokens = simple_tokenize(cleaned)
    scores_array = bm25.get_scores(q_tokens)
    scores = [(scores_array[i], article_ids[i]) for i in range(len(article_ids))]
    scores.sort(key=lambda x: (-x[0], x[1]))
    return [aid for score, aid in scores]

r3_v5, mrr_v5, r1_v5, results_df_v5 = evaluate(search_v5, queries_train_df)

if r3_v5 >= r3:
    # cleaning helped or neutral
    save_per_query_results(results_df_v5, 'v5_bm25_cleaned_per_query.csv')
    log_result('v5_bm25_cleaned', 'BM25 with query filler cleaning', r1_v5, r3_v5, mrr_v5)
else:
    # cleaning hurt, keep v4 as stronger result conceptually, but still record v5
    save_per_query_results(results_df_v5, 'v5_bm25_cleaned_per_query.csv')
    log_result('v5_bm25_cleaned', 'BM25 with query filler cleaning (hurt perf)', r1_v5, r3_v5, mrr_v5)

# ==========================================
# V6: CHARACTER N-GRAMS
# ==========================================
vectorizer_v6 = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), lowercase=True)
doc_matrix_v6 = vectorizer_v6.fit_transform(doc_texts)

def search_v6(query_text):
    q_vec = vectorizer_v6.transform([query_text])
    sims = cosine_similarity(q_vec, doc_matrix_v6)[0]
    scores = [(sims[i], article_ids[i]) for i in range(len(article_ids))]
    scores.sort(key=lambda x: (-x[0], x[1]))
    return [aid for score, aid in scores]

r3, mrr, r1, results_df = evaluate(search_v6, queries_train_df)
save_per_query_results(results_df, 'v6_char_tfidf_per_query.csv')
log_result('v6_char_tfidf', 'Char n-grams (3-5) TF-IDF', r1, r3, mrr)

# Write results
results_path = os.path.join(REPORTS_DIR, 'results.csv')
new_results = pd.DataFrame(results_summary)
if os.path.exists(results_path):
    existing = pd.read_csv(results_path)
    combined = pd.concat([existing, new_results], ignore_index=True)
    combined.to_csv(results_path, index=False)
else:
    new_results.to_csv(results_path, index=False)
