from pathlib import Path
import pandas as pd
import re
import os
import unicodedata
import sys

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
REPORT_DIR = PROJECT_ROOT / "reports"
MD_PATH = os.path.join(REPORT_DIR, 'phase2_baseline.md')
RESULTS_CSV_PATH = os.path.join(REPORT_DIR, 'results.csv')

# Load evaluation module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from evaluation import evaluate

def normalize(text):
    if not isinstance(text, str):
        return set()
    # Unicode normalization
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    # Lowercase
    text = text.lower()
    # Simple regex for words, handles punctuation and whitespace implicitly
    tokens = set(re.findall(r'\b\w+\b', text))
    return tokens

def build_article_docs():
    articles = pd.read_csv(os.path.join(DATA_DIR, 'articles.csv'))
    article_docs = {}
    for _, r in articles.iterrows():
        text = str(r['title']) + " " + str(r['body'])
        article_docs[r['article_id']] = normalize(text)
    return article_docs

def rank_articles(query_text, article_docs, scoring_method):
    query_tokens = normalize(query_text)
    scores = []
    
    for aid, doc_tokens in article_docs.items():
        score = scoring_method(query_tokens, doc_tokens)
        scores.append((score, aid))
        
    # Sort descending by score, ascending by article_id
    # To do this, sort ascending by (-score, aid)
    scores.sort(key=lambda x: (-x[0], x[1]))
    
    return [aid for score, aid in scores], scores

def shared_word_count(query_tokens, doc_tokens):
    return len(query_tokens.intersection(doc_tokens))

def jaccard(query_tokens, doc_tokens):
    intersection = len(query_tokens.intersection(doc_tokens))
    union = len(query_tokens.union(doc_tokens))
    if union == 0:
        return 0.0
    return intersection / union

def get_detailed_results(search_fn, queries_df, article_docs, scoring_method):
    agg_rec3, agg_mrr, agg_rec1, results_df = evaluate(lambda q: rank_articles(q, article_docs, scoring_method)[0], queries_df)
    
    # We want more details
    detailed_rows = []
    for _, r in queries_df.iterrows():
        qid = r['query_id']
        query = r['query']
        true_id = r['relevant_article_id']
        
        ranked_ids, scores = rank_articles(query, article_docs, scoring_method)
        scores_dict = dict([(aid, score) for score, aid in scores])
        
        try:
            rank = ranked_ids.index(true_id) + 1
            rr = 1.0 / rank
        except ValueError:
            rank = -1
            rr = 0.0
            
        detailed_rows.append({
            'query_id': qid,
            'query': query,
            'true_article_id': true_id,
            'rank_1': ranked_ids[0] if len(ranked_ids) > 0 else None,
            'rank_2': ranked_ids[1] if len(ranked_ids) > 1 else None,
            'rank_3': ranked_ids[2] if len(ranked_ids) > 2 else None,
            'rank_of_true_article': rank,
            'reciprocal_rank': rr,
            'score_of_true_article': scores_dict.get(true_id, 0),
            'score_of_top1_article': scores[0][0] if len(scores) > 0 else 0
        })
    return agg_rec1, agg_rec3, agg_mrr, pd.DataFrame(detailed_rows)

def main():
    article_docs = build_article_docs()
    queries_train = pd.read_csv(os.path.join(DATA_DIR, 'queries_train.csv'))
    
    # Baseline A
    rec1_a, rec3_a, mrr_a, df_a = get_detailed_results(
        shared_word_count, queries_train, article_docs, shared_word_count
    )
    df_a.to_csv(os.path.join(REPORT_DIR, 'v0_count_details.csv'), index=False)
    
    # Baseline B
    rec1_b, rec3_b, mrr_b, df_b = get_detailed_results(
        jaccard, queries_train, article_docs, jaccard
    )
    df_b.to_csv(os.path.join(REPORT_DIR, 'v0_jaccard_details.csv'), index=False)
    
    # Results CSV
    results_rows = [
        {
            'version': 'v0_count',
            'description': 'Shared Word Count Baseline',
            'uses_training_queries': False,
            'evaluation_method': 'Article-only retrieval evaluated on all 216 labeled queries',
            'recall_at_1': rec1_a,
            'recall_at_3': rec3_a,
            'mrr': mrr_a
        },
        {
            'version': 'v0_jaccard',
            'description': 'Jaccard Similarity Baseline',
            'uses_training_queries': False,
            'evaluation_method': 'Article-only retrieval evaluated on all 216 labeled queries',
            'recall_at_1': rec1_b,
            'recall_at_3': rec3_b,
            'mrr': mrr_b
        }
    ]
    results_df = pd.DataFrame(results_rows)
    if os.path.exists(RESULTS_CSV_PATH):
        existing_df = pd.read_csv(RESULTS_CSV_PATH)
        pd.concat([existing_df, results_df]).to_csv(RESULTS_CSV_PATH, index=False)
    else:
        results_df.to_csv(RESULTS_CSV_PATH, index=False)

    # Markdown Report
    best_method = 'v0_count' if rec3_a > rec3_b else 'v0_jaccard'
    
    # Sample queries
    # Successful: rec_rank == 1
    # Failure: rec_rank == 0 or rank > 3
    succ_a = df_a[df_a['rank_of_true_article'] == 1].head(3)
    fail_a = df_a[(df_a['rank_of_true_article'] > 3) | (df_a['rank_of_true_article'] == -1)].head(3)
    
    md = [
        "# Phase 2 — Baseline Retrieval",
        "\n## Method",
        "Two simple lexical retrieval baselines were implemented without any data cleaning, training query usage, or advanced techniques.",
        "1. **v0_count (Shared Word Count)**: Scores articles by the number of unique normalized query tokens present in the article.",
        "2. **v0_jaccard (Jaccard Similarity)**: Scores articles by the intersection size divided by the union size of query and article tokens.",
        "\n## Results",
        "| Method | Recall@1 | Recall@3 | MRR |",
        "|---|---|---|---|",
        f"| v0_count | {rec1_a:.4f} | {rec3_a:.4f} | {mrr_a:.4f} |",
        f"| v0_jaccard | {rec1_b:.4f} | {rec3_b:.4f} | {mrr_b:.4f} |",
        "\n## Comparison",
        f"The stronger baseline is **{best_method}** based on Recall@3 and MRR.",
        "\n## Examples",
        "### Successful Retrievals",
    ]
    for _, r in succ_a.iterrows():
        md.append(f"- **Query**: `{r['query']}` (True: {r['true_article_id']}) - Retrieved at Rank {r['rank_of_true_article']}")
    
    md.append("\n### Failed Retrievals")
    for _, r in fail_a.iterrows():
        md.append(f"- **Query**: `{r['query']}` (True: {r['true_article_id']}) - Ranked at {r['rank_of_true_article']}")
        
    md.append("\n## Limitations")
    md.append("Simple lexical matching (like count and Jaccard) struggles significantly because Phase 1 showed query-article vocabulary overlap is only ~32%. "
              "Queries often contain filler ('hi', 'please'), typos ('acocunt'), or different terminology ('1099 form' vs 'tax document') that simple token matching fails to capture.")
    
    with open(MD_PATH, 'w', encoding='utf-8') as f:
        f.write("\n".join(md))
        
    print(f"v0_count - R@1: {rec1_a:.4f}, R@3: {rec3_a:.4f}, MRR: {mrr_a:.4f}")
    print(f"v0_jaccard - R@1: {rec1_b:.4f}, R@3: {rec3_b:.4f}, MRR: {mrr_b:.4f}")
    print("Script Complete.")

if __name__ == "__main__":
    main()
