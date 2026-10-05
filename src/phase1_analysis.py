from pathlib import Path
import pandas as pd
import numpy as np
import re
import os
from collections import Counter

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
REPORT_DIR = PROJECT_ROOT / "reports\phase1"
MD_PATH = REPORT_DIR / "n.md"

os.makedirs(REPORT_DIR, exist_ok=True)

# Load data
articles = pd.read_csv(os.path.join(DATA_DIR, 'articles.csv'))
q_train = pd.read_csv(os.path.join(DATA_DIR, 'queries_train.csv'))
q_test = pd.read_csv(os.path.join(DATA_DIR, 'queries_test.csv'))

md_lines = []
md_lines.append("# Phase 1 — Data Exploration\n")

# Tokenizer
def tokenize(text):
    if not isinstance(text, str): return []
    return re.findall(r'\b\w+\b', text.lower())

def calc_stats(series):
    return {
        'min': series.min(), 'max': series.max(), 'mean': series.mean(),
        'median': series.median(), 'std': series.std(),
        'p10': series.quantile(0.10), 'p25': series.quantile(0.25),
        'p50': series.quantile(0.50), 'p75': series.quantile(0.75),
        'p90': series.quantile(0.90), 'p95': series.quantile(0.95)
    }

# 1. CLASS BALANCE
md_lines.append("## Class balance\n")
class_counts = q_train.groupby('relevant_article_id').size().reset_index(name='query_count')
class_counts = class_counts.merge(articles[['article_id', 'title']], left_on='relevant_article_id', right_on='article_id', how='left')
class_counts = class_counts[['article_id', 'title', 'query_count']].sort_values('query_count', ascending=False)
class_counts.to_csv(os.path.join(REPORT_DIR, 'class_balance.csv'), index=False)
stats = calc_stats(class_counts['query_count'])
md_lines.append(f"Queries per article: Mean {stats['mean']:.1f}, Min {stats['min']}, Max {stats['max']}, Std {stats['std']:.2f}")
md_lines.append(f"Total queries: {class_counts['query_count'].sum()} (Expected: 216)")
md_lines.append(f"Unique articles: {class_counts['article_id'].nunique()} (Expected: 24)")
md_lines.append(f"Dataset is {'perfectly balanced' if stats['std'] == 0 else 'imbalanced'}.\n")

# 2. QUERY LENGTH
md_lines.append("## Query length\n")
for df, name in [(q_train, 'Train'), (q_test, 'Test')]:
    df['char_count'] = df['query'].str.len()
    df['word_count'] = df['query'].str.split().str.len()
    df['tokens'] = df['query'].apply(tokenize)
    df['token_count'] = df['tokens'].apply(len)
    
q_train.to_csv(os.path.join(REPORT_DIR, 'train_query_lengths.csv'), index=False)
q_test.to_csv(os.path.join(REPORT_DIR, 'test_query_lengths.csv'), index=False)

for name, df in [('Train', q_train), ('Test', q_test)]:
    md_lines.append(f"### {name} Queries")
    for metric in ['char_count', 'word_count', 'token_count']:
        s = calc_stats(df[metric])
        md_lines.append(f"- **{metric}**: mean {s['mean']:.1f}, median {s['median']:.1f}, min {s['min']}, max {s['max']}, p90 {s['p90']:.1f}")
md_lines.append("\n")

# 3. FILLER/NOISE PATTERNS
md_lines.append("## Filler/noise patterns\n")
prefixes = {1: [], 2: [], 3: []}
suffixes = {1: [], 2: [], 3: []}

for tokens in q_train['tokens']:
    if len(tokens) >= 1:
        prefixes[1].append(tokens[0])
        suffixes[1].append(tokens[-1])
    if len(tokens) >= 2:
        prefixes[2].append(" ".join(tokens[:2]))
        suffixes[2].append(" ".join(tokens[-2:]))
    if len(tokens) >= 3:
        prefixes[3].append(" ".join(tokens[:3]))
        suffixes[3].append(" ".join(tokens[-3:]))

for n in [1, 2, 3]:
    mc = Counter(prefixes[n]).most_common(5)
    md_lines.append(f"Top {n}-token prefixes: " + ", ".join([f"'{k}'({v})" for k,v in mc]))
for n in [1, 2, 3]:
    mc = Counter(suffixes[n]).most_common(5)
    md_lines.append(f"Top {n}-token suffixes: " + ", ".join([f"'{k}'({v})" for k,v in mc]))
md_lines.append("\n")

# 4. ARTICLE-UNKNOWN VOCAB
md_lines.append("## Article-unknown vocabulary\n")
article_vocab = set()
for _, r in articles.iterrows():
    article_vocab.update(tokenize(r['title']))
    article_vocab.update(tokenize(r['body']))

unknown_tokens = []
unk_ratios = []
for tokens in q_train['tokens']:
    unk = [t for t in tokens if t not in article_vocab]
    unknown_tokens.extend(unk)
    ratio = len(unk) / len(tokens) if len(tokens)>0 else 0
    unk_ratios.append(ratio)
    
q_train['unknown_ratio'] = unk_ratios
s = calc_stats(pd.Series(unk_ratios))
md_lines.append(f"Unknown token ratio: mean {s['mean']:.3f}, median {s['median']:.3f}, max {s['max']:.3f}")
mc = Counter(unknown_tokens).most_common(10)
md_lines.append(f"Most frequent unknown tokens: " + ", ".join([f"'{k}'({v})" for k,v in mc]))
pd.DataFrame(Counter(unknown_tokens).most_common(), columns=['token','count']).to_csv(os.path.join(REPORT_DIR, 'unknown_tokens.csv'), index=False)
md_lines.append("\n")

# 5. QUERY/ARTICLE VOCABULARY OVERLAP
md_lines.append("## Query/article vocabulary overlap\n")
overlap_scores = []
unique_overlap_scores = []

for _, r in q_train.iterrows():
    q_tokens = r['tokens']
    if len(q_tokens) == 0:
        overlap_scores.append(0.0)
        unique_overlap_scores.append(0.0)
        continue
    
    a_row = articles[articles['article_id'] == r['relevant_article_id']].iloc[0]
    a_tokens = set(tokenize(a_row['title']) + tokenize(a_row['body']))
    
    overlap = sum(1 for t in q_tokens if t in a_tokens)
    overlap_scores.append(overlap / len(q_tokens))
    
    q_uniq = set(q_tokens)
    u_overlap = sum(1 for t in q_uniq if t in a_tokens)
    unique_overlap_scores.append(u_overlap / len(q_uniq))

q_train['overlap'] = overlap_scores
s1 = calc_stats(q_train['overlap'])
s2 = calc_stats(pd.Series(unique_overlap_scores))
md_lines.append(f"Token overlap ratio: mean {s1['mean']:.3f}, median {s1['median']:.3f}, max {s1['max']:.3f}")
md_lines.append(f"Unique token overlap ratio: mean {s2['mean']:.3f}, median {s2['median']:.3f}, max {s2['max']:.3f}")

lowest_overlap = q_train.sort_values('overlap').head(10)
lowest_overlap.to_csv(os.path.join(REPORT_DIR, 'lowest_overlap.csv'), index=False)

md_lines.append("\n## Vocabulary-gap examples\n")
for _, r in lowest_overlap.iterrows():
    a_row = articles[articles['article_id'] == r['relevant_article_id']].iloc[0]
    md_lines.append(f"**Query**: {r['query']} ({r['overlap']:.2f} overlap)")
    md_lines.append(f"**Title**: {a_row['title']}\n")

# 7. TRAIN VS TEST
md_lines.append("## Train vs test comparison\n")
q_test['unknown_ratio'] = q_test['tokens'].apply(lambda toks: sum(1 for t in toks if t not in article_vocab)/len(toks) if len(toks)>0 else 0)
md_lines.append(f"Train - Avg words: {q_train['word_count'].mean():.1f}, Avg unknown ratio: {q_train['unknown_ratio'].mean():.3f}")
md_lines.append(f"Test  - Avg words: {q_test['word_count'].mean():.1f}, Avg unknown ratio: {q_test['unknown_ratio'].mean():.3f}\n")

# 8. ARTICLE ANALYSIS
md_lines.append("## Article similarity/difficulty observations\n")
articles['title_len'] = articles['title'].str.len()
articles['body_len'] = articles['body'].str.len()
articles['token_count'] = articles['body'].apply(lambda x: len(tokenize(x)))
cat_dist = articles['category'].value_counts().to_dict()
md_lines.append(f"Category distribution: {cat_dist}")
md_lines.append(f"Title length: mean {articles['title_len'].mean():.1f}")
md_lines.append(f"Body tokens: mean {articles['token_count'].mean():.1f}")
md_lines.append("\n## Key findings\n(Generated from analysis)")

with open(MD_PATH, 'w', encoding='utf-8') as f:
    f.write("\n".join(md_lines))
    
print("Phase 1 Script Complete.")
