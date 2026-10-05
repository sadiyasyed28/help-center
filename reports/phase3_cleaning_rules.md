# Phase 3 — Query Cleaning Rules

Based on the Phase 1 Data Exploration, we identified a high prevalence of conversational filler and urgency terms in the user queries that do not correspond to the vocabulary of the target articles.

To test if a conservative query cleaner would improve traditional lexical retrieval (BM25), we implemented a deterministic cleaning function that uses regex to strip the following patterns from the query string (case-insensitively):

### List of patterns removed:
- `\bhow do i\b`
- `\bhow do\b`
- `\bhow\b`
- `\bneed help\b`
- `\bneed\b`
- `\bquestion\b`
- `\bhi\b`
- `\bhelp\b`
- `\bcan you tell\b`
- `\bcan you\b`
- `\basap\b`
- `\bplease\b`
- `\btell me\b`

### Rationale:
These terms were identified as the most frequent 1, 2, and 3-token prefixes/suffixes that have virtually 0 overlap with the article corpus.
We avoided aggressively removing domain words or frequent verbs (like "get", "account", "shop") to ensure no semantic meaning was inadvertently removed.

### Result:
These rules were tested in version `v5_bm25_cleaned` to directly compare against `v4_bm25` (no cleaning). The cleaning process did improve retrieval metrics marginally, confirming that conversational filler was slightly disrupting keyword matching.
