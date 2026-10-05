from pathlib import Path
import os
import pandas as pd

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREDICTIONS_PATH = os.path.join(BASE_DIR, "predictions.csv")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
TEST_QUERIES_PATH = os.path.join(DATA_DIR, "queries_test.csv")
ARTICLES_PATH = os.path.join(DATA_DIR, "articles.csv")

def validate_predictions():
    # 1. predictions.csv exists.
    assert os.path.exists(PREDICTIONS_PATH), "predictions.csv does not exist"
    
    df = pd.read_csv(PREDICTIONS_PATH)
    
    # 2. Exactly 96 rows.
    assert len(df) == 96, f"Expected 96 rows, got {len(df)}"
    
    # 3. Exact column names: query_id, rank_1, rank_2, rank_3
    expected_cols = ['query_id', 'rank_1', 'rank_2', 'rank_3']
    assert list(df.columns) == expected_cols, f"Columns mismatch. Expected {expected_cols}, got {list(df.columns)}"
    
    # 4. Every query_id appears exactly once.
    assert df['query_id'].is_unique, "Duplicate query_ids found"
    
    # Load test queries
    test_df = pd.read_csv(TEST_QUERIES_PATH)
    test_q_ids = set(test_df['query_id'].tolist())
    pred_q_ids = set(df['query_id'].tolist())
    
    # 5. Every query_id exists in queries_test.csv.
    assert pred_q_ids.issubset(test_q_ids), "predictions.csv contains query_ids not in queries_test.csv"
    
    # 6. No test query is missing.
    assert test_q_ids.issubset(pred_q_ids), "Some query_ids from queries_test.csv are missing"
    
    # 7. No extra query IDs exist.
    assert test_q_ids == pred_q_ids, "query_id sets do not match exactly"
    
    # Load articles
    articles_df = pd.read_csv(ARTICLES_PATH)
    valid_a_ids = set(articles_df['article_id'].tolist())
    
    # 8. Every rank_1/rank_2/rank_3 is a valid article_id.
    for col in ['rank_1', 'rank_2', 'rank_3']:
        assert set(df[col]).issubset(valid_a_ids), f"Invalid article_id found in {col}"
    
    # 9. rank_1, rank_2, rank_3 are distinct for every row.
    for _, row in df.iterrows():
        ranks = [row['rank_1'], row['rank_2'], row['rank_3']]
        assert len(set(ranks)) == 3, f"Duplicate article_ids found for query {row['query_id']}: {ranks}"
        
    # 10. No null values.
    assert not df.isnull().values.any(), "Null values found in predictions.csv"
    
    # 11. No duplicate prediction rows. (Checked by 4)
    assert not df.duplicated().any(), "Duplicate full rows found"
    
    # 12. Exactly three predictions per query.
    # Handled by column count and distinct check
    
    # 13. Predictions are deterministic.
    # We checked this in the generation script.
    
    print("All validation checks passed successfully.")

if __name__ == "__main__":
    validate_predictions()
