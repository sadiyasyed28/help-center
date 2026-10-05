import unittest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from evaluation import recall_at_k, mrr

class TestEvaluation(unittest.TestCase):
    
    def test_case_1(self):
        ranked = ["A", "B", "C"]
        true_id = "B"
        self.assertEqual(recall_at_k(ranked, true_id, k=3), 1)
        self.assertEqual(mrr(ranked, true_id), 0.5)

    def test_case_2(self):
        ranked = ["A", "B", "C"]
        true_id = "D"
        self.assertEqual(recall_at_k(ranked, true_id, k=3), 0)
        self.assertEqual(mrr(ranked, true_id), 0)

    def test_case_3(self):
        ranked = ["D", "A", "B"]
        true_id = "D"
        self.assertEqual(recall_at_k(ranked, true_id, k=3), 1)
        self.assertEqual(mrr(ranked, true_id), 1)

    def test_recall_at_1(self):
        ranked = ["A", "B", "C"]
        self.assertEqual(recall_at_k(ranked, "A", k=1), 1)
        self.assertEqual(recall_at_k(ranked, "B", k=1), 0)

    def test_true_article_at_rank_4(self):
        ranked = ["A", "B", "C", "D"]
        true_id = "D"
        self.assertEqual(recall_at_k(ranked, true_id, k=3), 0)
        self.assertEqual(recall_at_k(ranked, true_id, k=4), 1)
        self.assertEqual(mrr(ranked, true_id), 0.25)

    def test_empty_ranking(self):
        ranked = []
        true_id = "A"
        self.assertEqual(recall_at_k(ranked, true_id, k=3), 0)
        self.assertEqual(mrr(ranked, true_id), 0)

    def test_invalid_k(self):
        ranked = ["A", "B", "C"]
        true_id = "A"
        self.assertEqual(recall_at_k(ranked, true_id, k=0), 0)
        self.assertEqual(recall_at_k(ranked, true_id, k=-1), 0)

    def test_duplicate_ranking_ids(self):
        ranked = ["A", "B", "A"]
        true_id = "A"
        self.assertEqual(recall_at_k(ranked, true_id, k=3), 1)
        self.assertEqual(mrr(ranked, true_id), 1.0)
        
        ranked2 = ["B", "A", "A"]
        self.assertEqual(mrr(ranked2, "A"), 0.5)

if __name__ == '__main__':
    unittest.main()
