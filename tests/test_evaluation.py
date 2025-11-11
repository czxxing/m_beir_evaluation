"""
Unit tests for M-BEIR evaluation.
"""

import unittest
import tempfile
import json
from pathlib import Path
import numpy as np

from src.data.dataloader import MBEIRDataLoader, create_sample_dataset
from src.evaluation.metrics import calculate_metrics


class TestEvaluation(unittest.TestCase):
    """Test cases for evaluation components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir) / "data"
        
        # Create sample dataset
        create_sample_dataset(self.data_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_data_loading(self):
        """Test data loading functionality."""
        # Create sample dataset in the test directory
        from src.data.dataloader import create_sample_dataset
        
        # Create dataset subdirectory
        dataset_dir = self.data_dir / "sample"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        create_sample_dataset(dataset_dir)
        
        # Rename files to match the expected naming convention
        import os
        os.rename(dataset_dir / "sample_queries.jsonl", dataset_dir / "test_queries.jsonl")
        os.rename(dataset_dir / "sample_corpus.jsonl", dataset_dir / "test_corpus.jsonl")
        os.rename(dataset_dir / "sample_qrels.jsonl", dataset_dir / "test_qrels.jsonl")
        
        data_loader = MBEIRDataLoader(self.data_dir)
        dataset = data_loader.load_dataset("sample", "test")
        
        self.assertIn('queries', dataset)
        self.assertIn('corpus', dataset)
        self.assertIn('qrels', dataset)
        
        self.assertEqual(len(dataset['queries']), 3)
        self.assertEqual(len(dataset['corpus']), 5)
        self.assertEqual(len(dataset['qrels']), 3)
    
    def test_metrics_calculation(self):
        """Test metrics calculation."""
        # Mock retrieval results
        retrieval_results = {
            'q1': ['d1', 'd4', 'd2', 'd3', 'd5'],
            'q2': ['d2', 'd5', 'd1', 'd3', 'd4'],
            'q3': ['d3', 'd1', 'd2', 'd4', 'd5']
        }
        
        # Mock qrels
        qrels = {
            'q1': ['d1', 'd4'],
            'q2': ['d2', 'd5'],
            'q3': ['d3']
        }
        
        metrics = calculate_metrics(
            retrieval_results, 
            qrels, 
            ['ndcg@5', 'recall@5', 'map', 'mrr']
        )
        
        self.assertIn('ndcg@5', metrics)
        self.assertIn('recall@5', metrics)
        self.assertIn('map', metrics)
        self.assertIn('mrr', metrics)
        
        # All metrics should be between 0 and 1
        for metric_name, score in metrics.items():
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
    
    def test_perfect_retrieval(self):
        """Test metrics for perfect retrieval."""
        retrieval_results = {
            'q1': ['d1', 'd4', 'd2', 'd3', 'd5'],
            'q2': ['d2', 'd5', 'd1', 'd3', 'd4'],
            'q3': ['d3', 'd1', 'd2', 'd4', 'd5']
        }
        
        qrels = {
            'q1': ['d1', 'd4'],
            'q2': ['d2', 'd5'],
            'q3': ['d3']
        }
        
        # For perfect retrieval at top positions
        perfect_retrieval = {
            'q1': ['d1', 'd4', 'd2', 'd3', 'd5'],
            'q2': ['d2', 'd5', 'd1', 'd3', 'd4'],
            'q3': ['d3', 'd1', 'd2', 'd4', 'd5']
        }
        
        metrics = calculate_metrics(perfect_retrieval, qrels, ['recall@2', 'ndcg@2'])
        
        # Recall@2 should be 1.0 for perfect retrieval
        self.assertEqual(metrics['recall@2'], 1.0)
        # NDCG@2 should be high (close to 1.0)
        self.assertGreater(metrics['ndcg@2'], 0.9)


if __name__ == '__main__':
    unittest.main()