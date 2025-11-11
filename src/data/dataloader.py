"""
Data loading utilities for M-BEIR datasets.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class MBEIRDataLoader:
    """Data loader for M-BEIR datasets."""
    
    def __init__(self, data_dir: str, cache_dir: Optional[str] = None):
        """
        Initialize data loader.
        
        Args:
            data_dir: Directory containing dataset files
            cache_dir: Directory for caching processed data
        """
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else Path("./cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def load_dataset(self, dataset_name: str, split: str = "test") -> Dict:
        """
        Load M-BEIR dataset.
        
        Args:
            dataset_name: Name of the dataset
            split: Data split (train, validation, test)
            
        Returns:
            Dictionary containing queries, corpus, and qrels
        """
        logger.info(f"Loading dataset: {dataset_name}, split: {split}")
        
        # Define file paths - try both naming conventions
        queries_path = self.data_dir / dataset_name / f"{split}_queries.jsonl"
        corpus_path = self.data_dir / dataset_name / f"{split}_corpus.jsonl"
        qrels_path = self.data_dir / dataset_name / f"{split}_qrels.jsonl"
        
        # If files don't exist with split prefix, try dataset name prefix
        if not queries_path.exists():
            queries_path = self.data_dir / dataset_name / f"{dataset_name}_queries.jsonl"
        if not corpus_path.exists():
            corpus_path = self.data_dir / dataset_name / f"{dataset_name}_corpus.jsonl"
        if not qrels_path.exists():
            qrels_path = self.data_dir / dataset_name / f"{dataset_name}_qrels.jsonl"
        
        # Load queries
        queries = self._load_jsonl(queries_path)
        queries = {q['_id']: q for q in queries}
        
        # Load corpus
        corpus = self._load_jsonl(corpus_path)
        corpus = {doc['_id']: doc for doc in corpus}
        
        # Load qrels (query relevance judgments)
        qrels = self._load_jsonl(qrels_path)
        qrels_dict = {}
        for qrel in qrels:
            query_id = qrel['query_id']
            doc_id = qrel['doc_id']
            if query_id not in qrels_dict:
                qrels_dict[query_id] = []
            qrels_dict[query_id].append(doc_id)
        
        return {
            'queries': queries,
            'corpus': corpus,
            'qrels': qrels_dict
        }
    
    def _load_jsonl(self, file_path: Path) -> List[Dict]:
        """Load JSONL file."""
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        
        return data
    
    def get_modality_info(self, dataset: Dict) -> Dict[str, List[str]]:
        """
        Extract modality information from dataset.
        
        Args:
            dataset: Loaded dataset dictionary
            
        Returns:
            Dictionary mapping modality types to field names
        """
        modalities = {
            'text': [],
            'image': []
        }
        
        # Check first query and document for available fields
        if dataset['queries']:
            first_query = next(iter(dataset['queries'].values()))
            for field, value in first_query.items():
                if field.startswith('text') or isinstance(value, str):
                    modalities['text'].append(field)
                elif field.startswith('image') or (isinstance(value, str) and 
                                                  (value.endswith('.jpg') or value.endswith('.png'))):
                    modalities['image'].append(field)
        
        return modalities


def create_sample_dataset(data_dir: str):
    """
    Create a sample dataset for testing purposes.
    
    Args:
        data_dir: Directory to create sample dataset
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample queries
    queries = [
        {"_id": "q1", "text": "A cat playing with a ball", "modality": "text"},
        {"_id": "q2", "text": "A dog running in the park", "modality": "text"},
        {"_id": "q3", "text": "A sunset over mountains", "modality": "text"}
    ]
    
    # Sample corpus
    corpus = [
        {"_id": "d1", "text": "A cute cat playing with a red ball", "modality": "text"},
        {"_id": "d2", "text": "A golden retriever running happily", "modality": "text"},
        {"_id": "d3", "text": "Beautiful mountain sunset scenery", "modality": "text"},
        {"_id": "d4", "text": "A cat sleeping on a sofa", "modality": "text"},
        {"_id": "d5", "text": "A dog chasing its tail", "modality": "text"}
    ]
    
    # Sample qrels
    qrels = [
        {"query_id": "q1", "doc_id": "d1", "score": 1},
        {"query_id": "q1", "doc_id": "d4", "score": 1},
        {"query_id": "q2", "doc_id": "d2", "score": 1},
        {"query_id": "q2", "doc_id": "d5", "score": 1},
        {"query_id": "q3", "doc_id": "d3", "score": 1}
    ]
    
    # Save files
    with open(data_dir / "sample_queries.jsonl", 'w') as f:
        for query in queries:
            f.write(json.dumps(query) + '\n')
    
    with open(data_dir / "sample_corpus.jsonl", 'w') as f:
        for doc in corpus:
            f.write(json.dumps(doc) + '\n')
    
    with open(data_dir / "sample_qrels.jsonl", 'w') as f:
        for qrel in qrels:
            f.write(json.dumps(qrel) + '\n')
    
    logger.info(f"Sample dataset created at: {data_dir}")