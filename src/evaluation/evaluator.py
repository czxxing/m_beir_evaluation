"""
Main evaluation class for M-BEIR benchmark.
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import faiss
from sklearn.metrics.pairwise import cosine_similarity

from src.data.dataloader import MBEIRDataLoader
from src.models.retrieval_models import get_retrieval_model
from src.evaluation.metrics import calculate_metrics

logger = logging.getLogger(__name__)


class Evaluator:
    """Main evaluator class for M-BEIR benchmark."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize evaluator with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.data_loader = MBEIRDataLoader(
            config['dataset']['data_dir'],
            config['dataset'].get('cache_dir')
        )
        self.model = get_retrieval_model(config['model'])
        self.output_dir = Path(config['output']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run(self) -> Dict[str, Any]:
        """
        Run the complete evaluation pipeline.
        
        Returns:
            Dictionary containing evaluation results
        """
        logger.info("Starting M-BEIR evaluation pipeline")
        
        # Load dataset
        dataset = self.data_loader.load_dataset(
            self.config['dataset']['name'],
            self.config['dataset']['split']
        )
        
        # Extract text content
        queries_text = self._extract_text_content(dataset['queries'])
        corpus_text = self._extract_text_content(dataset['corpus'])
        
        # Encode queries and corpus
        logger.info("Encoding queries...")
        query_embeddings = self.model.encode_texts(
            queries_text, 
            batch_size=self.config['model']['batch_size']
        )
        
        logger.info("Encoding corpus...")
        corpus_embeddings = self.model.encode_texts(
            corpus_text,
            batch_size=self.config['model']['batch_size']
        )
        
        # Build FAISS index for efficient retrieval
        logger.info("Building FAISS index...")
        index = self._build_faiss_index(corpus_embeddings)
        
        # Perform retrieval
        logger.info("Performing retrieval...")
        retrieval_results = self._retrieve(
            query_embeddings, 
            index, 
            top_k=self.config['retrieval']['top_k']
        )
        
        # Calculate metrics
        logger.info("Calculating evaluation metrics...")
        metrics = calculate_metrics(
            retrieval_results, 
            dataset['qrels'],
            metrics=self.config['evaluation']['metrics']
        )
        
        # Prepare results
        results = {
            'config': self.config,
            'metrics': metrics,
            'retrieval_results': retrieval_results if self.config['output']['save_predictions'] else None,
            'embeddings': {
                'query_embeddings': query_embeddings if self.config['output']['save_embeddings'] else None,
                'corpus_embeddings': corpus_embeddings if self.config['output']['save_embeddings'] else None
            }
        }
        
        logger.info("Evaluation pipeline completed")
        return results
    
    def _extract_text_content(self, documents: Dict[str, Dict]) -> List[str]:
        """Extract text content from documents."""
        texts = []
        for doc_id, doc in documents.items():
            # Try to find text fields
            text_fields = ['text', 'content', 'title', 'description']
            for field in text_fields:
                if field in doc and doc[field]:
                    texts.append(str(doc[field]))
                    break
            else:
                # If no text field found, use the first string field
                for value in doc.values():
                    if isinstance(value, str):
                        texts.append(value)
                        break
                else:
                    texts.append("")
        return texts
    
    def _build_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Build FAISS index for efficient similarity search."""
        dimension = embeddings.shape[1]
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create index
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        index.add(embeddings.astype('float32'))
        
        return index
    
    def _retrieve(self, query_embeddings: np.ndarray, index: faiss.Index, top_k: int = 1000) -> Dict[str, List[str]]:
        """Perform retrieval using FAISS index."""
        # Normalize query embeddings
        faiss.normalize_L2(query_embeddings)
        
        # Search
        scores, indices = index.search(query_embeddings.astype('float32'), top_k)
        
        # Convert to document IDs (assuming indices correspond to document order)
        retrieval_results = {}
        for i, (query_scores, query_indices) in enumerate(zip(scores, indices)):
            query_id = f"q{i+1}"  # This should match actual query IDs
            retrieval_results[query_id] = [f"d{idx+1}" for idx in query_indices]
        
        return retrieval_results
    
    def save_results(self, results: Dict[str, Any]):
        """Save evaluation results to files."""
        # Save metrics
        metrics_file = self.output_dir / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(results['metrics'], f, indent=2)
        
        # Save configuration
        config_file = self.output_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(results['config'], f, indent=2)
        
        # Save retrieval results if requested
        if results['retrieval_results']:
            retrieval_file = self.output_dir / "retrieval_results.json"
            with open(retrieval_file, 'w') as f:
                json.dump(results['retrieval_results'], f, indent=2)
        
        # Save embeddings if requested
        if results['embeddings']['query_embeddings'] is not None:
            np.save(self.output_dir / "query_embeddings.npy", results['embeddings']['query_embeddings'])
        if results['embeddings']['corpus_embeddings'] is not None:
            np.save(self.output_dir / "corpus_embeddings.npy", results['embeddings']['corpus_embeddings'])
        
        logger.info(f"Results saved to: {self.output_dir}")