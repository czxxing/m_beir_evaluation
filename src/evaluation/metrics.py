"""
Evaluation metrics for retrieval performance.
"""

import numpy as np
from typing import Dict, List, Any
import math
from sklearn.metrics.pairwise import cosine_similarity


def calculate_metrics(retrieval_results: Dict[str, List[str]], 
                     qrels: Dict[str, List[str]], 
                     metrics: List[str]) -> Dict[str, float]:
    """
    Calculate retrieval evaluation metrics.
    
    Args:
        retrieval_results: Dictionary mapping query IDs to retrieved document IDs
        qrels: Dictionary mapping query IDs to relevant document IDs
        metrics: List of metric names to calculate
        
    Returns:
        Dictionary of metric scores
    """
    results = {}
    
    for metric in metrics:
        if metric.startswith('ndcg@'):
            k = int(metric.split('@')[1])
            results[metric] = calculate_ndcg(retrieval_results, qrels, k)
        elif metric.startswith('recall@'):
            k = int(metric.split('@')[1])
            results[metric] = calculate_recall(retrieval_results, qrels, k)
        elif metric == 'map':
            results[metric] = calculate_map(retrieval_results, qrels)
        elif metric == 'mrr':
            results[metric] = calculate_mrr(retrieval_results, qrels)
        elif metric == 'precision@':
            k = int(metric.split('@')[1])
            results[metric] = calculate_precision(retrieval_results, qrels, k)
    
    return results


def calculate_ndcg(retrieval_results: Dict[str, List[str]], 
                  qrels: Dict[str, List[str]], 
                  k: int = 10) -> float:
    """Calculate Normalized Discounted Cumulative Gain at k."""
    ndcg_scores = []
    
    for query_id, retrieved_docs in retrieval_results.items():
        if query_id not in qrels:
            continue
            
        relevant_docs = set(qrels[query_id])
        dcg = 0.0
        idcg = 0.0
        
        # Calculate DCG
        for i, doc_id in enumerate(retrieved_docs[:k]):
            if doc_id in relevant_docs:
                dcg += 1.0 / np.log2(i + 2)  # i+2 because i starts from 0
        
        # Calculate IDCG (ideal DCG)
        num_relevant = min(len(relevant_docs), k)
        for i in range(num_relevant):
            idcg += 1.0 / np.log2(i + 2)
        
        if idcg > 0:
            ndcg_scores.append(dcg / idcg)
        else:
            ndcg_scores.append(0.0)
    
    return np.mean(ndcg_scores) if ndcg_scores else 0.0


def calculate_recall(retrieval_results: Dict[str, List[str]], 
                    qrels: Dict[str, List[str]], 
                    k: int = 10) -> float:
    """Calculate Recall at k."""
    recall_scores = []
    
    for query_id, retrieved_docs in retrieval_results.items():
        if query_id not in qrels:
            continue
            
        relevant_docs = set(qrels[query_id])
        retrieved_at_k = set(retrieved_docs[:k])
        
        if len(relevant_docs) > 0:
            recall = len(retrieved_at_k.intersection(relevant_docs)) / len(relevant_docs)
            recall_scores.append(recall)
    
    return np.mean(recall_scores) if recall_scores else 0.0


def calculate_precision(retrieval_results: Dict[str, List[str]], 
                       qrels: Dict[str, List[str]], 
                       k: int = 10) -> float:
    """Calculate Precision at k."""
    precision_scores = []
    
    for query_id, retrieved_docs in retrieval_results.items():
        if query_id not in qrels:
            continue
            
        relevant_docs = set(qrels[query_id])
        retrieved_at_k = retrieved_docs[:k]
        
        num_relevant = sum(1 for doc_id in retrieved_at_k if doc_id in relevant_docs)
        precision = num_relevant / k
        precision_scores.append(precision)
    
    return np.mean(precision_scores) if precision_scores else 0.0


def calculate_map(retrieval_results: Dict[str, List[str]], 
                 qrels: Dict[str, List[str]]) -> float:
    """Calculate Mean Average Precision."""
    ap_scores = []
    
    for query_id, retrieved_docs in retrieval_results.items():
        if query_id not in qrels:
            continue
            
        relevant_docs = set(qrels[query_id])
        precision_at_k = []
        num_relevant_retrieved = 0
        
        for i, doc_id in enumerate(retrieved_docs):
            if doc_id in relevant_docs:
                num_relevant_retrieved += 1
                precision_at_k.append(num_relevant_retrieved / (i + 1))
        
        if precision_at_k:
            ap = np.mean(precision_at_k)
            ap_scores.append(ap)
    
    return np.mean(ap_scores) if ap_scores else 0.0


def calculate_mrr(retrieval_results: Dict[str, List[str]], 
                 qrels: Dict[str, List[str]]) -> float:
    """Calculate Mean Reciprocal Rank."""
    rr_scores = []
    
    for query_id, retrieved_docs in retrieval_results.items():
        if query_id not in qrels:
            continue
            
        relevant_docs = set(qrels[query_id])
        
        for i, doc_id in enumerate(retrieved_docs):
            if doc_id in relevant_docs:
                rr_scores.append(1.0 / (i + 1))
                break
        else:
            rr_scores.append(0.0)
    
    return np.mean(rr_scores) if rr_scores else 0.0


def calculate_metrics_batch(queries: List[str], 
                          corpus: List[str], 
                          qrels: Dict[str, List[str]], 
                          model, 
                          metrics: List[str], 
                          k: int = 10) -> Dict[str, float]:
    """
    Calculate metrics for a batch of queries and corpus.
    
    Args:
        queries: List of query texts
        corpus: List of document texts
        qrels: Query relevance judgments
        model: Retrieval model
        metrics: List of metrics to calculate
        k: Top-k for metrics
        
    Returns:
        Dictionary of metric scores
    """
    # Encode queries and corpus
    query_embeddings = model.encode_texts(queries)
    corpus_embeddings = model.encode_texts(corpus)
    
    # Calculate similarities
    similarities = cosine_similarity(query_embeddings, corpus_embeddings)
    
    # Get retrieval results
    retrieval_results = {}
    for i, query_similarities in enumerate(similarities):
        query_id = f"q{i}"
        ranked_indices = np.argsort(query_similarities)[::-1]
        retrieval_results[query_id] = [f"d{j}" for j in ranked_indices]
    
    # Calculate metrics
    return calculate_metrics(retrieval_results, qrels, metrics)