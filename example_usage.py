#!/usr/bin/env python3
"""
Example usage script for M-BEIR evaluation.
"""

import json
from pathlib import Path
from src.data.dataloader import MBEIRDataLoader, create_sample_dataset
from src.models.retrieval_models import SentenceTransformerModel
from src.evaluation.metrics import calculate_metrics_batch


def main():
    """Run example evaluation."""
    print("M-BEIR Evaluation Example")
    print("=" * 50)
    
    # Create sample dataset
    data_dir = Path("./data")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dataset subdirectory
    dataset_dir = data_dir / "sample"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    create_sample_dataset(dataset_dir)
    
    # Load dataset - files are already named correctly
    data_loader = MBEIRDataLoader(data_dir)
    dataset = data_loader.load_dataset("sample", "sample")
    
    # Extract text content
    queries = [doc['text'] for doc in dataset['queries'].values()]
    corpus = [doc['text'] for doc in dataset['corpus'].values()]
    
    # Create mock qrels for the example
    qrels = {}
    for i, query_id in enumerate(dataset['queries'].keys()):
        qrels[query_id] = dataset['qrels'].get(query_id, [])
    
    print(f"Loaded {len(queries)} queries and {len(corpus)} documents")
    
    # Initialize model
    model = SentenceTransformerModel("sentence-transformers/all-mpnet-base-v2", device="cpu")
    
    # Define metrics to calculate
    metrics_to_calculate = ['ndcg@5', 'recall@5', 'map', 'mrr']
    
    print("\nCalculating metrics...")
    
    # Calculate metrics
    results = calculate_metrics_batch(
        queries, 
        corpus, 
        qrels, 
        model, 
        metrics_to_calculate, 
        k=5
    )
    
    # Display results
    print("\nEvaluation Results:")
    print("-" * 30)
    for metric, score in results.items():
        print(f"{metric}: {score:.4f}")
    
    # Save results
    output_dir = Path("./results/example")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "example_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_dir / 'example_results.json'}")
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()