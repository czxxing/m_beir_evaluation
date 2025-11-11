#!/usr/bin/env python3
"""
Main script for running M-BEIR evaluation experiments.
"""

import argparse
import yaml
import logging
from pathlib import Path

from src.evaluation.evaluator import Evaluator
from src.utils.config import load_config


def setup_logging(log_level="INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main():
    parser = argparse.ArgumentParser(description="Run M-BEIR evaluation")
    parser.add_argument("--config", type=str, required=True, 
                       help="Path to configuration file")
    parser.add_argument("--model", type=str, default=None,
                       help="Specific model to evaluate")
    parser.add_argument("--dataset", type=str, default=None,
                       help="Specific dataset to evaluate on")
    parser.add_argument("--output", type=str, default=None,
                       help="Output directory for results")
    parser.add_argument("--log-level", type=str, default="INFO",
                       help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.model:
        config['model']['name'] = args.model
    if args.dataset:
        config['dataset']['name'] = args.dataset
    if args.output:
        config['output_dir'] = args.output
    
    logger.info(f"Starting M-BEIR evaluation with config: {args.config}")
    
    # Initialize evaluator
    evaluator = Evaluator(config)
    
    # Run evaluation
    results = evaluator.run()
    
    # Save results
    evaluator.save_results(results)
    
    logger.info("Evaluation completed successfully!")


if __name__ == "__main__":
    main()