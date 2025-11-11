#!/usr/bin/env python3
"""
Test script to verify project structure and basic functionality.
"""

import os
import sys
from pathlib import Path

def test_project_structure():
    """Test if all required files and directories exist."""
    
    required_files = [
        "README.md",
        "requirements.txt", 
        "requirements_minimal.txt",
        "run_evaluation.py",
        "example_usage.py",
        "setup.py",
        "install.sh",
        ".gitignore",
        "configs/default.yaml"
    ]
    
    required_dirs = [
        "src",
        "src/data",
        "src/models", 
        "src/evaluation",
        "src/utils",
        "tests",
        "configs"
    ]
    
    print("Testing project structure...")
    print("=" * 50)
    
    # Test directories
    print("\nChecking directories:")
    all_dirs_ok = True
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✓ {dir_path}")
        else:
            print(f"✗ {dir_path} - MISSING")
            all_dirs_ok = False
    
    # Test files
    print("\nChecking files:")
    all_files_ok = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} - MISSING")
            all_files_ok = False
    
    # Test Python imports
    print("\nTesting Python imports:")
    sys.path.insert(0, 'src')
    
    try:
        from src.utils.config import load_config, validate_config
        print("✓ src.utils.config")
    except ImportError as e:
        print(f"✗ src.utils.config - {e}")
        all_files_ok = False
    
    try:
        from src.data.dataloader import MBEIRDataLoader
        print("✓ src.data.dataloader")
    except ImportError as e:
        print(f"✗ src.data.dataloader - {e}")
        all_files_ok = False
    
    try:
        from src.evaluation.metrics import calculate_metrics
        print("✓ src.evaluation.metrics")
    except ImportError as e:
        print(f"✗ src.evaluation.metrics - {e}")
        all_files_ok = False
    
    # Test configuration loading
    print("\nTesting configuration loading:")
    try:
        # Create a minimal config for testing
        test_config = {
            'experiment': {'seed': 42},
            'model': {'name': 'test-model'},
            'dataset': {'name': 'test-dataset'},
            'output': {'output_dir': './results'}
        }
        validated = validate_config(test_config)
        print("✓ Configuration validation")
    except Exception as e:
        print(f"✗ Configuration validation - {e}")
        all_files_ok = False
    
    print("\n" + "=" * 50)
    if all_dirs_ok and all_files_ok:
        print("✓ Project structure is complete!")
        return True
    else:
        print("✗ Project structure has issues.")
        return False

def create_sample_data():
    """Create sample data for testing."""
    print("\nCreating sample data...")
    
    data_dir = Path("data/sample")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Import and create sample dataset
    sys.path.insert(0, 'src')
    from src.data.dataloader import create_sample_dataset
    
    try:
        create_sample_dataset(data_dir)
        print("✓ Sample dataset created")
        
        # Verify the files were created
        files_created = [
            "sample_queries.jsonl",
            "sample_corpus.jsonl", 
            "sample_qrels.jsonl"
        ]
        
        for file_name in files_created:
            file_path = data_dir / file_name
            if file_path.exists():
                print(f"✓ {file_name}")
            else:
                print(f"✗ {file_name}")
                
    except Exception as e:
        print(f"✗ Failed to create sample data: {e}")

def main():
    """Run all tests."""
    print("M-BEIR Evaluation Project Structure Test")
    print("=" * 60)
    
    # Test project structure
    structure_ok = test_project_structure()
    
    # Create sample data
    create_sample_data()
    
    print("\n" + "=" * 60)
    if structure_ok:
        print("🎉 Project setup completed successfully!")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements_minimal.txt")
        print("2. Run example: python example_usage.py")
        print("3. Run tests: python -m pytest tests/")
    else:
        print("❌ Project setup has issues. Please check the errors above.")

if __name__ == "__main__":
    main()