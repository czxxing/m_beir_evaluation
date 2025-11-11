"""
Configuration loading and validation utilities.
"""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary containing configuration
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return validate_config(config)


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate configuration and set default values.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Validated configuration
    """
    # Set default values if not provided
    defaults = {
        'experiment': {
            'seed': 42
        },
        'model': {
            'device': 'cpu',  # Simplified for testing
            'batch_size': 32
        },
        'output': {
            'output_dir': './results'
        }
    }
    
    # Merge with defaults
    for section, default_values in defaults.items():
        if section not in config:
            config[section] = {}
        for key, value in default_values.items():
            if key not in config[section]:
                config[section][key] = value
    
    return config


def save_config(config: Dict[str, Any], output_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        output_path: Path to save configuration
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)