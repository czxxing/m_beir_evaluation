# M-BEIR Evaluation Project

This project provides a comprehensive framework for evaluating retrieval models using the M-BEIR (Multi-modal BEIR) benchmark.

## Project Structure

```
├── src/                    # Source code
│   ├── data/              # Data loading and preprocessing
│   ├── models/            # Model implementations
│   ├── evaluation/        # Evaluation metrics and scripts
│   └── utils/             # Utility functions
├── configs/               # Configuration files
├── data/                  # Data storage
├── results/               # Evaluation results
├── tests/                 # Unit tests
├── requirements.txt       # Python dependencies
└── run_evaluation.py      # Main evaluation script
```

## Installation

1. Create and activate virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main evaluation script:
```bash
python run_evaluation.py --config configs/default.yaml
```

## Features

- Support for multiple retrieval models
- Comprehensive evaluation metrics
- Multi-modal data processing
- Configurable experiment settings
- Result visualization and analysis

## License

MIT License