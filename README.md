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

## 本地模型使用

本项目支持从本地路径加载预训练模型，适用于离线环境或需要版本控制的场景。

### 方法1: 直接使用LocalModel类

```python
from src.models.retrieval_models import LocalModel

# 加载本地模型
local_model = LocalModel(
    model_path="./models/all-mpnet-base-v2",
    device="cpu",
    model_type="sentence_transformer"
)

# 使用模型编码文本
embeddings = local_model.encode_texts(["测试文本1", "测试文本2"])
```

### 方法2: 使用配置字典

```python
from src.models.retrieval_models import get_retrieval_model

model_config = {
    'name': 'local-model',
    'type': 'local',
    'path': './models/all-mpnet-base-v2',
    'local_model_type': 'sentence_transformer',
    'device': 'cpu'
}

model = get_retrieval_model(model_config)
```

### 支持的模型类型

- **sentence_transformer**: Sentence Transformer格式的模型
- **huggingface**: HuggingFace Transformers格式的模型

### 示例代码

运行示例代码查看完整使用方法：
```bash
python examples/local_model_usage.py
```

## Features

- Support for multiple retrieval models
- Comprehensive evaluation metrics
- Multi-modal data processing
- Configurable experiment settings
- Result visualization and analysis

## License

MIT License