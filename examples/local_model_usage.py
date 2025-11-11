#!/usr/bin/env python3
"""
示例：如何使用本地模型进行M-BEIR评估
"""

import os
import sys
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.dataloader import MBEIRDataLoader, create_sample_dataset
from src.models.retrieval_models import LocalModel, get_retrieval_model
from src.evaluation.metrics import calculate_metrics_batch


def download_model_to_local():
    """将模型下载到本地目录的示例函数"""
    from sentence_transformers import SentenceTransformer
    
    # 本地模型目录 - 使用更小的模型
    local_model_dir = Path("../models/all-MiniLM-L6-v2")
    local_model_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"下载模型到本地目录: {local_model_dir}")
    print("使用更小的模型: all-MiniLM-L6-v2 (约80MB)")
    
    # 下载模型到本地目录 - 使用更小的模型
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    model.save(str(local_model_dir))
    
    print("模型下载完成！")
    return str(local_model_dir)


def load_local_model_example():
    """加载本地模型的示例"""
    print("=" * 60)
    print("本地模型加载示例")
    print("=" * 60)
    
    # 方法1：直接使用LocalModel类
    print("\\n方法1：直接使用LocalModel类")
    print("-" * 40)
    
    # 假设模型已经下载到本地
    model_path = "../models/all-MiniLM-L6-v2"
    
    if not os.path.exists(model_path):
        print("本地模型不存在，先下载模型...")
        model_path = download_model_to_local()
    
    # 创建本地模型实例
    local_model = LocalModel(
        model_path=model_path,
        device="cpu",
        model_type="sentence_transformer"
    )
    
    # 测试编码
    test_texts = ["这是一个测试文本", "这是另一个测试文本"]
    embeddings = local_model.encode_texts(test_texts)
    print(f"编码测试成功！嵌入维度: {embeddings.shape}")
    
    # 方法2：使用配置字典和工厂函数
    print("\\n方法2：使用配置字典和工厂函数")
    print("-" * 40)
    
    model_config = {
        'name': 'local-minilm-model',
        'type': 'local',
        'path': model_path,
        'local_model_type': 'sentence_transformer',
        'device': 'cpu'
    }
    
    model_from_factory = get_retrieval_model(model_config)
    
    # 测试编码
    embeddings2 = model_from_factory.encode_texts(test_texts)
    print(f"工厂方法编码测试成功！嵌入维度: {embeddings2.shape}")
    
    return local_model


def evaluate_with_local_model():
    """使用本地模型进行评估的完整示例"""
    print("\\n" + "=" * 60)
    print("使用本地模型进行评估")
    print("=" * 60)
    
    # 1. 准备数据
    data_dir = Path("../data")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    dataset_dir = data_dir / "sample"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建示例数据集
    create_sample_dataset(dataset_dir)
    
    # 2. 加载数据
    data_loader = MBEIRDataLoader(data_dir)
    dataset = data_loader.load_dataset("sample", "sample")
    
    # 提取文本内容
    queries = [doc['text'] for doc in dataset['queries'].values()]
    corpus = [doc['text'] for doc in dataset['corpus'].values()]
    
    # 创建qrels
    qrels = {}
    for query_id in dataset['queries'].keys():
        qrels[query_id] = dataset['qrels'].get(query_id, [])
    
    print(f"加载数据: {len(queries)} 个查询, {len(corpus)} 个文档")
    
    # 3. 加载本地模型
    model_path = "../models/all-MiniLM-L6-v2"
    if not os.path.exists(model_path):
        print("本地模型不存在，先下载模型...")
        model_path = download_model_to_local()
    
    model_config = {
        'name': 'local-minilm-model',
        'type': 'local',
        'path': model_path,
        'local_model_type': 'sentence_transformer',
        'device': 'cpu'
    }
    
    model = get_retrieval_model(model_config)
    
    # 4. 计算评估指标
    metrics_to_calculate = ['ndcg@3', 'recall@3', 'map']
    
    print("\\n计算评估指标...")
    results = calculate_metrics_batch(
        queries, 
        corpus, 
        qrels, 
        model, 
        metrics_to_calculate, 
        k=3
    )
    
    # 5. 显示结果
    print("\\n评估结果:")
    print("-" * 30)
    for metric, score in results.items():
        print(f"{metric}: {score:.4f}")
    
    return results


def main():
    """主函数"""
    print("M-BEIR 本地模型使用示例")
    
    try:
        # 示例1：加载本地模型
        model = load_local_model_example()
        
        # 示例2：使用本地模型进行评估
        results = evaluate_with_local_model()
        
        print("\\n" + "🎉 示例运行完成！")
        print("\\n使用说明:")
        print("1. 将预训练模型下载到本地目录")
        print("2. 使用LocalModel类或配置字典加载模型")
        print("3. 进行文本编码和评估")
        
    except Exception as e:
        print(f"\\n❌ 运行出错: {e}")
        print("\\n可能的原因:")
        print("1. 模型文件不存在")
        print("2. 模型文件格式不正确")
        print("3. 缺少必要的依赖包")


if __name__ == "__main__":
    main()