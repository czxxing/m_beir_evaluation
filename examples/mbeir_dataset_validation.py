#!/usr/bin/env python3
"""
M-BEIR数据集验证示例

这个脚本演示如何从Hugging Face加载M-BEIR数据集，
并使用本地模型进行检索评估。

数据集地址: https://huggingface.co/datasets/TIGER-Lab/M-BEIR
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.dataloader import MBEIRDataLoader
from src.models.retrieval_models import LocalModel
from src.evaluation.metrics import calculate_metrics

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_mbeir_dataset_from_hf(dataset_name: str, split: str = "test") -> Dict:
    """
    从Hugging Face加载M-BEIR数据集
    
    Args:
        dataset_name: 数据集名称
        split: 数据分割（train, validation, test）
        
    Returns:
        包含queries, corpus, qrels的字典
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("请先安装datasets包: pip install datasets")
        sys.exit(1)
    
    logger.info(f"正在从Hugging Face加载数据集: {dataset_name}, 分割: {split}")
    
    # 加载数据集
    dataset = load_dataset("TIGER-Lab/M-BEIR", dataset_name, split=split)
    
    # 转换数据格式
    queries = {}
    corpus = {}
    qrels = {}
    
    # 处理查询
    for i, item in enumerate(dataset):
        query_id = f"q{i+1}"
        
        # 构建查询对象
        query = {
            "_id": query_id,
            "text": item.get("query", ""),
            "modality": "text"
        }
        
        # 如果有图像信息
        if "query_image" in item and item["query_image"]:
            query["image"] = item["query_image"]
            query["modality"] = "multimodal"
        
        queries[query_id] = query
        
        # 处理文档
        if "corpus" in item:
            for j, doc in enumerate(item["corpus"]):
                doc_id = f"d{j+1}"
                
                document = {
                    "_id": doc_id,
                    "text": doc.get("text", ""),
                    "modality": "text"
                }
                
                # 如果有图像信息
                if "image" in doc and doc["image"]:
                    document["image"] = doc["image"]
                    document["modality"] = "multimodal"
                
                corpus[doc_id] = document
        
        # 处理相关性判断
        if "relevant_docs" in item:
            for doc_id in item["relevant_docs"]:
                if query_id not in qrels:
                    qrels[query_id] = []
                qrels[query_id].append(f"d{doc_id}")
    
    logger.info(f"加载完成: {len(queries)} 个查询, {len(corpus)} 个文档, {len(qrels)} 个相关性判断")
    
    return {
        'queries': queries,
        'corpus': corpus,
        'qrels': qrels
    }


def extract_text_content(documents: Dict[str, Dict]) -> List[str]:
    """从文档中提取文本内容"""
    texts = []
    for doc_id, doc in documents.items():
        # 尝试找到文本字段
        text_fields = ['text', 'content', 'title', 'description', 'query']
        for field in text_fields:
            if field in doc and doc[field]:
                texts.append(str(doc[field]))
                break
        else:
            # 如果没有找到文本字段，使用第一个字符串字段
            for value in doc.values():
                if isinstance(value, str) and value.strip():
                    texts.append(value)
                    break
            else:
                texts.append("")
    return texts


def simple_retrieval(queries_text: List[str], corpus_text: List[str], 
                    model, top_k: int = 10) -> Dict[str, List[str]]:
    """
    简单的检索实现
    
    Args:
        queries_text: 查询文本列表
        corpus_text: 文档文本列表
        model: 检索模型
        top_k: 返回前k个结果
        
    Returns:
        检索结果字典
    """
    logger.info("开始编码查询和文档...")
    
    # 编码查询
    query_embeddings = model.encode_texts(queries_text, batch_size=32)
    logger.info(f"查询编码完成，维度: {query_embeddings.shape}")
    
    # 编码文档
    corpus_embeddings = model.encode_texts(corpus_text, batch_size=32)
    logger.info(f"文档编码完成，维度: {corpus_embeddings.shape}")
    
    # 计算相似度
    logger.info("计算相似度...")
    similarities = []
    
    # 分批处理以避免内存问题
    batch_size = 100
    for i in range(0, len(query_embeddings), batch_size):
        batch_queries = query_embeddings[i:i+batch_size]
        
        # 计算余弦相似度
        from sklearn.metrics.pairwise import cosine_similarity
        batch_similarities = cosine_similarity(batch_queries, corpus_embeddings)
        similarities.extend(batch_similarities)
    
    similarities = np.array(similarities)
    logger.info(f"相似度矩阵维度: {similarities.shape}")
    
    # 获取top-k结果
    retrieval_results = {}
    for i, query_similarities in enumerate(similarities):
        query_id = f"q{i+1}"
        
        # 获取相似度最高的k个文档索引
        top_indices = np.argsort(query_similarities)[::-1][:top_k]
        
        # 转换为文档ID
        retrieved_docs = [f"d{idx+1}" for idx in top_indices]
        retrieval_results[query_id] = retrieved_docs
    
    logger.info(f"检索完成，共处理 {len(retrieval_results)} 个查询")
    return retrieval_results


def validate_dataset(dataset_name: str = "mscoco", split: str = "test", 
                     model_path: str = "./models/all-MiniLM-L6-v2",
                     top_k: int = 10):
    """
    验证M-BEIR数据集
    
    Args:
        dataset_name: 数据集名称
        split: 数据分割
        model_path: 模型路径
        top_k: 检索top-k结果
    """
    logger.info("=" * 60)
    logger.info(f"开始验证M-BEIR数据集: {dataset_name}")
    logger.info("=" * 60)
    
    # 1. 加载数据集
    try:
        dataset = load_mbeir_dataset_from_hf(dataset_name, split)
    except Exception as e:
        logger.error(f"加载数据集失败: {e}")
        logger.info("尝试使用本地示例数据...")
        
        # 使用本地示例数据
        data_loader = MBEIRDataLoader("data/sample")
        dataset = data_loader.load_dataset("sample")
    
    # 2. 加载模型
    logger.info(f"加载模型: {model_path}")
    try:
        model = LocalModel(model_path)
        logger.info("模型加载成功")
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        logger.info("请确保模型已下载，路径: ./models/all-MiniLM-L6-v2")
        return
    
    # 3. 提取文本内容
    queries_text = extract_text_content(dataset['queries'])
    corpus_text = extract_text_content(dataset['corpus'])
    
    logger.info(f"文本提取完成: {len(queries_text)} 个查询, {len(corpus_text)} 个文档")
    
    # 4. 执行检索
    retrieval_results = simple_retrieval(queries_text, corpus_text, model, top_k)
    
    # 5. 计算评估指标
    logger.info("计算评估指标...")
    metrics = calculate_metrics(
        retrieval_results,
        dataset['qrels'],
        metrics=['ndcg@3', 'ndcg@5', 'ndcg@10', 'recall@3', 'recall@5', 'recall@10', 'map']
    )
    
    # 6. 显示结果
    logger.info("=" * 60)
    logger.info("评估结果:")
    logger.info("=" * 60)
    
    for metric_name, metric_value in metrics.items():
        logger.info(f"{metric_name}: {metric_value:.4f}")
    
    # 7. 保存结果
    results_dir = project_root / "results" / dataset_name
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'dataset': dataset_name,
        'split': split,
        'metrics': metrics,
        'query_count': len(dataset['queries']),
        'corpus_size': len(dataset['corpus']),
        'qrels_count': len(dataset['qrels'])
    }
    
    results_file = results_dir / f"validation_results_{split}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"结果已保存到: {results_file}")
    
    # 8. 显示数据统计
    logger.info("=" * 60)
    logger.info("数据集统计:")
    logger.info("=" * 60)
    logger.info(f"查询数量: {len(dataset['queries'])}")
    logger.info(f"文档数量: {len(dataset['corpus'])}")
    logger.info(f"相关性判断数量: {len(dataset['qrels'])}")
    
    # 显示前几个查询和文档示例
    logger.info("-" * 40)
    logger.info("查询示例:")
    for i, (qid, query) in enumerate(list(dataset['queries'].items())[:3]):
        text_preview = query.get('text', '')[:50] + "..." if len(query.get('text', '')) > 50 else query.get('text', '')
        logger.info(f"  {qid}: {text_preview}")
    
    logger.info("-" * 40)
    logger.info("文档示例:")
    for i, (did, doc) in enumerate(list(dataset['corpus'].items())[:3]):
        text_preview = doc.get('text', '')[:50] + "..." if len(doc.get('text', '')) > 50 else doc.get('text', '')
        logger.info(f"  {did}: {text_preview}")
    
    logger.info("=" * 60)
    logger.info("验证完成!")
    logger.info("=" * 60)
    
    return results


def main():
    """主函数"""
    # 可用的M-BEIR数据集列表
    available_datasets = [
        "mscoco", "flickr30k", "visualnews", "nuswide", "mirflickr",
        "espgame", "sbs16", "tv2016", "tv2017", "tv2018"
    ]
    
    logger.info("可用的M-BEIR数据集:")
    for i, dataset in enumerate(available_datasets, 1):
        logger.info(f"  {i}. {dataset}")
    
    # 验证示例数据集
    try:
        # 验证MSCOCO数据集（图像描述检索）
        results = validate_dataset(
            dataset_name="mscoco",
            split="test",
            top_k=10
        )
        
        # 验证Flickr30K数据集
        results_flickr = validate_dataset(
            dataset_name="flickr30k", 
            split="test",
            top_k=10
        )
        
    except Exception as e:
        logger.error(f"验证过程中出现错误: {e}")
        logger.info("尝试使用本地示例数据进行验证...")
        
        # 使用本地示例数据
        results = validate_dataset(
            dataset_name="sample",
            split="test",
            top_k=5
        )


if __name__ == "__main__":
    main()