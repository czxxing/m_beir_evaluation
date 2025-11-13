#!/usr/bin/env python3
"""
测试Hugging Face M-BEIR数据集加载

这个脚本专门测试从Hugging Face加载M-BEIR数据集的功能。
"""

import os
import sys
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_hf_dataset_loading():
    """测试Hugging Face数据集加载"""
    logger.info("=" * 60)
    logger.info("测试Hugging Face M-BEIR数据集加载")
    logger.info("=" * 60)
    
    # 检查datasets包是否安装
    try:
        import datasets
        logger.info("✓ datasets包已安装")
    except ImportError:
        logger.error("✗ datasets包未安装，请运行: pip install datasets")
        return False
    
    # 尝试加载数据集信息
    try:
        from datasets import get_dataset_config_names
        
        logger.info("获取可用的数据集配置...")
        configs = get_dataset_config_names("TIGER-Lab/M-BEIR")
        logger.info(f"可用的数据集配置: {configs}")
        
        # 显示前几个配置的详细信息
        for config in configs[:5]:
            logger.info(f"  - {config}")
        
        if len(configs) > 5:
            logger.info(f"  ... 还有 {len(configs) - 5} 个配置")
        
    except Exception as e:
        logger.error(f"获取数据集配置失败: {e}")
        logger.info("可能是网络连接问题，请检查网络连接")
        return False
    
    # 尝试加载一个小数据集进行测试
    try:
        logger.info("尝试加载mscoco数据集...")
        from datasets import load_dataset
        
        # 只加载一小部分数据进行测试
        dataset = load_dataset("TIGER-Lab/M-BEIR", "mscoco", split="test[:10]")
        
        logger.info(f"✓ 成功加载数据集，样本数量: {len(dataset)}")
        
        # 显示数据集结构
        if len(dataset) > 0:
            first_sample = dataset[0]
            logger.info("数据集样本结构:")
            for key, value in first_sample.items():
                if isinstance(value, str):
                    preview = value[:100] + "..." if len(value) > 100 else value
                    logger.info(f"  {key}: {preview}")
                elif isinstance(value, list):
                    logger.info(f"  {key}: 列表，长度 {len(value)}")
                else:
                    logger.info(f"  {key}: {type(value)}")
        
        return True
        
    except Exception as e:
        logger.error(f"加载数据集失败: {e}")
        logger.info("可能是网络连接问题或数据集不存在")
        return False


def test_local_dataset():
    """测试本地数据集加载作为备选方案"""
    logger.info("=" * 60)
    logger.info("测试本地示例数据集")
    logger.info("=" * 60)
    
    try:
        from src.data.dataloader import MBEIRDataLoader
        
        data_loader = MBEIRDataLoader("data/sample")
        dataset = data_loader.load_dataset("sample")
        
        logger.info(f"✓ 本地数据集加载成功")
        logger.info(f"  查询数量: {len(dataset['queries'])}")
        logger.info(f"  文档数量: {len(dataset['corpus'])}")
        logger.info(f"  相关性判断: {len(dataset['qrels'])}")
        
        # 显示示例
        logger.info("查询示例:")
        for qid, query in list(dataset['queries'].items())[:2]:
            logger.info(f"  {qid}: {query.get('text', '')}")
        
        logger.info("文档示例:")
        for did, doc in list(dataset['corpus'].items())[:2]:
            logger.info(f"  {did}: {doc.get('text', '')}")
        
        return True
        
    except Exception as e:
        logger.error(f"本地数据集加载失败: {e}")
        return False


def main():
    """主函数"""
    logger.info("M-BEIR数据集验证工具")
    logger.info("=" * 60)
    
    # 测试Hugging Face数据集加载
    hf_success = test_hf_dataset_loading()
    
    if not hf_success:
        logger.info("-" * 40)
        logger.info("Hugging Face数据集加载失败，尝试本地数据集...")
        local_success = test_local_dataset()
        
        if local_success:
            logger.info("✓ 可以使用本地示例数据进行开发和测试")
        else:
            logger.error("✗ 所有数据集加载方式都失败了")
    else:
        logger.info("✓ Hugging Face数据集加载成功，可以进行完整验证")
    
    logger.info("=" * 60)
    logger.info("验证完成")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()