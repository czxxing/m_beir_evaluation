#!/usr/bin/env python3
"""
测试模型下载功能
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_model_download():
    """测试模型下载功能"""
    from sentence_transformers import SentenceTransformer
    
    # 本地模型目录
    local_model_dir = Path("./models/all-MiniLM-L6-v2")
    local_model_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"目标下载目录: {local_model_dir}")
    print(f"目录是否存在: {local_model_dir.exists()}")
    print(f"目录是否为空: {len(list(local_model_dir.iterdir())) == 0}")
    
    # 检查是否已经有模型文件
    model_files = list(local_model_dir.glob("*"))
    if model_files:
        print(f"发现已有模型文件: {model_files}")
        return True
    
    print("开始下载模型...")
    
    try:
        # 下载模型到本地目录
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        print("模型加载成功，开始保存到本地...")
        
        model.save(str(local_model_dir))
        print("模型保存完成！")
        
        # 检查保存的文件
        saved_files = list(local_model_dir.glob("*"))
        print(f"保存的文件列表: {saved_files}")
        
        return True
        
    except Exception as e:
        print(f"模型下载失败: {e}")
        return False

def main():
    """主函数"""
    print("测试模型下载功能")
    print("=" * 50)
    
    success = test_model_download()
    
    if success:
        print("\n✅ 模型下载测试成功！")
    else:
        print("\n❌ 模型下载测试失败！")

if __name__ == "__main__":
    main()