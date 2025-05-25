#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-0.6B 模型测试脚本
测试模型加载和基本推理功能
"""

import os
import sys
import torch
import logging
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

# 设置环境变量
os.environ['MODEL_NAME'] = 'Qwen/Qwen3-0.6B'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HUGGINGFACE_HUB_CACHE'] = './models/cache'
os.environ['HF_HOME'] = './models/cache'

from models.model_manager import ModelManager

def test_qwen3_model():
    """测试 Qwen3-0.6B 模型"""
    print("🚀 开始测试 Qwen3-0.6B 模型...")
    print(f"当前模型配置: {os.environ.get('MODEL_NAME', 'unknown')}")
    
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    try:
        # 初始化模型管理器
        model_manager = ModelManager()
        
        print(f"\n📱 设备信息: {model_manager.device}")
        print(f"CUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU 数量: {torch.cuda.device_count()}")
            print(f"当前 GPU: {torch.cuda.get_device_name()}")
        
        print("\n🔄 正在加载 Qwen3-0.6B 模型...")
        
        # 只测试 LLM 模型加载
        success = model_manager.load_llm_model()
        
        if success:
            print("✅ Qwen3-0.6B 模型加载成功！")
            
            # 测试基本推理
            print("\n🧪 测试模型推理...")
            test_texts = [
                "今天天气很好",
                "我们需要讨论一下教育政策",
                "这篇文章提到了一些争议性观点"
            ]
            
            for i, text in enumerate(test_texts, 1):
                print(f"\n--- 测试 {i}: {text} ---")
                try:
                    result = model_manager.analyze_text_risk(text)
                    print(f"风险等级: {result.get('risk_level', 'N/A')}")
                    print(f"风险分数: {result.get('risk_score', 'N/A')}")
                    print(f"分析结果: {result.get('detailed_analysis', 'N/A')[:100]}...")
                except Exception as e:
                    print(f"❌ 推理失败: {e}")
            
        else:
            print("❌ Qwen3-0.6B 模型加载失败")
            return False
            
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n🎉 Qwen3-0.6B 模型测试完成！")
    return True

if __name__ == "__main__":
    test_qwen3_model()
