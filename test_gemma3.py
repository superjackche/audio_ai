"""
测试 Gemma3 模型加载和基本功能
"""
import os
import sys
sys.path.append('.')

from models.model_manager import ModelManager
from config.settings import MODEL_CONFIG

def main():
    """测试主函数"""
    print("=== Gemma3 模型测试 ===")
    print(f"配置的模型: {MODEL_CONFIG['llm_model']}")
    print(f"设备: {MODEL_CONFIG['device']}")
    
    # 设置环境变量强制使用 Gemma3
    os.environ['MODEL_NAME'] = 'google/gemma-3-1b-it'
    
    # 创建模型管理器
    manager = ModelManager()
    
    # 只测试 LLM 加载（不加载 Whisper 以节省时间）
    print("\n--- 加载 LLM 模型 ---")
    llm_success = manager.load_llm_model()
    
    if llm_success:
        print("✅ LLM 模型加载成功")
        
        # 获取模型信息
        model_info = manager.get_model_info()
        print(f"模型信息: {model_info}")
        
        # 测试文本分析
        print("\n--- 测试文本分析 ---")
        test_text = "今天的课程内容很有意思，讲述了中国的发展历程。"
        
        try:
            result = manager.analyze_text_risk(test_text)
            print(f"分析结果: {result}")
        except Exception as e:
            print(f"分析失败: {e}")
            
    else:
        print("❌ LLM 模型加载失败")
        return False
    
    print("\n=== 测试完成 ===")
    return True

if __name__ == "__main__":
    main()
