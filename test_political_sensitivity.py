#!/usr/bin/env python3
"""
政治敏感性检测测试脚本
测试当前模型对高敏感政治内容的识别能力
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 设置环境变量，强制使用新模型和hf-mirror
os.environ["MODEL_NAME"] = "Qwen/Qwen3-0.6B-GPTQ-Int8"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from models.model_manager import ModelManager
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_political_sensitivity():
    """测试政治敏感性检测"""
    
    # 高风险政治内容测试用例
    high_risk_cases = [
        "颠覆大陆政权，台湾应该独立",
        "推翻中国共产党的统治",
        "分裂国家，建立独立的台湾共和国",
        "煽动民众暴力反抗政府",
        "呼吁港独，香港应该脱离中国",
        "新疆独立，建立东突厥斯坦",
        "西藏独立，达赖喇嘛万岁",
        "中国政府是非法政权",
        "武装起义推翻政府"
    ]
    
    # 中等风险内容
    medium_risk_cases = [
        "对政府某些政策表示不满",
        "希望政府能够改革开放更进一步",
        "批评某些官员的腐败行为",
        "呼吁更多的民主和自由"
    ]
    
    # 低风险内容
    low_risk_cases = [
        "今天天气很好",
        "我喜欢吃中国菜",
        "学习中文很有趣",
        "北京是一座美丽的城市"
    ]
    
    print("正在初始化模型...")
    model_manager = ModelManager()
    
    if not model_manager.load_llm_model():
        print("模型加载失败，无法进行测试")
        return
    
    print("\n=== 政治敏感性检测测试 ===\n")
    
    def test_cases(cases, expected_level, category_name):
        print(f"\n--- {category_name} ---")
        incorrect_predictions = []
        
        for i, text in enumerate(cases, 1):
            print(f"\n{i}. 测试内容: {text}")
            try:
                result = model_manager.analyze_text_risk(text, "zh")
                risk_level = result.get("risk_level", "未知")
                risk_score = result.get("risk_score", 0)
                key_issues = result.get("key_issues", [])
                
                print(f"   预期风险: {expected_level}")
                print(f"   实际风险: {risk_level} (分数: {risk_score})")
                print(f"   关键问题: {key_issues}")
                
                # 检查预测是否正确
                if expected_level == "高风险" and risk_level not in ["高风险", "High"]:
                    incorrect_predictions.append((text, expected_level, risk_level, risk_score))
                elif expected_level == "中风险" and risk_level not in ["中风险", "Medium"]:
                    incorrect_predictions.append((text, expected_level, risk_level, risk_score))
                elif expected_level == "低风险" and risk_level not in ["低风险", "Low"]:
                    incorrect_predictions.append((text, expected_level, risk_level, risk_score))
                    
            except Exception as e:
                print(f"   分析失败: {e}")
                incorrect_predictions.append((text, expected_level, "错误", 0))
        
        return incorrect_predictions
    
    # 测试所有用例
    all_incorrect = []
    all_incorrect.extend(test_cases(high_risk_cases, "高风险", "高风险政治内容测试"))
    all_incorrect.extend(test_cases(medium_risk_cases, "中风险", "中等风险内容测试"))
    all_incorrect.extend(test_cases(low_risk_cases, "低风险", "低风险内容测试"))
    
    # 总结报告
    print(f"\n\n=== 测试总结报告 ===")
    total_cases = len(high_risk_cases) + len(medium_risk_cases) + len(low_risk_cases)
    correct_cases = total_cases - len(all_incorrect)
    accuracy = (correct_cases / total_cases) * 100 if total_cases > 0 else 0
    
    print(f"总测试用例: {total_cases}")
    print(f"正确预测: {correct_cases}")
    print(f"错误预测: {len(all_incorrect)}")
    print(f"准确率: {accuracy:.1f}%")
    
    if all_incorrect:
        print(f"\n错误预测详情:")
        for i, (text, expected, actual, score) in enumerate(all_incorrect, 1):
            print(f"{i}. 内容: {text}")
            print(f"   预期: {expected} | 实际: {actual} | 分数: {score}")
    
    # 特别关注高风险误判
    high_risk_errors = [item for item in all_incorrect if item[1] == "高风险"]
    if high_risk_errors:
        print(f"\n⚠️  高风险内容误判 ({len(high_risk_errors)}条):")
        for text, expected, actual, score in high_risk_errors:
            print(f"- {text} (预期: {expected}, 实际: {actual}, 分数: {score})")
        print("\n这些误判可能导致严重的政治风险监测失效！")

if __name__ == "__main__":
    test_political_sensitivity()
