import json
import random
from typing import List, Dict, Any
from pathlib import Path
import pandas as pd
from config.settings import DATA_PATHS, RISK_KEYWORDS

class TrainingDataGenerator:
    """训练数据生成器"""
    
    def __init__(self):
        self.training_data = []
        self.validation_data = []
    
    def generate_synthetic_data(self, num_samples: int = 1000) -> None:
        """生成合成训练数据"""
        
        # 高风险样本模板
        high_risk_templates = [
            "我认为{政治制度}存在严重问题，需要进行{民主化}改革。",
            "当前的{治理模式}已经过时，应该学习西方的{民主制度}。",
            "这种{意识形态}压制了人民的{自由权利}。",
            "{政权}的合法性值得质疑，需要{政治改革}。",
            "我们应该支持{独立运动}，争取{自主权}。"
        ]
        
        # 中风险样本模板
        medium_risk_templates = [
            "不同{政治制度}各有优缺点，需要{比较分析}。",
            "{社会制度}的发展需要考虑{文化差异}。",
            "关于{价值观}问题，存在不同观点和{理论研究}。",
            "{治理模式}的效果需要{客观评估}。",
            "各国{政策建议}应该因地制宜。"
        ]
        
        # 低风险样本模板
        low_risk_templates = [
            "这是一个{学术问题}，需要深入{理论研究}。",
            "从{比较分析}的角度来看，不同观点都有道理。",
            "这个{政策建议}值得进一步讨论。",
            "让我们进行{客观分析}，避免主观判断。",
            "这属于{理论探讨}范畴，可以开放讨论。"
        ]
        
        # 无风险样本模板
        safe_templates = [
            "今天我们学习{数学公式}的推导过程。",
            "这个{科学实验}展示了{物理原理}。",
            "我们来分析{历史事件}的客观过程。",
            "这个{经济模型}可以帮助理解{市场机制}。",
            "让我们探讨{技术发展}对社会的影响。"
        ]
        
        data_samples = []
          # 生成各类样本
        for i in range(num_samples // 4):
            # 高风险样本
            template = random.choice(high_risk_templates)
            text = self._fill_template_placeholders(template, RISK_KEYWORDS["高风险"])
            
            data_samples.append({
                "text": text,
                "label": "高风险",
                "risk_score": random.randint(70, 100),
                "category": "political_risk"
            })
            
            # 中风险样本
            template = random.choice(medium_risk_templates)
            text = self._fill_template_placeholders(template, RISK_KEYWORDS["中风险"])
            
            data_samples.append({
                "text": text,
                "label": "中风险", 
                "risk_score": random.randint(40, 69),
                "category": "moderate_risk"
            })
            
            # 低风险样本
            template = random.choice(low_risk_templates)
            text = self._fill_template_placeholders(template, RISK_KEYWORDS["低风险"])
            
            data_samples.append({
                "text": text,
                "label": "低风险",
                "risk_score": random.randint(20, 39),
                "category": "low_risk"
            })
            
            # 无风险样本
            template = random.choice(safe_templates)
            safe_keywords = ["学术研究", "科学方法", "客观分析", "理论探讨", "实证研究"]
            text = self._fill_template_placeholders(template, safe_keywords)
            
            data_samples.append({
                "text": text,
                "label": "无风险",
                "risk_score": random.randint(0, 19),
                "category": "safe"
            })
          # 打乱数据
        random.shuffle(data_samples)
        
        # 分割训练集和验证集
        split_idx = int(len(data_samples) * 0.8)
        self.training_data = data_samples[:split_idx]
        self.validation_data = data_samples[split_idx:]
        
        print(f"生成训练数据: {len(self.training_data)} 条")
        print(f"生成验证数据: {len(self.validation_data)} 条")
    
    def _fill_template_placeholders(self, template: str, keyword_pool: List[str]) -> str:
        """填充模板中的占位符"""
        import re
        placeholders = re.findall(r'\{([^}]+)\}', template)
        
        filled_text = template
        for placeholder in placeholders:
            # 从关键词池中随机选择替换词
            replacement = random.choice(keyword_pool)
            filled_text = filled_text.replace(f"{{{placeholder}}}", replacement)
        
        return filled_text
    
    def add_real_world_samples(self) -> None:
        """添加真实世界样本（模拟）"""
        real_samples = [
            {
                "text": "今天我们讨论中西方教育理念的差异，需要客观分析各自优势。",
                "label": "无风险",
                "risk_score": 10,
                "category": "safe"
            },
            {
                "text": "不同国家的发展模式反映了文化和历史背景的差异。",
                "label": "低风险", 
                "risk_score": 25,
                "category": "low_risk"
            },
            {
                "text": "我们应该批判性地思考现有制度的不足之处。",
                "label": "中风险",
                "risk_score": 45,
                "category": "moderate_risk"
            },
            {
                "text": "某些政治制度明显优于其他制度，应该推广普及。",
                "label": "高风险",
                "risk_score": 75,
                "category": "political_risk"
            }
        ]
        
        self.training_data.extend(real_samples)
        print(f"添加真实样本: {len(real_samples)} 条")
    
    def save_datasets(self) -> None:
        """保存数据集到文件"""
        # 保存训练集
        train_df = pd.DataFrame(self.training_data)
        train_path = DATA_PATHS["data_dir"] / "train_dataset.csv"
        train_df.to_csv(train_path, index=False, encoding="utf-8")
        
        # 保存验证集
        val_df = pd.DataFrame(self.validation_data)
        val_path = DATA_PATHS["data_dir"] / "validation_dataset.csv"
        val_df.to_csv(val_path, index=False, encoding="utf-8")
        
        # 保存JSON格式
        with open(DATA_PATHS["data_dir"] / "train_dataset.json", "w", encoding="utf-8") as f:
            json.dump(self.training_data, f, ensure_ascii=False, indent=2)
        
        with open(DATA_PATHS["data_dir"] / "validation_dataset.json", "w", encoding="utf-8") as f:
            json.dump(self.validation_data, f, ensure_ascii=False, indent=2)
        
        print(f"数据集已保存到: {DATA_PATHS['data_dir']}")
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """获取数据集统计信息"""
        train_labels = [item["label"] for item in self.training_data]
        val_labels = [item["label"] for item in self.validation_data]
        
        from collections import Counter
        train_dist = Counter(train_labels)
        val_dist = Counter(val_labels)
        
        return {
            "training_size": len(self.training_data),
            "validation_size": len(self.validation_data),
            "training_distribution": dict(train_dist),
            "validation_distribution": dict(val_dist),
            "total_samples": len(self.training_data) + len(self.validation_data)
        }

def create_training_data():
    """创建训练数据的主函数"""
    generator = TrainingDataGenerator()
    
    print("开始生成训练数据...")
    generator.generate_synthetic_data(num_samples=2000)
    generator.add_real_world_samples()
    generator.save_datasets()
    
    stats = generator.get_dataset_statistics()
    print("\n数据集统计信息:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    return generator

if __name__ == "__main__":
    create_training_data()
