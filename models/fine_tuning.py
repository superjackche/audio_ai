"""
政治风险检测模型微调脚本
使用LoRA技术对ChatGLM3-6B进行微调
"""

import os
import json
import torch
import pandas as pd
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, List, Any

from transformers import (
    AutoTokenizer, 
    AutoModel, 
    TrainingArguments, 
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel
)
from datasets import Dataset

from config.settings import MODEL_CONFIG, DATA_PATHS

class PoliticalRiskModelTrainer:
    """政治风险检测模型训练器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.tokenizer = None
        self.peft_model = None
        
    def load_base_model(self):
        """加载基础模型"""
        self.logger.info("正在加载基础模型...")
        
        try:
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                MODEL_CONFIG['llm_model'],
                trust_remote_code=True,
                cache_dir=str(DATA_PATHS['models_dir'])
            )
            
            # 加载模型
            self.model = AutoModel.from_pretrained(
                MODEL_CONFIG['llm_model'],
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto",
                cache_dir=str(DATA_PATHS['models_dir'])
            )
            
            # 设置pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.logger.info("基础模型加载成功")
            return True
            
        except Exception as e:
            self.logger.error(f"基础模型加载失败: {e}")
            return False
    
    def setup_lora_config(self):
        """配置LoRA参数"""
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,  # rank
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
        )
        
        self.peft_model = get_peft_model(self.model, lora_config)
        self.peft_model.print_trainable_parameters()
        
        self.logger.info("LoRA配置完成")
        return lora_config
    
    def prepare_dataset(self):
        """准备训练数据集"""
        self.logger.info("正在准备数据集...")
        
        try:
            # 加载训练数据
            train_file = DATA_PATHS["data_dir"] / "train_dataset.json"
            val_file = DATA_PATHS["data_dir"] / "validation_dataset.json"
            
            with open(train_file, 'r', encoding='utf-8') as f:
                train_data = json.load(f)
            
            with open(val_file, 'r', encoding='utf-8') as f:
                val_data = json.load(f)
            
            # 转换为训练格式
            train_dataset = self._convert_to_training_format(train_data)
            val_dataset = self._convert_to_training_format(val_data)
            
            # 创建Dataset对象
            train_dataset = Dataset.from_list(train_dataset)
            val_dataset = Dataset.from_list(val_dataset)
            
            # 预处理数据
            train_dataset = train_dataset.map(self._preprocess_function, batched=True)
            val_dataset = val_dataset.map(self._preprocess_function, batched=True)
            
            self.logger.info(f"训练集大小: {len(train_dataset)}")
            self.logger.info(f"验证集大小: {len(val_dataset)}")
            
            return train_dataset, val_dataset
            
        except Exception as e:
            self.logger.error(f"数据集准备失败: {e}")
            return None, None
    
    def _convert_to_training_format(self, data: List[Dict]) -> List[Dict]:
        """将数据转换为训练格式"""
        formatted_data = []
        
        for item in data:
            # 构建输入文本
            input_text = f"请分析以下文本的政治风险等级：{item['text']}"
            
            # 构建输出文本
            output_text = f"风险等级：{item['label']}，风险分数：{item['risk_score']}，类别：{item['category']}"
            
            formatted_data.append({
                "input_text": input_text,
                "target_text": output_text
            })
        
        return formatted_data
    
    def _preprocess_function(self, examples):
        """预处理函数"""
        inputs = examples["input_text"]
        targets = examples["target_text"]
        
        # 构建完整的对话格式
        conversations = []
        for inp, tgt in zip(inputs, targets):
            conversation = f"<|im_start|>user\n{inp}<|im_end|>\n<|im_start|>assistant\n{tgt}<|im_end|>"
            conversations.append(conversation)
        
        # 编码
        model_inputs = self.tokenizer(
            conversations,
            max_length=512,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        
        # 设置labels
        model_inputs["labels"] = model_inputs["input_ids"].clone()
        
        return model_inputs
    
    def train_model(self, train_dataset, val_dataset, output_dir: str = None):
        """训练模型"""
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = str(DATA_PATHS["models_dir"] / f"fine_tuned_model_{timestamp}")
        
        self.logger.info(f"开始训练，输出目录: {output_dir}")
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            eval_steps=100,
            save_steps=200,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=None,  # 禁用wandb等
            dataloader_pin_memory=False,
            fp16=torch.cuda.is_available(),
            remove_unused_columns=False,
        )
        
        # 数据收集器
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.peft_model,
            padding=True
        )
        
        # 创建训练器
        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # 开始训练
        try:
            self.logger.info("开始训练...")
            trainer.train()
            
            # 保存模型
            trainer.save_model()
            self.tokenizer.save_pretrained(output_dir)
            
            self.logger.info(f"训练完成，模型已保存到: {output_dir}")
            
            # 评估模型
            eval_results = trainer.evaluate()
            self.logger.info(f"评估结果: {eval_results}")
            
            return output_dir
            
        except Exception as e:
            self.logger.error(f"训练失败: {e}")
            return None
    
    def test_model(self, model_path: str, test_texts: List[str]):
        """测试训练后的模型"""
        self.logger.info("正在测试微调后的模型...")
        
        try:
            # 加载微调后的模型
            model = AutoModel.from_pretrained(
                MODEL_CONFIG['llm_model'],
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            model = PeftModel.from_pretrained(model, model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            
            results = []
            for text in test_texts:
                prompt = f"请分析以下文本的政治风险等级：{text}"
                
                response, _ = model.chat(
                    tokenizer,
                    prompt,
                    history=[],
                    max_length=512,
                    temperature=0.1
                )
                
                results.append({
                    "input": text,
                    "output": response
                })
                
                self.logger.info(f"输入: {text}")
                self.logger.info(f"输出: {response}")
                self.logger.info("-" * 50)
            
            return results
            
        except Exception as e:
            self.logger.error(f"模型测试失败: {e}")
            return None

def main():
    """主函数"""
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("开始模型微调流程...")
    
    # 创建训练器
    trainer = PoliticalRiskModelTrainer()
    
    # 加载基础模型
    if not trainer.load_base_model():
        logger.error("基础模型加载失败，退出")
        return
    
    # 配置LoRA
    trainer.setup_lora_config()
    
    # 准备数据集
    train_dataset, val_dataset = trainer.prepare_dataset()
    if train_dataset is None or val_dataset is None:
        logger.error("数据集准备失败，退出")
        return
    
    # 训练模型
    model_path = trainer.train_model(train_dataset, val_dataset)
    if model_path is None:
        logger.error("模型训练失败，退出")
        return
    
    # 测试模型
    test_texts = [
        "今天我们学习数学公式的推导过程。",
        "不同政治制度各有优缺点，需要客观分析。",
        "当前制度存在严重问题，需要进行彻底改革。",
        "我们应该批判性地思考现有体制。"
    ]
    
    test_results = trainer.test_model(model_path, test_texts)
    if test_results:
        logger.info("模型测试完成")
    
    logger.info("微调流程结束")

if __name__ == "__main__":
    main()
