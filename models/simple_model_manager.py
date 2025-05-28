#!/usr/bin/env python3
"""
简化的快速模型管理器 - 恢复原始启动速度
使用 Whisper-large-v3 + Qwen2.5-7B-Instruct 组合
"""

import os
import sys
import torch
import time
import hashlib
import logging
import librosa
import numpy as np
import json
import re
from typing import Optional, List, Dict, Any
from pathlib import Path

# 设置环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

# 添加项目根目录到Python路径
_current_file_dir = os.path.dirname(__file__)
_project_root = os.path.abspath(os.path.join(_current_file_dir, '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    WhisperProcessor, 
    WhisperForConditionalGeneration,
    pipeline
)

from config.settings import MODEL_CONFIG, DATA_PATHS

class SimpleModelManager:
    """简化的快速模型管理器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Whisper模型用于语音转文字
        self.whisper_pipeline: Optional[Any] = None
        
        # Qwen2.5模型用于文本分析
        self.llm_model: Optional[Any] = None
        self.llm_tokenizer: Optional[AutoTokenizer] = None
        
        self.device = self._get_device()
        
        # 简单缓存
        self.inference_cache = {}
        
        # 模型配置
        self.whisper_model_name = "openai/whisper-large-v3"
        self.llm_model_name = "Qwen/Qwen2.5-7B-Instruct"  # 恢复使用7B模型
        
        self.logger.info(f"简化模型管理器初始化完成，设备: {self.device}")
    
    def _get_device(self) -> str:
        """获取计算设备"""
        if torch.cuda.is_available():
            device = f"cuda:{torch.cuda.current_device()}"
            self.logger.info(f"使用GPU: {device}")
            return device
        else:
            self.logger.info("使用CPU")
            return "cpu"
    
    def load_whisper_model(self) -> bool:
        """快速加载Whisper模型"""
        try:
            self.logger.info(f"正在加载Whisper模型: {self.whisper_model_name}")
            
            # 简单快速的加载方式
            self.whisper_pipeline = pipeline(
                "automatic-speech-recognition",
                model=self.whisper_model_name,
                device=self.device,
                torch_dtype=torch.float16 if self.device.startswith("cuda") else torch.float32,
                return_timestamps=True
            )
            
            self.logger.info("Whisper模型加载成功")
            return True
            
        except Exception as e:
            self.logger.error(f"Whisper模型加载失败: {e}")
            return False
    
    def load_llm_model(self) -> bool:
        """快速加载LLM模型"""
        try:
            self.logger.info(f"正在加载LLM模型: {self.llm_model_name}")
            
            # 加载tokenizer
            self.llm_tokenizer = AutoTokenizer.from_pretrained(
                self.llm_model_name,
                trust_remote_code=True,
                cache_dir=str(DATA_PATHS['models_dir'] / 'cache')
            )
            
            # 加载模型
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if self.device.startswith("cuda") else torch.float32,
                "low_cpu_mem_usage": True,
                "cache_dir": str(DATA_PATHS['models_dir'] / 'cache')
            }
            
            # 设备映射
            if self.device.startswith("cuda"):
                model_kwargs["device_map"] = "auto"
            
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                self.llm_model_name,
                **model_kwargs
            )
            
            self.llm_model.eval()
            
            self.logger.info("LLM模型加载成功")
            return True
            
        except Exception as e:
            self.logger.error(f"LLM模型加载失败: {e}")
            return False
    
    def initialize_models(self) -> bool:
        """快速初始化所有模型"""
        self.logger.info("开始初始化模型...")
        
        # 加载Whisper
        if not self.load_whisper_model():
            return False
        
        # 加载LLM
        if not self.load_llm_model():
            return False
        
        self.logger.info("模型初始化成功")
        return True
    
    def transcribe_audio(self, audio_path: str) -> Dict[str, Any]:
        """语音转文字"""
        if not self.whisper_pipeline:
            raise RuntimeError("Whisper模型未加载")
        
        try:
            start_time = time.time()
            
            # 使用pipeline处理音频
            result = self.whisper_pipeline(audio_path)
            
            transcribe_time = time.time() - start_time
            
            # 提取文本
            transcription = result.get("text", "").strip()
            
            self.logger.info(f"语音转文字完成，耗时: {transcribe_time:.2f}秒")
            
            return {
                "text": transcription,
                "language": "zh",
                "processing_time": transcribe_time
            }
            
        except Exception as e:
            self.logger.error(f"语音转文字失败: {e}")
            raise
    
    def analyze_text_risk(self, text: str) -> Dict[str, Any]:
        """文本风险分析"""
        if not self.llm_model or not self.llm_tokenizer:
            raise RuntimeError("LLM模型未加载")
        
        try:
            start_time = time.time()
            
            # 检查简单缓存
            cache_key = hashlib.md5(text.encode()).hexdigest()
            if cache_key in self.inference_cache:
                return self.inference_cache[cache_key]
            
            # 构建提示词
            prompt = self._build_risk_analysis_prompt(text)
            
            # 使用chat模板格式
            messages = [
                {"role": "system", "content": "你是一个专业的内容风险分析专家。"},
                {"role": "user", "content": prompt}
            ]
            
            # 应用chat模板
            formatted_prompt = self.llm_tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # 编码
            inputs = self.llm_tokenizer(
                formatted_prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=2048
            )
            
            # 移动到设备
            if self.device.startswith("cuda"):
                inputs = inputs.to(self.device)
            
            # 生成
            with torch.no_grad():
                outputs = self.llm_model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self.llm_tokenizer.eos_token_id
                )
            
            # 解码
            input_length = inputs["input_ids"].shape[1]
            generated_ids = outputs[:, input_length:]
            response = self.llm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            analysis_time = time.time() - start_time
            
            # 解析结果
            risk_analysis = self._parse_risk_analysis(response)
            risk_analysis["processing_time"] = analysis_time
            
            # 简单缓存（限制大小）
            if len(self.inference_cache) < 100:
                self.inference_cache[cache_key] = risk_analysis
            
            self.logger.info(f"文本分析完成，耗时: {analysis_time:.2f}秒")
            
            return risk_analysis
            
        except Exception as e:
            self.logger.error(f"文本分析失败: {e}")
            raise
    
    def process_audio_complete(self, audio_path: str) -> Dict[str, Any]:
        """完整的音频处理流程"""
        try:
            total_start_time = time.time()
            
            # 1. 语音转文字
            transcription_result = self.transcribe_audio(audio_path)
            text = transcription_result["text"]
            
            if not text.strip():
                return {
                    "text": "",
                    "language": "zh",
                    "risk_analysis": {
                        "risk_level": "无内容",
                        "risk_score": 0,
                        "key_issues": ["音频无有效语音内容"],
                        "suggestions": ["请检查音频文件"],
                        "detailed_analysis": "音频文件中未检测到有效的语音内容"
                    },
                    "processing_method": "simple_fast",
                    "total_processing_time": time.time() - total_start_time
                }
            
            # 2. 文本风险分析
            risk_analysis = self.analyze_text_risk(text)
            
            total_time = time.time() - total_start_time
            
            return {
                "text": text,
                "language": transcription_result.get("language", "zh"),
                "risk_analysis": risk_analysis,
                "processing_method": "simple_fast",
                "transcription_time": transcription_result.get("processing_time", 0),
                "analysis_time": risk_analysis.get("processing_time", 0),
                "total_processing_time": total_time
            }
            
        except Exception as e:
            self.logger.error(f"完整音频处理失败: {e}")
            raise
    
    def _build_risk_analysis_prompt(self, text: str) -> str:
        """构建风险分析提示词"""
        prompt = f"""请分析以下文本内容是否存在政治风险或意识形态问题：

文本内容：
{text}

分析要求：
1. 识别政治敏感内容
2. 评估意识形态倾向  
3. 检测价值观冲突
4. 给出风险等级（低风险/中风险/高风险）

请以JSON格式返回分析结果：
{{
    "risk_level": "风险等级",
    "risk_score": 风险分数(0-100),
    "key_issues": ["关键问题列表"],
    "suggestions": ["改进建议"],
    "detailed_analysis": "详细分析"
}}

注意：只返回JSON格式的结果，不要包含其他文本。"""
        
        return prompt
    
    def _parse_risk_analysis(self, response: str) -> Dict[str, Any]:
        """解析风险分析结果"""
        try:
            # 提取JSON部分
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                parsed_result = json.loads(json_str)
                
                # 确保包含必要字段
                required_fields = ["risk_level", "risk_score", "key_issues", "suggestions", "detailed_analysis"]
                for field in required_fields:
                    if field not in parsed_result:
                        parsed_result[field] = "未知" if field == "risk_level" else (0 if field == "risk_score" else [])
                
                return parsed_result
            else:
                # 如果没有找到JSON，尝试从文本中提取信息
                return self._extract_info_from_text(response)
                
        except Exception as e:
            self.logger.error(f"解析分析结果失败: {e}")
            return {
                "risk_level": "解析错误",
                "risk_score": 0,
                "key_issues": ["结果解析失败"],
                "suggestions": ["请重新分析"],
                "detailed_analysis": response,
                "error": str(e)
            }
    
    def _extract_info_from_text(self, response: str) -> Dict[str, Any]:
        """从文本中提取风险信息"""
        result = {
            "risk_level": "未知",
            "risk_score": 0,
            "key_issues": [],
            "suggestions": [],
            "detailed_analysis": response
        }
        
        response_lower = response.lower()
        
        # 提取风险等级
        if '高风险' in response or 'high risk' in response_lower:
            result["risk_level"] = "高风险"
            result["risk_score"] = 80
        elif '中风险' in response or 'medium risk' in response_lower:
            result["risk_level"] = "中风险"
            result["risk_score"] = 50
        elif '低风险' in response or 'low risk' in response_lower:
            result["risk_level"] = "低风险"
            result["risk_score"] = 20
        
        return result
