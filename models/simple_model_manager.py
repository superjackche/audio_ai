#!/usr/bin/env python3
"""
简化的快速模型管理器 - 恢复原始启动速度
使用 Whisper-large-v3 + Qwen2.5-7B-Instruct 组合

Copyright (c) 2025 superjackche
GitHub: https://github.com/superjackche/audio_ai
Licensed under the MIT License. See LICENSE file for details.
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
import glob
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

# FunASR 相关导入
try:
    from funasr import AutoModel
    FUNASR_AVAILABLE = True
    print(f"✅ FunASR库导入成功，AutoModel可用")
except ImportError as e:
    FUNASR_AVAILABLE = False
    print(f"❌ FunASR库导入失败: {e}")
    print("   💡 请确认FunASR已正确安装: pip install funasr")

from config.settings import MODEL_CONFIG, DATA_PATHS

class SimpleModelManager:
    """简化的快速模型管理器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"DATA_PATHS['models_dir'] is configured as: {DATA_PATHS['models_dir']}")
        
        # 显示FunASR状态
        print(f"🔧 FunASR状态检查: {'✅ 可用' if FUNASR_AVAILABLE else '❌ 不可用'}")
        
        # Whisper模型用于语音转文字
        self.whisper_pipeline: Optional[Any] = None
        
        # Qwen2.5模型用于文本分析
        self.llm_model: Optional[Any] = None
        self.llm_tokenizer: Optional[AutoTokenizer] = None
        
        self.device = self._get_device()
        
        # 简单缓存
        self.inference_cache = {}
        
        # 模型配置 - 只使用SenseVoice和LLM
        # 使用SenseVoice - 专为中文优化的轻量级ASR模型（仅50MB）
        self.whisper_model_name = "iic/SenseVoiceSmall"
        self.use_sensevoice = True  # 只使用SenseVoice，不使用Whisper
        self.skip_whisper = True    # 跳过Whisper下载
        
        # 检查多个可能的本地LLM模型路径，包括HuggingFace缓存
        cache_dir = DATA_PATHS['models_dir'] / 'cache'
        possible_paths = [
            DATA_PATHS['models_dir'] / "Qwen2.5-7B-Instruct",
            DATA_PATHS['models_dir'] / "Qwen--Qwen2.5-7B-Instruct", 
            Path("/new_disk/cwh/models/Qwen2.5-7B-Instruct"),
            Path("/new_disk/cwh/models/Qwen--Qwen2.5-7B-Instruct"),
            # 检查HuggingFace缓存目录
            cache_dir / "models--Qwen--Qwen2.5-7B-Instruct" / "snapshots",
        ]
        
        self.llm_model_path = None
        self.llm_model_name = "Qwen/Qwen2.5-7B-Instruct"  # 用于在线下载的fallback
        
        # 寻找本地模型
        print("🔍 检查本地LLM模型...")
        for path in possible_paths:
            print(f"  检查路径: {path}")
            if path.exists() and path.is_dir():
                # 对于HuggingFace缓存目录，需要进入snapshots子目录
                if "snapshots" in str(path):
                    # 查找snapshots目录下的具体版本
                    snapshot_dirs = [d for d in path.iterdir() if d.is_dir()]
                    if snapshot_dirs:
                        # 使用最新的snapshot
                        latest_snapshot = max(snapshot_dirs, key=lambda x: x.stat().st_mtime)
                        path = latest_snapshot
                        print(f"    发现缓存快照: {latest_snapshot.name}")
                
                # 检查是否包含必要的模型文件
                required_files = ["config.json"]
                
                # 检查模型权重文件：支持多种格式
                has_single_model = any((path / f).exists() for f in ["model.safetensors", "pytorch_model.bin"])
                has_sharded_safetensors = any((path / f"model-{i:05d}-of-*.safetensors").exists() for i in range(1, 10))
                has_sharded_pytorch = any((path / f"pytorch_model-{i:05d}-of-*.bin").exists() for i in range(1, 10))
                
                # 新增：检查分片命名格式如 model-00001-of-00004.safetensors
                sharded_files = glob.glob(str(path / "model-*-of-*.safetensors"))
                has_new_sharded_format = len(sharded_files) > 0
                
                has_model_files = has_single_model or has_sharded_safetensors or has_sharded_pytorch or has_new_sharded_format
                has_tokenizer = any((path / f).exists() for f in ["tokenizer.json", "tokenizer_config.json"])
                
                # 调试信息
                print(f"    模型文件检查: 单文件={has_single_model}, 旧分片={has_sharded_safetensors or has_sharded_pytorch}, 新分片={has_new_sharded_format}")
                print(f"    tokenizer检查: {has_tokenizer}")
                if has_new_sharded_format:
                    print(f"    发现分片文件: {len(sharded_files)}个")
                
                if all((path / f).exists() for f in required_files) and has_model_files and has_tokenizer:
                    self.llm_model_path = path
                    print(f"  ✅ 找到完整的本地LLM模型: {path}")
                    break
                else:
                    print(f"  ⚠️  路径存在但模型文件不完整: {path}")
                    print(f"    缺少: config.json={not all((path / f).exists() for f in required_files)}, model_files={not has_model_files}, tokenizer={not has_tokenizer}")
                    # 调试：列出目录内容
                    try:
                        files = list(path.iterdir())[:10]  # 只显示前10个文件
                        print(f"    目录内容示例: {[f.name for f in files]}")
                    except:
                        pass
            else:
                print(f"  ❌ 路径不存在: {path}")
        
        if not self.llm_model_path:
            print(f"  📥 未找到本地LLM模型，将使用已缓存的模型: {self.llm_model_name}")
            print(f"  💡 缓存目录: {cache_dir}")
            # 即使没有找到完整路径，也可以使用模型名称，让transformers自动找缓存
        
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
        """快速加载SenseVoice语音识别模型 - 不使用Whisper"""
        try:
            # 强制检查FunASR状态
            print(f"🔍 检查FunASR状态: FUNASR_AVAILABLE = {FUNASR_AVAILABLE}")
            
            if not FUNASR_AVAILABLE:
                print("❌ FunASR库不可用，无法使用SenseVoice")
                print("   💡 请安装FunASR: pip install funasr")
                return False
            
            print(f"🎙️  加载SenseVoice中文语音识别模型: {self.whisper_model_name}")
            print("   ⚡ SenseVoice模型仅50MB，专为中文优化，加载超快！")
            
            self.logger.info(f"正在使用FunASR加载SenseVoice模型: {self.whisper_model_name}")
            
            # 使用FunASR的AutoModel加载SenseVoice
            # 根据文档，需要设置正确的参数
            model_kwargs = {
                "trust_remote_code": True,
                "device": self.device,
                "disable_update": True  # 禁用更新检查，加快启动
            }
            
            # 添加镜像源环境变量支持
            if 'HF_ENDPOINT' in os.environ:
                model_kwargs["hub_base_url"] = os.environ['HF_ENDPOINT']
            
            self.whisper_pipeline = AutoModel(
                model=self.whisper_model_name,
                **model_kwargs
            )
            
            print("✅ SenseVoice中文语音识别模型加载成功")
            self.logger.info("SenseVoice模型加载成功")
            return True
            
        except Exception as e:
            print(f"❌ SenseVoice模型加载失败: {e}")
            self.logger.error(f"SenseVoice模型加载失败: {e}")
            return False
    
    def load_llm_model(self) -> bool:
        """快速加载LLM模型"""
        try:
            if self.llm_model_path:
                # 使用本地模型
                print(f"📂 使用本地LLM模型: {self.llm_model_path}")
                self.logger.info(f"从本地路径加载LLM模型: {self.llm_model_path}")
                model_path_str = str(self.llm_model_path)
                use_local_only = True
            else:
                # 使用在线模型
                print(f"📥 需要下载LLM模型: {self.llm_model_name}")
                print("   📦 模型大小约13GB，这将需要较长时间...")
                print("   💡 建议：将本地7B模型放置在以下任一路径：")
                print("      - /new_disk/cwh/models/Qwen2.5-7B-Instruct/")
                print("      - /new_disk/cwh/audio_ai/models/Qwen2.5-7B-Instruct/")
                print("   ⏳ 开始下载模型文件...")
                self.logger.info(f"从网络下载LLM模型: {self.llm_model_name}")
                model_path_str = self.llm_model_name
                use_local_only = False

            # 加载tokenizer
            print("🔧 加载Tokenizer...")
            tokenizer_kwargs = {
                "trust_remote_code": True,
                "cache_dir": str(DATA_PATHS['models_dir'] / 'cache'),
            }
            if use_local_only:
                tokenizer_kwargs["local_files_only"] = True
            else:
                # 设置镜像源和下载配置
                tokenizer_kwargs["local_files_only"] = False
                tokenizer_kwargs["resume_download"] = True
                print("   🌐 使用hf-mirror镜像源下载...")
            
            try:
                self.llm_tokenizer = AutoTokenizer.from_pretrained(
                    model_path_str,
                    **tokenizer_kwargs
                )
                print("✅ Tokenizer加载成功")
            except Exception as e:
                if not use_local_only:
                    print(f"   ⚠️  Tokenizer下载可能较慢，请耐心等待...")
                    print(f"   🔄 重试中... (错误: {e})")
                    # 重试一次，增加超时时间
                    import time
                    time.sleep(2)
                    self.llm_tokenizer = AutoTokenizer.from_pretrained(
                        model_path_str,
                        **tokenizer_kwargs
                    )
                    print("✅ Tokenizer下载并加载成功")
                else:
                    raise e
            
            # 加载模型
            print("🚀 加载LLM模型权重...")
            if not use_local_only:
                print("   ⏳ 正在下载模型文件，请耐心等待...")
                print("   📊 下载进度将在命令行显示...")
            
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if self.device.startswith("cuda") else torch.float32,
                "low_cpu_mem_usage": True,
                "cache_dir": str(DATA_PATHS['models_dir'] / 'cache'),
            }
            
            if use_local_only:
                model_kwargs["local_files_only"] = True
            else:
                model_kwargs["local_files_only"] = False
                model_kwargs["resume_download"] = True
                # 强制使用镜像源
                model_kwargs["proxies"] = None
            
            if self.device.startswith("cuda"):
                model_kwargs["device_map"] = "auto"
            
            try:
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    model_path_str,
                    **model_kwargs
                )
            except Exception as e:
                if not use_local_only and "offline" not in str(e).lower():
                    print(f"   ⚠️  模型下载可能较慢，请耐心等待...")
                    print(f"   🔄 重试下载... (错误: {e})")
                    # 重试一次，可能是网络问题
                    import time
                    time.sleep(5)
                    self.llm_model = AutoModelForCausalLM.from_pretrained(
                        model_path_str,
                        **model_kwargs
                    )
                else:
                    raise e
            
            self.llm_model.eval()
            
            print("✅ LLM模型加载成功")
            if not use_local_only:
                print(f"   💾 模型已缓存到: {DATA_PATHS['models_dir']}/cache")
            self.logger.info(f"LLM模型加载成功: {model_path_str}")
            return True
            
        except Exception as e: 
            print(f"❌ LLM模型加载失败: {e}")
            print("   💡 可能的解决方案：")
            print("      1. 检查网络连接")
            print("      2. 检查磁盘空间 (需要约20GB)")
            print("      3. 使用本地模型文件")
            print("      4. 尝试使用VPN或更换网络")
            self.logger.error(f"LLM模型加载失败: {e}", exc_info=True)
            return False
    
    def initialize_models(self) -> bool:
        """快速初始化所有模型 - 只加载SenseVoice和LLM"""
        print("\n🤖 开始初始化AI模型...")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        self.logger.info("开始初始化模型...")
        
        # 加载SenseVoice
        print("🎙️  正在加载SenseVoice语音识别模型...")
        if not self.load_whisper_model():
            print("❌ SenseVoice模型初始化失败")
            return False
        
        # 加载LLM
        print("\n🧠 正在加载LLM文本分析模型...")
        if not self.load_llm_model():
            print("❌ LLM模型初始化失败")
            return False
        
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("🎉 所有AI模型初始化成功！")
        print("   ✅ SenseVoice中文语音识别模型已就绪")
        print("   ✅ Qwen2.5-7B文本分析模型已就绪")
        self.logger.info("模型初始化成功")
        return True
    
    def transcribe_audio(self, audio_path: str) -> Dict[str, Any]:
        """语音转文字，使用SenseVoice进行中英文混合识别"""
        # 懒加载检查
        if not self.whisper_pipeline:
            print("🎙️  首次使用，正在加载SenseVoice模型...")
            if not self.load_whisper_model():
                raise RuntimeError("语音识别模型加载失败")
        
        try:
            start_time = time.time()
            
            # 使用FunASR的SenseVoice模型进行识别
            self.logger.info("使用SenseVoice模型进行语音识别...")
            
            # SenseVoice使用FunASR接口
            result = self.whisper_pipeline.generate(
                input=audio_path,
                cache={},
                language="auto",  # 自动检测语言，支持中英混合
                use_itn=True,     # 使用逆文本标准化
                batch_size_s=60  # 批处理大小
            )
            
            # 处理FunASR的返回结果
            if isinstance(result, list) and len(result) > 0:
                transcription = result[0].get("text", "").strip()
            else:
                transcription = ""
                    
            transcribe_time = time.time() - start_time
            
            # 检测语言
            detected_language = self._detect_language(transcription)
            
            self.logger.info(f"语音转文字完成，耗时: {transcribe_time:.2f}秒")
            self.logger.info(f"检测语言: {detected_language}, 转录长度: {len(transcription)}字符")
            
            return {
                "text": transcription,
                "language": detected_language,
                "processing_time": transcribe_time,
                "method": "sensevoice"
            }
            
        except Exception as e:
            self.logger.error(f"语音转文字失败: {e}")
            raise
    
    def analyze_text_risk(self, text: str) -> Dict[str, Any]:
        """文本风险分析"""
        # 懒加载检查
        if not self.llm_model or not self.llm_tokenizer:
            print("🧠 首次使用，正在加载LLM模型...")
            if not self.load_llm_model():
                raise RuntimeError("LLM模型加载失败")
        
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
    
    def _detect_language(self, text: str) -> str:
        """检测文本语言"""
        try:
            if not text or not text.strip():
                return "unknown"
            
            # 简单的中英文检测
            chinese_chars = len([c for c in text if '\u4e00' <= c <= '\u9fff'])
            english_chars = len([c for c in text if c.isalpha() and c.isascii()])
            total_chars = len(text.strip())
            
            if total_chars == 0:
                return "unknown"
            
            chinese_ratio = chinese_chars / total_chars
            english_ratio = english_chars / total_chars
            
            # 判断主要语言
            if chinese_ratio > 0.3:
                if english_ratio > 0.2:
                    return "zh-en"  # 中英混合
                else:
                    return "zh"     # 主要是中文
            elif english_ratio > 0.5:
                return "en"         # 主要是英文
            else:
                return "mixed"      # 其他混合情况
                
        except Exception as e:
            self.logger.warning(f"语言检测失败: {e}")
            return "unknown"
