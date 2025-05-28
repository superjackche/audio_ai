#!/usr/bin/env python3
"""
快速高效的语音转文字+LLM分析管理器
使用 Whisper-large-v3 + Qwen2.5-7B-Instruct 组合
"""

import sys
import os

# 优先设置HuggingFace镜像环境变量（必须在导入transformers之前）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

# Add project root to sys.path to allow direct execution
# Determine the project root directory, which is one level up from the 'models' directory
_current_file_dir = os.path.dirname(__file__)
_project_root = os.path.abspath(os.path.join(_current_file_dir, '..'))
# Add the project root to sys.path if it's not already there
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    WhisperProcessor, 
    WhisperForConditionalGeneration,
    pipeline
)

# 特殊处理Qwen2.5-Omni模型
try:
    from transformers import Qwen2_5OmniForConditionalGeneration
except ImportError:
    Qwen2_5OmniForConditionalGeneration = None
import logging
import os
import librosa
import numpy as np
from typing import Optional, List, Dict, Any
import time

from config.settings import MODEL_CONFIG, DATA_PATHS

# 导入性能优化配置
try:
    from config.performance_optimization import (
        UltraPerformanceOptimizer, 
        get_production_config,
        RealTimeMonitor,
        SmartPreloader
    )
    PERFORMANCE_OPTIMIZATION_AVAILABLE = True
except ImportError:
    PERFORMANCE_OPTIMIZATION_AVAILABLE = False
    UltraPerformanceOptimizer = None
    RealTimeMonitor = None
    SmartPreloader = None

# 设置HuggingFace镜像源环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HUGGINGFACE_HUB_CACHE'] = str(DATA_PATHS['models_dir'] / 'cache')
os.environ['HF_HOME'] = str(DATA_PATHS['models_dir'] / 'cache')

# 强制使用镜像源
try:
    from huggingface_hub import hf_hub_download
    import huggingface_hub
    huggingface_hub.constants.HUGGINGFACE_HUB_CACHE = str(DATA_PATHS['models_dir'] / 'cache')
    huggingface_hub.constants.HF_HUB_CACHE = str(DATA_PATHS['models_dir'] / 'cache')
except ImportError:
    pass

class FastModelManager:
    """快速高效的语音转文字+LLM分析管理器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Whisper模型用于语音转文字
        self.whisper_model: Optional[Any] = None
        self.whisper_processor: Optional[WhisperProcessor] = None
        
        # Qwen2.5-7B模型用于文本分析
        self.llm_model: Optional[Any] = None
        self.llm_tokenizer: Optional[AutoTokenizer] = None
        
        self.device = self._get_device()
        
        # 性能优化组件
        self.performance_optimizer = None
        self.monitor = None
        self.preloader = None
        self.inference_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # 初始化性能优化
        if PERFORMANCE_OPTIMIZATION_AVAILABLE:
            self._initialize_performance_optimization()
        
        # 模型配置 - 使用优化后的模型设置
        self.whisper_model_name = "openai/whisper-large-v3"
        
        # 根据性能优化结果选择模型
        if self.performance_optimizer:
            config = self.performance_optimizer.get_optimized_config()
            self.llm_model_name = config.get("model_name", "Qwen/Qwen2.5-3B-Instruct")
            self.batch_size = config.get("batch_size", 8)
            self.max_length = config.get("max_length", 2048)
        else:
            self.llm_model_name = "Qwen/Qwen2.5-7B-Instruct"
            self.batch_size = 8
            self.max_length = 2048
        
    def _initialize_performance_optimization(self):
        """初始化性能优化组件"""
        try:
            self.logger.info("初始化性能优化组件...")
            
            # 创建性能优化器
            self.performance_optimizer = UltraPerformanceOptimizer()
            config = self.performance_optimizer.optimize_system()
            
            # 创建实时监控器
            self.monitor = RealTimeMonitor()
            self.monitor.start_monitoring()
            
            # 创建智能预加载器
            self.preloader = SmartPreloader()
            
            self.logger.info("性能优化组件初始化完成")
            self.logger.info(f"优化配置: {config}")
            
        except Exception as e:
            self.logger.error(f"性能优化初始化失败: {e}")
            self.performance_optimizer = None
            self.monitor = None
            self.preloader = None
    
    def _get_cache_key(self, text: str) -> str:
        """生成缓存键"""
        import hashlib
        return hashlib.md5(text.encode()).hexdigest()
    
    def _check_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """检查缓存"""
        if cache_key in self.inference_cache:
            self.cache_hits += 1
            self.logger.debug(f"缓存命中: {cache_key}")
            return self.inference_cache[cache_key]
        else:
            self.cache_misses += 1
            return None
    
    def _update_cache(self, cache_key: str, result: Dict[str, Any]):
        """更新缓存"""
        # 限制缓存大小
        if len(self.inference_cache) > 1000:
            # 删除最旧的条目
            oldest_key = next(iter(self.inference_cache))
            del self.inference_cache[oldest_key]
        
        self.inference_cache[cache_key] = result
        self.logger.debug(f"缓存更新: {cache_key}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": f"{hit_rate:.2f}%",
            "cache_size": len(self.inference_cache)
        }
        
    def _get_device(self) -> str:
        """获取计算设备，优先使用CUDA，并根据空闲显存选择GPU"""
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            self.logger.info(f"CUDA可用，检测到GPU数量: {gpu_count}")

            if gpu_count > 0:
                # 记录每个GPU的详细信息
                for i in range(gpu_count):
                    try:
                        props = torch.cuda.get_device_properties(i)
                        total_memory_gb = props.total_memory / (1024**3)
                        torch.cuda.set_device(i) 
                        free_memory_bytes, total_memory_bytes = torch.cuda.mem_get_info(i)
                        used_memory_bytes = total_memory_bytes - free_memory_bytes
                        
                        self.logger.info(
                            f"GPU {i} ({props.name}): "
                            f"总显存: {total_memory_gb:.2f}GB, "
                            f"已用显存: {used_memory_bytes / (1024**3):.2f}GB, "
                            f"空闲显存: {free_memory_bytes / (1024**3):.2f}GB"
                        )
                    except Exception as e:
                        self.logger.warning(f"获取GPU {i}详细信息失败: {e}")
                
                # 根据空闲显存选择GPU
                best_gpu_idx = 0
                max_free_memory = -1 
                
                if gpu_count > 0: # This check is a bit redundant given the outer if, but good for clarity
                    for i in range(gpu_count):
                        try:
                            torch.cuda.set_device(i) 
                            free_memory_bytes, _ = torch.cuda.mem_get_info(i)
                            if free_memory_bytes > max_free_memory:
                                max_free_memory = free_memory_bytes
                                best_gpu_idx = i
                        except Exception as e:
                            self.logger.warning(f"检查GPU {i}的空闲内存时出错: {e}")
                
                selected_device_id = best_gpu_idx
                selected_device_str = f"cuda:{selected_device_id}"
                
                chosen_gpu_name = "Unknown GPU"
                try:
                    chosen_gpu_name = torch.cuda.get_device_name(selected_device_id)
                except Exception as e:
                    self.logger.warning(f"获取选定GPU {selected_device_id} 名称失败: {e}")

                self.logger.info(f"将使用具有最大空闲显存的GPU {selected_device_id} ({chosen_gpu_name}) 进行计算。空闲显存: {max_free_memory / (1024**3):.2f}GB")
                
                # Set the default device for PyTorch to the selected GPU for this manager instance
                # This might affect other parts of the application if not handled carefully.
                # However, subsequent model loading explicitly uses self.device or device_map.
                # torch.cuda.set_device(selected_device_id) # Optional: set default device for torch ops

                return selected_device_str
            else: 
                self.logger.warning("CUDA可用但未检测到GPU设备，将使用CPU。")
                return "cpu"
        else:
            self.logger.info("CUDA不可用，将使用CPU运行")
            return "cpu"
    
    def load_whisper_model(self) -> bool:
        """加载Whisper模型"""
        try:
            self.logger.info(f"正在加载Whisper模型: {self.whisper_model_name}")
            
            # 检查本地缓存是否存在
            cache_dir = DATA_PATHS['models_dir'] / 'cache'
            model_cache_path = cache_dir / f"models--{self.whisper_model_name.replace('/', '--')}"
            
            if model_cache_path.exists():
                self.logger.info(f"发现本地缓存: {model_cache_path}")
                local_files_only = True
            else:
                self.logger.info("未发现本地缓存，将从镜像源下载")
                local_files_only = False
            
            # 性能优化配置
            model_kwargs = {
                "use_safetensors": True,
                "cache_dir": str(cache_dir),
                "local_files_only": local_files_only,
            }
            
            # 设置数据类型
            torch_dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
            
            # 应用性能优化
            if self.performance_optimizer:
                config = self.performance_optimizer.get_optimized_config()
                whisper_config = config.get("whisper_kwargs", {})
                # 避免重复的参数，排除会与pipeline参数冲突的设置
                excluded_keys = {"torch_dtype", "device", "device_map"}
                for key, value in whisper_config.items():
                    if key not in excluded_keys and key not in model_kwargs:
                        model_kwargs[key] = value
            
            # 使用pipeline方式加载
            self.whisper_pipeline = pipeline(
                "automatic-speech-recognition",
                model=self.whisper_model_name,
                device=self.device,
                torch_dtype=torch_dtype,
                token=False,  # 直接在pipeline中设置token参数
                model_kwargs=model_kwargs
            )
            
            # 设置生成参数以提高速度
            self.whisper_pipeline.model.config.forced_decoder_ids = None
            self.whisper_pipeline.model.config.suppress_tokens = []
            
            # 预加载模型（如果启用）
            if self.preloader:
                self.preloader.preload_model(self.whisper_pipeline, "whisper")
            
            self.logger.info("Whisper模型加载成功")
            return True
            
        except Exception as e:
            self.logger.error(f"Whisper模型加载失败: {e}")
            return False
    
    def load_llm_model(self) -> bool:
        """加载LLM模型"""
        try:
            self.logger.info(f"正在加载LLM模型: {self.llm_model_name} 到设备: {self.device}")
            
            # 设置HuggingFace镜像
            import os
            os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
            
            # 判断是否为本地路径
            is_local_path = os.path.exists(self.llm_model_name)
            
            # 加载tokenizer
            tokenizer_kwargs = {
                "trust_remote_code": True,
            }
            
            if is_local_path:
                tokenizer_kwargs["local_files_only"] = True
            else:
                tokenizer_kwargs["cache_dir"] = str(DATA_PATHS['models_dir'] / 'cache')
                tokenizer_kwargs["local_files_only"] = False
            
            self.llm_tokenizer = AutoTokenizer.from_pretrained(
                self.llm_model_name,
                **tokenizer_kwargs
            )
            
            # 加载模型
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if self.device.startswith("cuda") else torch.float32,
                "low_cpu_mem_usage": True,
            }
            
            # 应用性能优化配置
            if self.performance_optimizer:
                config = self.performance_optimizer.get_optimized_config()
                llm_specific_kwargs = config.get("model_kwargs", {})
                # Remove unexpected arguments that are not supported by AutoModelForCausalLM
                unexpected_args = ['primary_model', 'fallback_model', 'quantization', 'use_flash_attention_2', 'attn_implementation']
                for arg in unexpected_args:
                    if arg in llm_specific_kwargs:
                        self.logger.warning(f"Removing unexpected '{arg}' argument from LLM kwargs.")
                        del llm_specific_kwargs[arg]
                model_kwargs.update(llm_specific_kwargs)
                
                # 检查量化配置 - 暂时禁用以避免兼容性问题
                try:
                    model_optimization = config.get("model_optimization", {})
                    quantization_config = model_optimization.get("quantization", {})
                    if (quantization_config.get("enabled", False) and 
                        self.device.startswith("cuda") and 
                        quantization_config.get("load_in_4bit", False)):
                        from transformers import BitsAndBytesConfig
                        model_kwargs["quantization_config"] = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.float16,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type="nf4"
                        )
                        self.logger.info("启用4-bit量化优化")
                    else:
                        self.logger.info("量化已禁用，使用标准精度")
                except Exception as e:
                    self.logger.warning(f"量化配置失败，使用标准精度: {e}")
            else:
                self.logger.info("未检测到性能优化器，使用默认配置")
            
            if is_local_path:
                model_kwargs["local_files_only"] = True
            else:
                model_kwargs["cache_dir"] = str(DATA_PATHS['models_dir'] / 'cache')
                model_kwargs["local_files_only"] = False
            
            if self.device.startswith("cuda"):
                if torch.cuda.device_count() == 1 or ":" in self.device:
                     gpu_id_to_use = int(self.device.split(':')[1]) if ':' in self.device else 0
                     model_kwargs["device_map"] = {"": gpu_id_to_use}
                     self.logger.info(f"LLM模型将加载到单个GPU: cuda:{gpu_id_to_use}")
                else:
                    model_kwargs["device_map"] = "auto"
                    self.logger.info("LLM模型将使用 device_map='auto' 自动分配到可用GPU。")
            else:
                model_kwargs["device_map"] = {"": "cpu"}
                self.logger.info("LLM模型将加载到CPU。")

            self.llm_model = AutoModelForCausalLM.from_pretrained(
                self.llm_model_name,
                **model_kwargs
            )
            
            self.llm_model.eval()
            
            # 预加载模型（如果启用）
            if self.preloader:
                self.preloader.preload_model(self.llm_model, "llm")
            
            self.logger.info("LLM模型加载成功")
            return True
            
        except Exception as e:
            self.logger.error(f"LLM模型加载失败: {e}")
            return False
    
    def initialize_models(self) -> bool:
        """初始化所有模型"""
        self.logger.info("开始初始化快速模型组合...")
        
        # 先加载Whisper
        if not self.load_whisper_model():
            return False
        
        # 再加载LLM
        if not self.load_llm_model():
            return False
        
        self.logger.info("快速模型组合初始化成功")
        return True
    
    def transcribe_audio(self, audio_path: str) -> Dict[str, Any]:
        """语音转文字"""
        if not self.whisper_pipeline:
            raise RuntimeError("Whisper模型未加载")
        
        try:
            start_time = time.time()
            
            # 记录性能指标
            if self.monitor:
                self.monitor.log_operation_start("transcribe_audio")
            
            # 预处理音频：限制长度和采样率
            audio_data = self._preprocess_audio(audio_path)
            
            # 动态调整处理参数
            chunk_length = 30
            batch_size = self.batch_size
            
            if self.performance_optimizer:
                config = self.performance_optimizer.get_optimized_config()
                chunk_length = config.get("whisper_chunk_length", 30)
                batch_size = config.get("whisper_batch_size", self.batch_size)
            
            # 使用pipeline处理音频，简化参数避免冲突
            result = self.whisper_pipeline(
                audio_data,
                return_timestamps=True,
                chunk_length_s=chunk_length,
                batch_size=batch_size
            )
            
            transcribe_time = time.time() - start_time
            
            # 记录性能指标
            if self.monitor:
                self.monitor.log_operation_end("transcribe_audio", transcribe_time)
            
            # 提取文本
            if isinstance(result, dict) and "text" in result:
                transcription = result["text"].strip()
            elif isinstance(result, list) and len(result) > 0:
                # 如果返回的是chunks列表，合并所有文本
                transcription = " ".join([chunk.get("text", "") for chunk in result]).strip()
            else:
                transcription = str(result).strip()
            
            self.logger.info(f"语音转文字完成，耗时: {transcribe_time:.2f}秒")
            self.logger.info(f"转录结果: {transcription[:100]}...")
            
            return {
                "text": transcription,
                "language": "zh",
                "processing_time": transcribe_time,
                "segments": result.get("chunks", []) if isinstance(result, dict) else []
            }
            
        except Exception as e:
            if self.monitor:
                self.monitor.log_error("transcribe_audio", str(e))
            self.logger.error(f"语音转文字失败: {e}")
            raise
    
    def _preprocess_audio(self, audio_path: str) -> np.ndarray:
        """预处理音频文件"""
        try:
            # 加载音频文件
            audio, sr = librosa.load(audio_path, sr=16000)  # Whisper期望16kHz采样率
            
            # 限制音频长度（最多5分钟，以避免处理时间过长）
            # max_duration = 300  # 5分钟
            # max_samples = max_duration * sr
            
            # if len(audio) > max_samples:
            #     self.logger.warning(f"音频过长，截取前{max_duration}秒")
            #     audio = audio[:max_samples]
            
            return audio
            
        except Exception as e:
            self.logger.error(f"音频预处理失败: {e}")
            raise
    
    def analyze_text_risk(self, text: str) -> Dict[str, Any]:
        """文本风险分析"""
        if not self.llm_model or not self.llm_tokenizer:
            raise RuntimeError("LLM模型未加载")
        
        try:
            start_time = time.time()
            
            # 检查缓存
            cache_key = self._get_cache_key(text)
            cached_result = self._check_cache(cache_key)
            if cached_result:
                self.logger.info(f"使用缓存结果，耗时: 0.01秒")
                return cached_result
            
            # 记录性能指标
            if self.monitor:
                self.monitor.log_operation_start("analyze_text_risk")
            
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
                max_length=self.max_length
            )
            
            # 移动到设备
            if self.device.startswith("cuda"):
                inputs = inputs.to(self.device)
            
            # 获取优化的生成参数
            generation_config = {
                "max_new_tokens": 512,
                "temperature": 0.1,
                "do_sample": False,
                "pad_token_id": self.llm_tokenizer.eos_token_id,
                "eos_token_id": self.llm_tokenizer.eos_token_id,
            }
            
            if self.performance_optimizer:
                config = self.performance_optimizer.get_optimized_config()
                generation_config.update(config.get("generation_kwargs", {}))
            
            # 生成（使用优化的参数）
            with torch.no_grad():
                outputs = self.llm_model.generate(
                    **inputs,
                    **generation_config
                )
            
            # 解码
            input_length = inputs["input_ids"].shape[1]
            generated_ids = outputs[:, input_length:]
            response = self.llm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            analysis_time = time.time() - start_time
            
            # 记录性能指标
            if self.monitor:
                self.monitor.log_operation_end("analyze_text_risk", analysis_time)
            
            self.logger.info(f"文本分析完成，耗时: {analysis_time:.2f}秒")
            
            # 解析结果
            risk_analysis = self._parse_risk_analysis(response)
            risk_analysis["processing_time"] = analysis_time
            
            # 更新缓存
            self._update_cache(cache_key, risk_analysis)
            
            return risk_analysis
            
        except Exception as e:
            if self.monitor:
                self.monitor.log_error("analyze_text_risk", str(e))
            self.logger.error(f"文本分析失败: {e}")
            raise
    
    def process_audio_complete(self, audio_path: str) -> Dict[str, Any]:
        """完整的音频处理流程：转录 + 分析"""
        try:
            total_start_time = time.time()
            
            # 记录整体性能指标
            if self.monitor:
                self.monitor.log_operation_start("process_audio_complete")
            
            # 1. 语音转文字
            transcription_result = self.transcribe_audio(audio_path)
            text = transcription_result["text"]
            
            if not text.strip():
                result = {
                    "text": "",
                    "language": "zh",
                    "risk_analysis": {
                        "risk_level": "无内容",
                        "risk_score": 0,
                        "key_issues": ["音频无有效语音内容"],
                        "suggestions": ["请检查音频文件"],
                        "detailed_analysis": "音频文件中未检测到有效的语音内容"
                    },
                    "processing_method": "whisper_llm_fast_optimized",
                    "total_processing_time": time.time() - total_start_time,
                    "cache_stats": self.get_cache_stats()
                }
                
                if self.monitor:
                    self.monitor.log_operation_end("process_audio_complete", result["total_processing_time"])
                
                return result
            
            # 2. 文本风险分析
            risk_analysis = self.analyze_text_risk(text)
            
            total_time = time.time() - total_start_time
            
            # 记录整体性能指标
            if self.monitor:
                self.monitor.log_operation_end("process_audio_complete", total_time)
            
            # 3. 构建完整结果
            result = {
                "text": text,
                "language": transcription_result.get("language", "zh"),
                "risk_analysis": risk_analysis,
                "processing_method": "whisper_llm_fast_optimized",
                "transcription_time": transcription_result.get("processing_time", 0),
                "analysis_time": risk_analysis.get("processing_time", 0),
                "total_processing_time": total_time,
                "cache_stats": self.get_cache_stats(),
                "performance_metrics": self.get_performance_metrics()
            }
            
            self.logger.info(f"完整处理完成，总耗时: {total_time:.2f}秒")
            self.logger.info(f"缓存统计: {result['cache_stats']}")
            
            return result
            
        except Exception as e:
            if self.monitor:
                self.monitor.log_error("process_audio_complete", str(e))
            self.logger.error(f"完整音频处理失败: {e}")
            raise
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        metrics = {}
        
        if self.monitor:
            metrics.update(self.monitor.get_current_metrics())
        
        if self.performance_optimizer:
            metrics.update(self.performance_optimizer.get_system_info())
        
        return metrics
    
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
            import json
            import re
            
            self.logger.info(f"LLM响应: {response[:200]}...")
            
            # 提取JSON部分
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                parsed_result = json.loads(json_str)
                
                # 确保包含必要字段
                required_fields = {
                    "risk_level": "未知",
                    "risk_score": 0,
                    "key_issues": [],
                    "suggestions": [],
                    "detailed_analysis": ""
                }
                
                for field, default_value in required_fields.items():
                    if field not in parsed_result:
                        parsed_result[field] = default_value
                
                return parsed_result
            else:
                # 简单文本解析
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
    
    def cleanup(self):
        """清理资源"""
        try:
            self.logger.info("正在清理资源...")
            
            # 停止监控
            if hasattr(self, 'monitor') and self.monitor:
                self.monitor.stop_monitoring()
            
            # 清理缓存
            if hasattr(self, 'inference_cache'):
                self.inference_cache.clear()
            
            # 清理GPU缓存
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    self.logger.info("GPU缓存已清理")
            except (ImportError, AttributeError):
                # PyTorch可能已经被清理或不可用
                pass
            
            self.logger.info("资源清理完成")
            
        except Exception as e:
            self.logger.error(f"资源清理失败: {e}")
    
    def __del__(self):
        """析构函数"""
        try:
            # 安全的清理，避免在对象销毁时出错
            if hasattr(self, 'monitor') and self.monitor:
                try:
                    self.monitor.stop_monitoring()
                except:
                    pass
            
            if hasattr(self, 'inference_cache'):
                try:
                    self.inference_cache.clear()
                except:
                    pass
        except:
            pass
