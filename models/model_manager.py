import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoProcessor
import logging
import os
import librosa
import numpy as np
from typing import Optional, List, Dict, Any
from config.settings import MODEL_CONFIG, DATA_PATHS

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

class ModelManager:
    """重构后的AI模型管理器，基于Qwen2.5-Omni-7B多模态模型，直接处理音频"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 使用Qwen2.5-Omni-7B多模态模型替代Whisper+文本模型
        self.multimodal_model: Optional[Any] = None
        self.processor: Optional[AutoProcessor] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.device = self._get_device()
        
        # 设置模型名称为Qwen2.5-Omni-7B
        self.model_name = "Qwen/Qwen2.5-Omni-7B"
    
    def _get_device(self) -> str:
        """获取计算设备"""
        if torch.cuda.is_available():
            self.logger.info(f"CUDA可用，GPU数量: {torch.cuda.device_count()}")
            return "cuda"
        else:
            self.logger.info("使用CPU运行")
            return "cpu"
    
    def load_multimodal_model(self) -> bool:
        """加载Qwen2.5-Omni-7B多模态模型"""
        try:
            self.logger.info(f"正在加载Qwen2.5-Omni-7B多模态模型: {self.model_name}")
            
            # 确保使用镜像源
            os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
            
            # 加载processor用于音频预处理
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                cache_dir=str(DATA_PATHS['models_dir'] / 'cache')
            )
            
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                cache_dir=str(DATA_PATHS['models_dir'] / 'cache')
            )
            
            # 添加pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 配置模型加载参数
            model_kwargs = {
                "trust_remote_code": True,
                "cache_dir": str(DATA_PATHS['models_dir'] / 'cache'),
                "torch_dtype": MODEL_CONFIG['default'].get('torch_dtype', torch.float16 if self.device == "cuda" else torch.float32),
                "low_cpu_mem_usage": MODEL_CONFIG['default'].get('low_cpu_mem_usage', True),
            }
            
            # 设备映射
            if self.device == "cuda":
                model_kwargs["device_map"] = MODEL_CONFIG['default'].get('device_map', 'auto')
            else:
                model_kwargs["torch_dtype"] = torch.float32
            
            # 加载Qwen2.5-Omni多模态模型
            self.multimodal_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            # 设置为评估模式
            self.multimodal_model.eval()
            
            self.logger.info(f"Qwen2.5-Omni-7B多模态模型加载成功")
            return True
            
        except Exception as e:
            self.logger.error(f"Qwen2.5-Omni-7B模型加载失败: {e}")
            return False
    
    def _load_audio_file(self, audio_path: str) -> np.ndarray:
        """加载音频文件并转换为模型所需格式"""
        try:
            # 使用librosa加载音频，采样率16kHz
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # 确保音频长度不超过30秒（480000采样点）
            max_length = 16000 * 30
            if len(audio) > max_length:
                audio = audio[:max_length]
                self.logger.warning(f"音频文件过长，已截取前30秒")
            
            return audio
            
        except Exception as e:
            self.logger.error(f"音频文件加载失败: {e}")
            raise
    
    def process_audio_directly(self, audio_path: str) -> Dict[str, Any]:
        """直接处理音频文件进行政治风险分析（替代transcribe_audio + analyze_text_risk）"""
        if not self.multimodal_model or not self.processor:
            raise RuntimeError("Qwen2.5-Omni多模态模型未加载")
        
        try:
            # 1. 加载音频文件
            audio_data = self._load_audio_file(audio_path)
            
            # 2. 构建多模态提示词
            prompt = self._build_audio_risk_analysis_prompt()
            
            # 3. 使用processor处理音频和文本
            inputs = self.processor(
                audio=audio_data,
                text=prompt,
                sampling_rate=16000,
                return_tensors="pt"
            )
            
            # 移动到设备
            if self.device == "cuda":
                inputs = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            # 4. 生成分析结果
            with torch.no_grad():
                generated_ids = self.multimodal_model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=MODEL_CONFIG['default'].get('temperature', 0.7),
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # 5. 解码结果
            # 移除输入部分，只保留生成的内容
            input_length = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
            generated_ids = generated_ids[:, input_length:]
            
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # 6. 解析风险分析结果
            risk_analysis = self._parse_risk_analysis(response)
            
            # 7. 提取音频特征用于补充信息
            audio_features = self._extract_basic_audio_features(audio_data)
            
            # 8. 构建综合结果
            result = {
                "text": risk_analysis.get("transcription", ""),  # 如果模型返回了转录文本
                "language": "zh",  # Qwen2.5-Omni主要支持中文
                "risk_analysis": risk_analysis,
                "audio_features": audio_features,
                "processing_method": "qwen2.5_omni_direct"
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"音频直接处理失败: {e}")
            raise
    
    def _build_audio_risk_analysis_prompt(self) -> str:
        """构建音频风险分析提示词"""
        prompt = """你是一个专业的音频内容政治风险分析专家。请直接分析这段音频内容，识别其中可能存在的政治风险或意识形态问题。

分析要求：
1. 理解音频中的语音内容
2. 识别可能的政治敏感内容
3. 评估意识形态倾向
4. 检测是否存在价值观冲突
5. 给出风险等级（低风险/中风险/高风险）
6. 提供具体的风险点说明

请以JSON格式返回分析结果，包含以下字段：
- transcription: 音频转录文本
- risk_level: 风险等级
- risk_score: 风险分数(0-100)
- key_issues: 关键问题列表
- suggestions: 改进建议
- detailed_analysis: 详细分析

注意：整个分析结果必须是一个有效的JSON对象，不要在JSON前后包含其他文本。"""
        
        return prompt
    
    def _extract_basic_audio_features(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """提取基本音频特征"""
        try:
            duration = len(audio_data) / 16000  # 采样率16kHz
            
            # 计算基本统计特征
            features = {
                "duration": duration,
                "sample_rate": 16000,
                "length": len(audio_data),
                "rms_energy": float(np.sqrt(np.mean(audio_data**2))),
                "max_amplitude": float(np.max(np.abs(audio_data))),
                "zero_crossing_rate": float(np.mean(librosa.feature.zero_crossing_rate(audio_data))),
            }
            
            # 如果音频不为空，计算更多特征
            if len(audio_data) > 0 and np.max(np.abs(audio_data)) > 0:
                try:
                    # 提取MFCC特征
                    mfccs = librosa.feature.mfcc(y=audio_data, sr=16000, n_mfcc=13)
                    features["mfcc_mean"] = float(np.mean(mfccs))
                    features["mfcc_std"] = float(np.std(mfccs))
                    
                    # 提取音调特征
                    pitches, magnitudes = librosa.piptrack(y=audio_data, sr=16000)
                    features["pitch_mean"] = float(np.mean(pitches[pitches > 0])) if np.any(pitches > 0) else 0.0
                    
                except Exception as e:
                    self.logger.warning(f"提取高级音频特征失败: {e}")
            
            return features
            
        except Exception as e:
            self.logger.warning(f"提取音频特征失败: {e}")
            return {
                "duration": len(audio_data) / 16000 if len(audio_data) > 0 else 0,
                "sample_rate": 16000,
                "length": len(audio_data),
                "error": str(e)
            }
    
    def _parse_risk_analysis(self, response: str) -> Dict[str, Any]:
        """解析风险分析结果"""
        try:
            self.logger.info(f"Raw multimodal model response:\n{response}")
            
            # 尝试解析JSON格式的响应
            import json
            import re
            
            # 提取JSON部分
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                parsed_result = json.loads(json_str)
                
                # 确保包含必要字段
                if "risk_level" not in parsed_result:
                    parsed_result["risk_level"] = "未知"
                if "risk_score" not in parsed_result:
                    parsed_result["risk_score"] = 0
                if "transcription" not in parsed_result:
                    parsed_result["transcription"] = ""
                
                return parsed_result
            else:
                # 如果无法解析JSON，尝试从文本中提取关键信息
                return self._extract_info_from_text(response)
                
        except Exception as e:
            self.logger.error(f"解析分析结果失败: {e}")
            return {
                "transcription": "",
                "risk_level": "解析错误",
                "risk_score": 0,
                "key_issues": ["结果解析失败"],
                "suggestions": ["请重新分析"],
                "detailed_analysis": response,
                "error": str(e)
            }
    
    def _extract_info_from_text(self, response: str) -> Dict[str, Any]:
        """从非JSON格式的响应中提取信息"""
        # 尝试从文本中提取转录内容和风险信息
        result = {
            "transcription": "",
            "risk_level": "未知",
            "risk_score": 0,
            "key_issues": [],
            "suggestions": [],
            "detailed_analysis": response
        }
        
        # 简单的文本解析逻辑
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if '转录' in line or '内容' in line:
                # 尝试提取转录内容
                if ':' in line:
                    result["transcription"] = line.split(':', 1)[1].strip()
            elif '风险' in line and ('高' in line or '中' in line or '低' in line):
                if '高风险' in line:
                    result["risk_level"] = "高风险"
                    result["risk_score"] = 80
                elif '中风险' in line:
                    result["risk_level"] = "中风险"
                    result["risk_score"] = 50
                elif '低风险' in line:
                    result["risk_level"] = "低风险"
                    result["risk_score"] = 20
        
        return result
    
    def initialize_models(self) -> bool:
        """初始化Qwen2.5-Omni多模态模型"""
        self.logger.info("开始初始化Qwen2.5-Omni-7B多模态模型...")
        
        success = self.load_multimodal_model()
        
        if success:
            self.logger.info("Qwen2.5-Omni-7B多模态模型初始化成功，系统支持直接音频处理")
            return True
        else:
            self.logger.error("Qwen2.5-Omni-7B多模态模型初始化失败")
            return False
    
    # 为了兼容性，保留原有方法名，但重定向到新的多模态处理
    def transcribe_audio(self, audio_path: str) -> Dict[str, Any]:
        """兼容性方法：语音转文本（现在通过多模态模型处理）"""
        result = self.process_audio_directly(audio_path)
        return {
            "text": result.get("text", ""),
            "segments": [],  # 多模态模型通常不返回segment信息
            "language": result.get("language", "zh")
        }
    
    def analyze_text_risk(self, text: str) -> Dict[str, Any]:
        """兼容性方法：分析文本风险（现在可以处理纯文本）"""
        if not self.multimodal_model or not self.tokenizer:
            raise RuntimeError("Qwen2.5-Omni多模态模型未加载")
        
        try:
            # 构建文本风险分析提示词
            prompt = self._build_text_risk_analysis_prompt(text)
            
            # 使用tokenizer处理纯文本
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            
            # 移动到设备
            if self.device == "cuda":
                inputs = inputs.to("cuda")
            
            # 生成分析结果
            with torch.no_grad():
                generated_ids = self.multimodal_model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=MODEL_CONFIG['default'].get('temperature', 0.7),
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # 解码结果
            input_length = inputs["input_ids"].shape[1]
            generated_ids = generated_ids[:, input_length:]
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # 解析结果
            return self._parse_risk_analysis(response)
            
        except Exception as e:
            self.logger.error(f"文本风险分析失败: {e}")
            raise
    
    def _build_text_risk_analysis_prompt(self, text: str) -> str:
        """构建文本风险分析提示词"""
        prompt = f"""你是一个专业的意识形态风险分析专家。请分析以下文本内容是否存在政治风险或意识形态问题。

分析要求：
1. 识别可能的政治敏感内容
2. 评估意识形态倾向
3. 检测是否存在价值观冲突
4. 给出风险等级（低风险/中风险/高风险）
5. 提供具体的风险点说明

需要分析的文本：
{text}

请以JSON格式返回分析结果，包含以下字段：
- risk_level: 风险等级
- risk_score: 风险分数(0-100)
- key_issues: 关键问题列表
- suggestions: 改进建议
- detailed_analysis: 详细分析

注意：整个分析结果必须是一个有效的JSON对象，不要在JSON前后包含其他文本。"""
        
        return prompt
