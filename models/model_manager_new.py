import torch
import whisper
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig
import logging
import os
import re
import json
from typing import Optional, List, Dict, Any
from config.settings import MODEL_CONFIG, DATA_PATHS

class ModelManager:
    """AI模型管理器，负责加载和管理语音识别和大语言模型"""
    
    def __init__(self):
        # 设置日志（需要在其他初始化之前）
        self.logger = logging.getLogger(__name__)
        
        self.whisper_model: Optional[whisper.Whisper] = None
        self.llm_model: Optional[Any] = None
        self.llm_tokenizer: Optional[Any] = None
        self.device = self._get_device()
        
        # 确保模型缓存目录存在
        cache_dir = DATA_PATHS['models_dir'] / 'cache'
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ['HUGGINGFACE_HUB_CACHE'] = str(cache_dir)
        os.environ['HF_HOME'] = str(cache_dir)
        
    def _get_device(self) -> str:
        """获取计算设备"""
        if torch.cuda.is_available():
            self.logger.info(f"CUDA可用，GPU数量: {torch.cuda.device_count()}")
            return "cuda"
        else:
            self.logger.info("使用CPU运行")
            return "cpu"
    
    def load_whisper_model(self) -> bool:
        """加载Whisper语音识别模型"""
        try:
            self.logger.info(f"正在加载Whisper模型: {MODEL_CONFIG['whisper_model']}")
            self.whisper_model = whisper.load_model(
                MODEL_CONFIG['whisper_model'],
                device=self.device
            )
            self.logger.info("Whisper模型加载成功")
            return True
        except Exception as e:
            self.logger.error(f"Whisper模型加载失败: {e}")
            return False
    
    def load_llm_model(self) -> bool:
        """加载大语言模型"""
        try:
            self.logger.info(f"正在加载大语言模型: {MODEL_CONFIG['llm_model']}")
            self.logger.info(f"使用镜像源: {os.environ.get('HF_ENDPOINT', 'default')}")
            
            # 加载tokenizer
            self.llm_tokenizer = AutoTokenizer.from_pretrained(
                MODEL_CONFIG['llm_model'],
                trust_remote_code=MODEL_CONFIG['trust_remote_code'],
                cache_dir=str(DATA_PATHS['models_dir'] / 'cache')
            )
            
            # 为某些模型添加pad_token
            if self.llm_tokenizer.pad_token is None:
                self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
            
            # 配置模型加载参数
            model_kwargs = {
                "trust_remote_code": MODEL_CONFIG['trust_remote_code'],
                "cache_dir": str(DATA_PATHS['models_dir'] / 'cache'),
                "torch_dtype": MODEL_CONFIG['torch_dtype'],
                "low_cpu_mem_usage": MODEL_CONFIG['low_cpu_mem_usage']
            }
            
            # 根据设备和配置设置量化和设备映射
            if self.device == "cuda" and MODEL_CONFIG.get("load_in_8bit", False):
                model_kwargs["load_in_8bit"] = True
                model_kwargs["device_map"] = MODEL_CONFIG['device_map']
            elif self.device == "cuda":
                model_kwargs["device_map"] = MODEL_CONFIG['device_map']
            else:
                # CPU模式
                model_kwargs["torch_dtype"] = torch.float32
                
            # 加载生成模型
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                MODEL_CONFIG['llm_model'],
                **model_kwargs
            )
            
            # 如果是CPU模式且没有自动设备映射
            if self.device == "cpu":
                self.llm_model = self.llm_model.to("cpu")
            elif self.device == "cuda" and not MODEL_CONFIG.get("load_in_8bit", False) and MODEL_CONFIG['device_map'] is None:
                self.llm_model = self.llm_model.cuda()
            
            # 设置为评估模式
            self.llm_model.eval()
            
            self.logger.info("大语言模型加载成功")
            return True
        except Exception as e:
            self.logger.error(f"大语言模型加载失败: {e}")
            self.logger.warning("将启用备用分析模式（无大模型深度分析）")
            return False
    
    def transcribe_audio(self, audio_path: str) -> Dict[str, Any]:
        """语音转文本"""
        if not self.whisper_model:
            raise RuntimeError("Whisper模型未加载")
        
        try:
            result = self.whisper_model.transcribe(
                audio_path,
                language="zh",  # 主要针对中文
                task="transcribe"
            )
            
            return {
                "text": result["text"],
                "segments": result.get("segments", []),
                "language": result.get("language", "zh")
            }
        except Exception as e:
            self.logger.error(f"语音转文本失败: {e}")
            raise
    
    def analyze_text_risk(self, text: str) -> Dict[str, Any]:
        """分析文本的政治风险"""
        if not self.llm_model or not self.llm_tokenizer:
            raise RuntimeError("大语言模型未加载")
        
        try:
            # 构建提示词
            prompt = self._build_risk_analysis_prompt(text)
            
            # 根据模型类型选择不同的生成方法
            model_name = MODEL_CONFIG['llm_model'].lower()
            
            if "gemma" in model_name:
                # Gemma模型使用chat template
                messages = [{"role": "user", "content": prompt}]
                text_input = self.llm_tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                model_inputs = self.llm_tokenizer([text_input], return_tensors="pt")
                if self.device == "cuda":
                    model_inputs = model_inputs.to("cuda")
                
                generated_ids = self.llm_model.generate(
                    model_inputs.input_ids,
                    max_new_tokens=512,
                    temperature=MODEL_CONFIG['temperature'],
                    do_sample=True,
                    pad_token_id=self.llm_tokenizer.eos_token_id
                )
                
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]
                
                response = self.llm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            else:
                # 通用的generate接口
                inputs = self.llm_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MODEL_CONFIG['max_length'])
                if self.device == "cuda":
                    inputs = inputs.to("cuda")
                
                with torch.no_grad():
                    outputs = self.llm_model.generate(
                        **inputs,
                        max_new_tokens=512,
                        temperature=MODEL_CONFIG['temperature'],
                        do_sample=True,
                        pad_token_id=self.llm_tokenizer.eos_token_id
                    )
                
                response = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
                # 移除输入部分
                response = response.replace(prompt, "").strip()
            
            # 解析结果
            return self._parse_risk_analysis(response)
        except Exception as e:
            self.logger.error(f"文本风险分析失败: {e}")
            raise
    
    def _build_risk_analysis_prompt(self, text: str) -> str:
        """构建风险分析提示词"""
        prompt = f"""You are a professional ideological risk analysis expert. Please analyze whether the following text contains political risks or ideological issues.

Analysis requirements:
1. Identify possible politically sensitive content
2. Assess ideological tendencies
3. Detect value conflicts
4. Give risk level (low risk/medium risk/high risk)
5. Provide specific risk points

Text to analyze:
{text}

Please return the analysis results in JSON format with the following fields:
- risk_level: Risk level
- risk_score: Risk score (0-100)
- key_issues: List of key issues
- suggestions: Improvement suggestions
- detailed_analysis: Detailed analysis
"""
        return prompt
    
    def _parse_risk_analysis(self, response: str) -> Dict[str, Any]:
        """解析风险分析结果"""
        try:
            # 尝试解析JSON格式的响应
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            else:
                # 如果无法解析JSON，返回默认格式
                return {
                    "risk_level": "unknown",
                    "risk_score": 0,
                    "key_issues": [],
                    "suggestions": [],
                    "detailed_analysis": response
                }
        except Exception as e:
            self.logger.error(f"解析分析结果失败: {e}")
            return {
                "risk_level": "parse_error",
                "risk_score": 0,
                "key_issues": ["Result parsing failed"],
                "suggestions": ["Please re-analyze"],
                "detailed_analysis": response
            }
    
    def initialize_models(self) -> bool:
        """初始化所有模型"""
        self.logger.info("开始初始化AI模型...")
        
        whisper_success = self.load_whisper_model()
        llm_success = self.load_llm_model()
        
        if whisper_success:
            if llm_success:
                self.logger.info("所有模型初始化成功")
            else:
                self.logger.warning("语音识别模型加载成功，大语言模型加载失败，系统将以基础模式运行")
            return True
        else:
            self.logger.error("语音识别模型加载失败，系统无法正常运行")
            return False
