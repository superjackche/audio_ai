import torch
import whisper
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import logging
from typing import Optional, List, Dict, Any
from config.settings import MODEL_CONFIG, DATA_PATHS

class ModelManager:
    """AI模型管理器，负责加载和管理语音识别和大语言模型"""
    def __init__(self):
        # 设置日志（需要在其他初始化之前）
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.whisper_model: Optional[whisper.Whisper] = None
        self.llm_model: Optional[Any] = None
        self.llm_tokenizer: Optional[Any] = None
        self.device = self._get_device()
        
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
            model_name = MODEL_CONFIG['llm_model']
            self.logger.info(f"正在加载大语言模型: {model_name}")
              # 对于 Qwen 模型，使用 AutoModelForCausalLM
            if "qwen" in model_name.lower():
                # 加载tokenizer
                self.llm_tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=MODEL_CONFIG.get('trust_remote_code', True),
                    cache_dir=str(DATA_PATHS['models_dir'] / 'cache')
                )
                
                # 添加 pad_token
                if self.llm_tokenizer.pad_token is None:
                    self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
                
                # 配置模型加载参数
                model_kwargs = {
                    "trust_remote_code": MODEL_CONFIG.get('trust_remote_code', True),
                    "cache_dir": str(DATA_PATHS['models_dir'] / 'cache'),
                    "torch_dtype": MODEL_CONFIG.get('torch_dtype', torch.float16 if self.device == "cuda" else torch.float32),
                    "low_cpu_mem_usage": MODEL_CONFIG.get('low_cpu_mem_usage', True),
                }
                
                # 设备映射
                if self.device == "cuda":
                    model_kwargs["device_map"] = MODEL_CONFIG.get('device_map', 'auto')
                else:
                    model_kwargs["torch_dtype"] = torch.float32
                
                # 加载模型 - 对于 Qwen 使用 AutoModelForCausalLM
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    **model_kwargs
                )
            elif "gemma" in model_name.lower():
                # 加载tokenizer
                self.llm_tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=MODEL_CONFIG.get('trust_remote_code', True),
                    cache_dir=str(DATA_PATHS['models_dir'] / 'cache')
                )
                
                # 添加 pad_token
                if self.llm_tokenizer.pad_token is None:
                    self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
                
                # 配置模型加载参数
                model_kwargs = {
                    "trust_remote_code": MODEL_CONFIG.get('trust_remote_code', True),
                    "cache_dir": str(DATA_PATHS['models_dir'] / 'cache'),
                    "torch_dtype": MODEL_CONFIG.get('torch_dtype', torch.float16 if self.device == "cuda" else torch.float32),
                    "low_cpu_mem_usage": MODEL_CONFIG.get('low_cpu_mem_usage', True),
                }
                
                # 设备映射
                if self.device == "cuda":
                    model_kwargs["device_map"] = MODEL_CONFIG.get('device_map', 'auto')
                else:
                    model_kwargs["torch_dtype"] = torch.float32
                
                # 加载模型 - 对于 Gemma 使用 AutoModelForCausalLM
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    **model_kwargs
                )
            else:
                # 其他模型的加载方式
                self.llm_tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    cache_dir=str(DATA_PATHS['models_dir'] / 'cache')
                )
                
                model_kwargs = {
                    "trust_remote_code": True,
                    "cache_dir": str(DATA_PATHS['models_dir'] / 'cache'),
                    "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                }
                
                if self.device == "cpu":
                    model_kwargs["torch_dtype"] = torch.float32
                else:
                    model_kwargs["device_map"] = "auto"
                
                self.llm_model = AutoModel.from_pretrained(
                    model_name,
                    **model_kwargs
                )
            
            # 设置为评估模式
            self.llm_model.eval()
            
            self.logger.info(f"大语言模型 {model_name} 加载成功")
            return True
        except Exception as e:
            self.logger.error(f"大语言模型加载失败: {e}")
            self.logger.warning("将启用备用分析模式（无大模型深度分析）")
            return False
    
    def transcribe_audio(self, audio_path: str, progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """语音转文本，自动检测语言"""
        if not self.whisper_model:
            self.logger.error("Whisper模型未加载，无法进行语音转文本")
            raise RuntimeError("Whisper模型未加载")
        
        self.logger.info(f"开始语音转文本: {audio_path}")
        try:
            # 自动检测语言，并在CUDA可用时使用fp16加速
            result = self.whisper_model.transcribe(
                audio_path,
                language=None,  # 自动检测语言
                task="transcribe",
                fp16=torch.cuda.is_available() # 使用fp16在GPU上加速
            )
            
            detected_language = result.get("language", "unknown")
            self.logger.info(f"语音转文本完成。检测到的语言: {detected_language}")
            
            if progress_callback:
                progress_callback(100.0, f"语音转文本完成，检测到语言: {detected_language}")

            return {
                "text": result["text"],
                "segments": result.get("segments", []),
                "language": detected_language 
            }
        except Exception as e:
            self.logger.error(f"语音转文本失败: {e}")
            if progress_callback:
                progress_callback(-1.0, f"语音转文本失败: {e}")
            raise
    
    def analyze_text_risk(self, text: str, language: str = "zh") -> Dict[str, Any]: # Added language parameter
        """分析文本的政治风险，根据语言选择提示"""
        if not self.llm_model or not self.llm_tokenizer:
            self.logger.error("大语言模型未加载，无法进行文本风险分析")
            raise RuntimeError("大语言模型未加载")
        
        self.logger.info(f"开始文本风险分析，语言: {language}")
        try:
            prompt = self._build_risk_analysis_prompt(text, language)
            # ... (rest of the generation logic remains the same) ...
            model_name = MODEL_CONFIG['llm_model'].lower()
            
            if "gemma" in model_name:
                messages = [{"role": "user", "content": prompt}]
                formatted_text = self.llm_tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                model_inputs = self.llm_tokenizer([formatted_text], return_tensors="pt")
                if self.device == "cuda":
                    model_inputs = model_inputs.to("cuda")
                generated_ids = self.llm_model.generate(
                    model_inputs.input_ids,
                    max_new_tokens=512,
                    temperature=MODEL_CONFIG['temperature'],
                    do_sample=True,
                    pad_token_id=self.llm_tokenizer.eos_token_id,
                    eos_token_id=self.llm_tokenizer.eos_token_id
                )
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]
                response = self.llm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            elif "chatglm" in model_name:
                response, _ = self.llm_model.chat(
                    self.llm_tokenizer,
                    prompt,
                    history=[],
                    max_length=MODEL_CONFIG['max_length'],
                    temperature=MODEL_CONFIG['temperature']
                )
            elif "qwen" in model_name:
                messages = [{"role": "user", "content": prompt}]
                text_for_model = self.llm_tokenizer.apply_chat_template( # Renamed 'text' to 'text_for_model' to avoid conflict
                    messages, tokenize=False, add_generation_prompt=True
                )
                model_inputs = self.llm_tokenizer([text_for_model], return_tensors="pt")
                if self.device == "cuda":
                    model_inputs = model_inputs.to("cuda")
                generated_ids = self.llm_model.generate(
                    model_inputs.input_ids,
                    max_new_tokens=512,
                    temperature=MODEL_CONFIG['temperature'],
                    do_sample=True
                )
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]
                response = self.llm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            else:
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
                response = response.replace(prompt, "").strip()
            
            self.logger.info("文本风险分析完成")
            return self._parse_risk_analysis(response, language)
        except Exception as e:
            self.logger.error(f"文本风险分析失败: {e}")
            raise
    
    def _build_risk_analysis_prompt(self, text: str, language: str = "zh") -> str:
        """构建风险分析提示词，根据语言选择模板"""
        if language.lower().startswith("en"):
            # English Prompt
            prompt = f"""
You are an expert in analyzing text for potential risks, including misinformation, sensitive topics, and controversial content. Please analyze the following text.

Analysis requirements:
1. Identify any potentially sensitive or controversial statements.
2. Assess the general tone and sentiment.
3. Provide a risk level (Low/Medium/High).
4. Briefly explain the key reasons for your assessment.

Text to analyze:
{text}

Please return the analysis in JSON format with the following fields:
- risk_level: (Low/Medium/High)
- risk_score: (0-100, where 100 is highest risk)
- key_issues: (List of strings describing key issues)
- suggestions: (List of strings for improvement or caution)
- detailed_analysis: (A brief overall analysis)
"""
        else:  # Default to Chinese prompt
            prompt = f"""
你是一个专业的意识形态风险分析专家。请分析以下文本内容是否存在政治风险或意识形态问题。

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
"""
        return prompt
    
    def _parse_risk_analysis(self, response: str, language: str = "zh") -> Dict[str, Any]: # Added language parameter
        """解析风险分析结果"""
        try:
            import json
            import re
            
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            else:
                self.logger.warning(f"无法从LLM响应中提取JSON: {response}")
                default_message = "Could not parse analysis from LLM response." if language.startswith("en") else "无法从大模型响应中解析分析结果。"
                return {
                    "risk_level": "Unknown" if language.startswith("en") else "未知",
                    "risk_score": 0,
                    "key_issues": [default_message],
                    "suggestions": [],
                    "detailed_analysis": response
                }
        except Exception as e:
            self.logger.error(f"解析分析结果失败: {e}")
            default_error_message = "Error parsing analysis result." if language.startswith("en") else "结果解析失败"
            return {
                "risk_level": "Error" if language.startswith("en") else "解析错误",
                "risk_score": 0,
                "key_issues": [default_error_message],
                "suggestions": ["Please try again." if language.startswith("en") else "请重新分析"],
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
