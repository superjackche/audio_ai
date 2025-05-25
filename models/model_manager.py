import torch
import whisper
from transformers import AutoTokenizer, AutoModel
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
            self.logger.info(f"正在加载大语言模型: {MODEL_CONFIG['llm_model']}")
            
            # 加载tokenizer
            self.llm_tokenizer = AutoTokenizer.from_pretrained(
                MODEL_CONFIG['llm_model'],
                trust_remote_code=True,
                cache_dir=str(DATA_PATHS['models_dir'])
            )
            
            # 配置模型加载参数
            model_kwargs = {
                "trust_remote_code": True,
                "cache_dir": str(DATA_PATHS['models_dir']),
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
            }
            
            # 如果是CPU模式或者配置了量化
            if self.device == "cpu" or MODEL_CONFIG.get("load_in_8bit", False):
                if self.device == "cuda" and MODEL_CONFIG.get("load_in_8bit", False):
                    model_kwargs["load_in_8bit"] = True
                    model_kwargs["device_map"] = "auto"
                else:
                    model_kwargs["device_map"] = "cpu"
            else:
                model_kwargs["device_map"] = "auto"
            
            # 加载模型
            self.llm_model = AutoModel.from_pretrained(
                MODEL_CONFIG['llm_model'],
                **model_kwargs
            )
            
            # 如果不是自动设备映射，手动移动到设备
            if "device_map" not in model_kwargs or model_kwargs["device_map"] == "cpu":
                if self.device == "cuda":
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
            
            if "chatglm" in model_name:
                # ChatGLM的chat接口
                response, _ = self.llm_model.chat(
                    self.llm_tokenizer,
                    prompt,
                    history=[],
                    max_length=MODEL_CONFIG['max_length'],
                    temperature=MODEL_CONFIG['temperature']
                )
            elif "qwen" in model_name:
                # Qwen的generate接口
                messages = [{"role": "user", "content": prompt}]
                text = self.llm_tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                model_inputs = self.llm_tokenizer([text], return_tensors="pt")
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
    
    def _parse_risk_analysis(self, response: str) -> Dict[str, Any]:
        """解析风险分析结果"""
        try:
            # 尝试解析JSON格式的响应
            import json
            import re
            
            # 提取JSON部分
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            else:
                # 如果无法解析JSON，返回默认格式
                return {
                    "risk_level": "未知",
                    "risk_score": 0,
                    "key_issues": [],
                    "suggestions": [],
                    "detailed_analysis": response
                }
        except Exception as e:
            self.logger.error(f"解析分析结果失败: {e}")
            return {
                "risk_level": "解析错误",
                "risk_score": 0,
                "key_issues": ["结果解析失败"],
                "suggestions": ["请重新分析"],
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
