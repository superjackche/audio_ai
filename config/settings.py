import os
import torch
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 设置HuggingFace镜像源
os.environ['HF_ENDPOINT'] = os.getenv('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ['HUGGINGFACE_HUB_CACHE'] = os.getenv('HUGGINGFACE_HUB_CACHE', './models/cache')
os.environ['HF_HOME'] = os.getenv('HF_HOME', './models/cache')

# 导入性能优化配置
try:
    from .performance_optimization import get_production_config, UltraPerformanceOptimizer
    
    # 应用性能优化
    optimizer = UltraPerformanceOptimizer()
    optimizer.apply_all_optimizations()
    PERFORMANCE_CONFIG = get_production_config()
except ImportError:
    PERFORMANCE_CONFIG = None

# 项目根目录
BASE_DIR = Path(__file__).parent.parent.absolute()

# 模型配置
_llm_model_name = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct")  # 使用更轻量的3B模型避免卡死

# 动态配置：根据性能优化配置调整
if PERFORMANCE_CONFIG:
    try:
        optimal_config = PERFORMANCE_CONFIG["model_config"]["model_optimization"]
        _llm_model_name = optimal_config["primary_model"]
    except (KeyError, TypeError):
        # 如果性能配置有问题，使用默认值
        _llm_model_name = "Qwen/Qwen2.5-7B-Instruct"

MODEL_CONFIG = {
    "default": {
        "llm_model": _llm_model_name,
        "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "max_length": 2048,
        "temperature": 0.1,  # 更低温度提高一致性和速度
        "trust_remote_code": True,
        "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
        "low_cpu_mem_usage": True,
        "device_map": "auto" if torch.cuda.is_available() else None,
        
        # 性能优化配置
        "load_in_4bit": True,  # 4bit量化加速
        "bnb_4bit_compute_dtype": torch.float16,
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_quant_type": "nf4",
        "use_flash_attention_2": True,
        "attn_implementation": "flash_attention_2",
        
        # 推理优化
        "max_new_tokens": 512,
        "top_p": 0.9,
        "top_k": 50,
        "do_sample": True,
        "use_cache": True,
        "pad_token_id": 0,
        "eos_token_id": 1,
    }
}

# 数据路径
DATA_PATHS = {
    "models_dir": BASE_DIR / "models",
    "data_dir": BASE_DIR / "data",
    "upload_dir": BASE_DIR / "data" / "uploads",
    "output_dir": BASE_DIR / "data" / "outputs",
    "logs_dir": BASE_DIR / "data" / "logs",
}

# Web服务配置 - 仅支持HTTP
WEB_CONFIG = {
    "host": "0.0.0.0",  # 监听所有网络接口
    "port": 8080,       # HTTP端口
    "reload": False,    # 生产环境关闭热重载
    "protocol": "http", # 明确指定HTTP协议
}

# 政治风险关键词配置
RISK_KEYWORDS = {
    "高风险": [
        "政治制度", "政权更迭", "民主化", "独立运动", "分裂主义",
        "人权问题", "宗教极端", "恐怖主义", "颠覆国家",
    ],
    "中风险": [
        "政治改革", "社会制度", "意识形态", "价值观冲突",
        "文化差异", "治理模式", "社会运动",
    ],
    "低风险": [
        "政策建议", "学术讨论", "理论研究", "比较分析",
    ]
}

# 评分权重配置
SCORING_WEIGHTS = {
    "keyword_score": 0.4,      # 关键词匹配分数权重
    "context_score": 0.3,      # 上下文语义分数权重
    "frequency_score": 0.2,    # 频率分数权重
    "sentiment_score": 0.1,    # 情感倾向分数权重
}

# 创建必要目录
for path in DATA_PATHS.values():
    path.mkdir(parents=True, exist_ok=True)
