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

# 项目根目录
BASE_DIR = Path(__file__).parent.parent.absolute()

# 模型配置
MODEL_CONFIG = {
    "whisper_model": os.getenv("WHISPER_MODEL", "base"),  # 语音识别模型（支持多语言）
    "llm_model": os.getenv("MODEL_NAME", "Qwen/Qwen3-0.6B"),  # Qwen3 0.6B 模型
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "max_length": 1024,  # 减少最大长度，适合小模型
    "temperature": 0.7,
    "trust_remote_code": True,  # Qwen需要trust_remote_code
    "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
    "low_cpu_mem_usage": True,
    "device_map": "auto" if torch.cuda.is_available() else None,
    "load_in_8bit": False,  # 小模型不需要量化
}

# 数据路径
DATA_PATHS = {
    "models_dir": BASE_DIR / "models",
    "data_dir": BASE_DIR / "data",
    "upload_dir": BASE_DIR / "data" / "uploads",
    "output_dir": BASE_DIR / "data" / "outputs",
    "logs_dir": BASE_DIR / "data" / "logs",
}

# Web服务配置
WEB_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "reload": True,
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
