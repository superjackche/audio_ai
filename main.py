#!/usr/bin/env python3
"""
AI语音政治风险监测系统
主启动文件
"""

import sys
import os
from pathlib import Path
import logging

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# 导入必要模块
from config.settings import WEB_CONFIG, DATA_PATHS
from utils.audio_utils import Logger
from models.training_data import create_training_data

def setup_environment():
    """设置运行环境"""
    # 创建必要的目录
    for path in DATA_PATHS.values():
        path.mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    Logger.setup_logging(DATA_PATHS["logs_dir"])
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 50)
    logger.info("AI语音政治风险监测系统启动")
    logger.info("=" * 50)
    
    return logger

def check_dependencies():
    """检查依赖项"""
    logger = logging.getLogger(__name__)
    
    try:
        import torch
        logger.info(f"PyTorch版本: {torch.__version__}")
        
        if torch.cuda.is_available():
            logger.info(f"CUDA可用: {torch.cuda.get_device_name()}")
            logger.info(f"CUDA版本: {torch.version.cuda}")
        else:
            logger.warning("CUDA不可用，将使用CPU运行")
        
        import transformers
        logger.info(f"Transformers版本: {transformers.__version__}")
        
        import whisper
        logger.info("Whisper语音识别可用")
        
        import fastapi
        logger.info(f"FastAPI版本: {fastapi.__version__}")
        
        return True
        
    except ImportError as e:
        logger.error(f"缺少依赖项: {e}")
        logger.error("请运行: pip install -r requirements.txt")
        return False

def initialize_training_data():
    """初始化训练数据"""
    logger = logging.getLogger(__name__)
    
    # 检查训练数据是否存在
    train_file = DATA_PATHS["data_dir"] / "train_dataset.json"
    
    if not train_file.exists():
        logger.info("未找到训练数据，开始生成...")
        try:
            create_training_data()
            logger.info("训练数据生成完成")
        except Exception as e:
            logger.error(f"训练数据生成失败: {e}")
            return False
    else:
        logger.info("训练数据已存在")
    
    return True

def main():
    """主函数"""
    # 设置环境
    logger = setup_environment()
    
    # 检查依赖
    if not check_dependencies():
        logger.error("依赖项检查失败，程序退出")
        sys.exit(1)
    
    # 初始化训练数据
    if not initialize_training_data():
        logger.error("训练数据初始化失败，程序退出")
        sys.exit(1)
    
    # 启动Web服务
    try:
        logger.info("正在启动Web服务...")
        logger.info(f"访问地址: http://localhost:{WEB_CONFIG['port']}")
        
        # 导入并启动FastAPI应用
        import uvicorn
        from app.main import app
        
        uvicorn.run(
            app,
            host=WEB_CONFIG["host"],
            port=WEB_CONFIG["port"],
            reload=False,  # 生产环境关闭自动重载
            log_level="info"
        )
        
    except KeyboardInterrupt:
        logger.info("用户中断，程序退出")
    except Exception as e:
        logger.error(f"启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
