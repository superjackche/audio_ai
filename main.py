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
        
        # 导入FastAPI应用实例
        from app.main_new import app
        import uvicorn

        # SSL证书路径 (相对于PROJECT_ROOT)
        # PROJECT_ROOT is defined at the top of this file
        ssl_keyfile_path = PROJECT_ROOT / "config" / "key.pem"
        ssl_certfile_path = PROJECT_ROOT / "config" / "cert.pem"

        if ssl_keyfile_path.exists() and ssl_certfile_path.exists():
            logger.info(f"检测到SSL证书，将以 HTTPS 模式启动。")
            logger.info(f"  证书文件: {ssl_certfile_path}")
            logger.info(f"  密钥文件: {ssl_keyfile_path}")
            logger.info(f"访问地址: https://{WEB_CONFIG['host']}:{WEB_CONFIG['port']}")
            uvicorn.run(
                app,
                host=WEB_CONFIG["host"],
                port=WEB_CONFIG["port"],
                reload=False,
                log_level="info",
                ssl_keyfile=str(ssl_keyfile_path),
                ssl_certfile=str(ssl_certfile_path)
            )
        else:
            logger.warning("警告: SSL 证书文件 (key.pem, cert.pem) 未在 config 目录中找到。")
            logger.warning(f"  - 期望的 key.pem 路径: {ssl_keyfile_path}")
            logger.warning(f"  - 期望的 cert.pem 路径: {ssl_certfile_path}")
            logger.warning("请使用 OpenSSL 生成自签名证书，或提供有效的证书路径以启用 HTTPS。")
            logger.warning("示例命令 (在项目根目录下执行): openssl req -x509 -newkey rsa:4096 -nodes -out config/cert.pem -keyout config/key.pem -days 365")
            logger.info("应用将以 HTTP 模式启动。")
            logger.info(f"访问地址: http://{WEB_CONFIG['host']}:{WEB_CONFIG['port']}")
            uvicorn.run(
                app,
                host=WEB_CONFIG["host"],
                port=WEB_CONFIG["port"],
                reload=False,
                log_level="info"
            )
        
    except KeyboardInterrupt:
        logger.info("用户中断，程序退出")
    except Exception as e:
        logger.error(f"启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
