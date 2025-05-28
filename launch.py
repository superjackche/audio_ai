#!/usr/bin/env python3
"""
🚀 AI语音政治风险监控系统 - 超快速启动器
==========================================

优化启动流程，前端立即可用，模型后台加载
"""

import os
import sys
import time
import uvicorn
import logging
from pathlib import Path

def setup_environment():
    """快速设置环境"""
    print("🚀 正在启动AI语音政治风险监控系统...")
    
    # 设置工作目录
    os.chdir(Path(__file__).parent)
    
    # 设置环境变量
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['TRANSFORMERS_OFFLINE'] = '0'  # 允许在线下载
    
    # 设置日志级别为WARNING减少输出
    logging.basicConfig(level=logging.WARNING)
    
    print("✅ 环境设置完成")

def kill_existing_processes():
    """快速清理已存在的进程"""
    os.system("pkill -f 'uvicorn.*8080' 2>/dev/null || true")
    os.system("pkill -f 'python.*main_new' 2>/dev/null || true")
    time.sleep(0.5)  # 很短的等待

def check_basic_requirements():
    """极简检查"""
    try:
        import torch
        import transformers
        import fastapi
        print(f"✅ 核心库: PyTorch {torch.__version__}, CUDA可用: {torch.cuda.is_available()}")
        return True
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        return False

def launch_service():
    """立即启动服务"""
    print("\n🌐 立即启动Web服务...")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("🌐 服务地址: http://localhost:8080")
    print("📚 API文档: http://localhost:8080/docs")
    print("🔧 状态检查: http://localhost:8080/api/status")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("📝 注意: 前端立即可用，AI模型在后台加载中...")
    
    try:
        # 立即启动，无任何延迟
        uvicorn.run(
            "app.main_new:app",
            host="0.0.0.0",
            port=8080,
            reload=False,
            log_level="error",  # 减少日志输出
            access_log=False    # 关闭访问日志减少输出
        )
    except KeyboardInterrupt:
        print("\n🛑 服务已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        sys.exit(1)

def main():
    """主函数"""
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║               🚀 AI语音政治风险监控系统  超快启动器             ║
    ║                                                               ║
    ║           ⚡ 零等待启动 | 🎯 前端秒开 | 🔧 模型后台加载        ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # 清理进程
    kill_existing_processes()
    
    # 快速设置
    setup_environment()
    
    # 极简检查
    if not check_basic_requirements():
        print("❌ 环境检查失败")
        sys.exit(1)
    
    # 立即启动服务
    launch_service()

if __name__ == "__main__":
    main()
