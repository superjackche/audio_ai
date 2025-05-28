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
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['TRANSFORMERS_OFFLINE'] = '0'  # 允许在线下载

    # 新增：尝试解决ALSA问题和改善输出
    os.environ['PYTHONUNBUFFERED'] = '1' # 确保输出立即刷新
    os.environ['SDL_AUDIODRIVER'] = 'dummy' # 尝试阻止SDL（如果作为依赖存在）使用ALSA
    
    # 设置日志级别为WARNING减少输出 (Python logging, not Uvicorn)
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

def preload_models():
    """预加载AI模型并返回模型管理器实例"""
    try:
        print("\n🚀 开始预加载AI模型...")
        print("   注意: 首次运行可能需要下载模型文件，请耐心等待")
        
        # 导入并初始化模型管理器
        from models.simple_model_manager import SimpleModelManager
        
        model_manager = SimpleModelManager()
        success = model_manager.initialize_models()
        
        if success:
            print("🎉 AI模型预加载成功！")
            return model_manager  # 返回成功初始化的模型管理器
        else:
            print("⚠️  AI模型预加载失败")
            return None
            
    except Exception as e:
        print(f"❌ AI模型预加载异常: {e}")
        import traceback
        traceback.print_exc()
        return None

def launch_service(preloaded_model_manager=None):
    """启动Web服务"""
    print("\n🌐 启动Web服务...")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("🌐 服务地址: http://localhost:8080")
    print("📚 API文档: http://localhost:8080/docs")
    print("🔧 状态检查: http://localhost:8080/api/status")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    # 如果有预加载的模型管理器，传递给FastAPI应用
    if preloaded_model_manager:
        print("✅ 将预加载的模型管理器传递给Web应用")
        try:
            # 导入FastAPI应用的设置函数
            from app.main_new import set_global_model_manager
            set_global_model_manager(preloaded_model_manager)
            print("✅ 预加载模型已设置给Web应用")
        except Exception as e:
            print(f"⚠️ 设置预加载模型失败: {e}")
            print("   Web应用将使用懒加载机制")
    else:
        print("⚠️ 无预加载模型，Web应用将使用懒加载机制")
    
    print("✅ 系统准备就绪！")
    
    try:
        # 启动Web服务
        uvicorn.run(
            "app.main_new:app",
            host="0.0.0.0",
            port=8080,
            reload=False,
            log_level="info",
            access_log=False
        )
    except KeyboardInterrupt:
        print("\n🛑 服务已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        # 详细打印异常信息
        import traceback
        traceback.print_exc()
        sys.exit(1)

def main():
    """主函数"""
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║               🚀 AI语音政治风险监控系统  超快启动器             ║
    ║                                                               ║
    ║           ⚡ 模型先行 | 🤖 AI优先 | 🌐 Web后启                ║
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
    
    # 预加载模型
    preloaded_manager = preload_models()
    if not preloaded_manager:
        print("❌ 模型预加载失败，但仍会启动Web服务...")
        print("   Web应用将使用懒加载机制")
    
    # 启动Web服务
    launch_service(preloaded_manager)

if __name__ == "__main__":
    main()
