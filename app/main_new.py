#!/usr/bin/env python3
"""
AI语音政治风险监控系统 - FastAPI Web应用
=====================================

Copyright (c) 2025 superjackche
GitHub: https://github.com/superjackche/audio_ai
Licensed under the MIT License. See LICENSE file for details.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import JSONResponse, HTMLResponse
import uvicorn
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
import logging
from typing import Optional
import tempfile
from pydantic import BaseModel

# 添加父目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 导入自定义模块
from models.simple_model_manager import SimpleModelManager
from models.risk_analyzer import PoliticalRiskAnalyzer
from utils.audio_utils import AudioProcessor, FileManager, Logger
from config.settings import WEB_CONFIG, DATA_PATHS

# Define the secure base directory for server-side audio files
SERVER_AUDIO_FILES_BASE_DIR = Path("/new_disk/cwh/audio_ai/server_audio_files/")
SUPPORTED_AUDIO_EXTENSIONS = ['.wav', '.mp3', '.m4a', '.flac']

# 初始化FastAPI应用
app = FastAPI(
    title="AI语音政治风险监测系统",
    description="基于SenseVoice中文语音识别 + Qwen2.5-7B-Instruct的快速语音政治风险分析系统",
    version="2.0.0"
)

# 获取 app/main.py 文件所在的目录
APP_MAIN_PY_DIR = Path(__file__).resolve().parent

# 更新静态文件和模板目录的路径
app.mount("/static", StaticFiles(directory=APP_MAIN_PY_DIR.parent / "static"), name="static")
templates = Jinja2Templates(directory=APP_MAIN_PY_DIR.parent / "templates")

class ServerFileRequest(BaseModel):
    filename: str

class TextInput(BaseModel):  # Added TextInput definition
    text: str

# 全局变量
model_manager: Optional[SimpleModelManager] = None
risk_analyzer: Optional[PoliticalRiskAnalyzer] = None
analysis_results = []

def set_global_model_manager(manager: SimpleModelManager):
    global model_manager
    model_manager = manager
    logging.getLogger(__name__).info(f"Global model_manager set via set_global_model_manager. Speech loaded: {model_manager.whisper_pipeline is not None}, LLM loaded: {model_manager.llm_model is not None}")

@app.on_event("startup")
async def startup_event():
    global model_manager, risk_analyzer
    logger = logging.getLogger(__name__)
    logger.info("FastAPI startup_event triggered.")

    if model_manager is None:
        logger.warning("模型管理器未被预加载。将初始化新的实例并依赖懒加载。")
        # Create a new instance; models will be loaded on first use by API endpoints.
        temp_model_manager = SimpleModelManager() # Constructor doesn't load models by default
        set_global_model_manager(temp_model_manager) # Set it globally
        # Optionally, one could attempt a full load here if that's desired for non-preloaded scenarios
        # logger.info("Attempting to initialize models in startup_event as a fallback for non-preloaded manager...")
        # if not model_manager.initialize_models():
        #     logger.error("Fallback model initialization in startup_event failed.")
    else:
        logger.info("确认使用预加载的模型管理器。")
        if model_manager.whisper_pipeline and model_manager.llm_model:
            logger.info("预加载的模型管理器已成功加载语音和LLM模型。")
        else:
            logger.warning(f"预加载的模型管理器状态: 语音模型加载 = {model_manager.whisper_pipeline is not None}, LLM模型加载 = {model_manager.llm_model is not None}。部分模型可能仍需懒加载。")

    # Initialize other components if they haven't been set (e.g., by a future preloading mechanism for them)
    if risk_analyzer is None:
        try:
            # 将模型管理器传递给风险分析器，让它能够使用AI模型
            risk_analyzer = PoliticalRiskAnalyzer(model_manager=model_manager)
            logger.info("PoliticalRiskAnalyzer initialized with AI model support.")
        except Exception as e:
            logger.error(f"PoliticalRiskAnalyzer initialization failed: {e}")
            # 如果初始化失败，尝试不带模型管理器初始化
            try:
                risk_analyzer = PoliticalRiskAnalyzer()
                logger.warning("PoliticalRiskAnalyzer initialized without AI model support.")
            except Exception as e2:
                logger.error(f"PoliticalRiskAnalyzer fallback initialization also failed: {e2}")
                risk_analyzer = None # Ensure it's None if init fails
            
    logger.info("Web服务启动事件处理完成。")
    print("🌐 Web服务组件已通过startup_event (重新)初始化/检查。")

def get_risk_analyzer() -> Optional[PoliticalRiskAnalyzer]:  # Added get_risk_analyzer function
    """获取全局风险分析器实例，如果未初始化则尝试初始化。"""
    global risk_analyzer
    global model_manager # Ensure model_manager is also accessible
    logger = logging.getLogger(__name__)

    if risk_analyzer is None:
        logger.warning("get_risk_analyzer: Risk analyzer is None. Attempting to initialize.")
        if model_manager is None:
            logger.warning("get_risk_analyzer: Model manager is also None. Initializing a new SimpleModelManager.")
            model_manager = SimpleModelManager()
            # Potentially initialize models if SimpleModelManager doesn't do it on construction or lazy load
            # For example: model_manager.initialize_models()

        try:
            risk_analyzer = PoliticalRiskAnalyzer(model_manager=model_manager)
            logger.info("get_risk_analyzer: PoliticalRiskAnalyzer initialized successfully with model_manager.")
        except Exception as e:
            logger.error(f"get_risk_analyzer: Failed to initialize PoliticalRiskAnalyzer: {e}")
            risk_analyzer = None # Explicitly set to None on failure
    
    # Ensure the existing risk_analyzer has a model_manager if it's supposed to.
    if risk_analyzer and risk_analyzer.model_manager is None and model_manager is not None:
        logger.warning("get_risk_analyzer: Existing risk_analyzer has no model_manager. Attempting to assign the global model_manager.")
        risk_analyzer.model_manager = model_manager
        logger.info(f"get_risk_analyzer: Assigned model_manager. Risk analyzer AI support: {risk_analyzer.model_manager is not None}")

    return risk_analyzer

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """主页"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/status")
async def get_system_status():
    """获取系统状态"""
    logger = logging.getLogger(__name__)
    
    # 检查模型是否初始化
    sensevoice_ready = False
    llm_ready = False
    models_info = {
        "speech_model": "未加载",
        "llm_model": "未加载",
        "device": "未知"
    }
    
    if model_manager:
        try:
            # 检查设备信息
            models_info["device"] = getattr(model_manager, 'device', '未知')
            
            # 检查SenseVoice模型状态 - 检查所有可能的模型属性
            if hasattr(model_manager, 'whisper_pipeline') and model_manager.whisper_pipeline is not None:
                sensevoice_ready = True
                models_info["speech_model"] = "SenseVoice中文语音识别-已加载"
                print(f"✅ 状态检查: SenseVoice模型已加载 (whisper_pipeline存在)")
            else:
                print(f"⚠️ 状态检查: SenseVoice模型未加载")
                print(f"   - hasattr(model_manager, 'whisper_pipeline'): {hasattr(model_manager, 'whisper_pipeline') if model_manager else 'model_manager为None'}")
                if hasattr(model_manager, 'whisper_pipeline'):
                    print(f"   - model_manager.whisper_pipeline is not None: {model_manager.whisper_pipeline is not None}")
            
            # 检查LLM模型状态
            if hasattr(model_manager, 'llm_model') and model_manager.llm_model is not None:
                llm_ready = True
                if hasattr(model_manager, 'llm_model_path') and model_manager.llm_model_path:
                    models_info["llm_model"] = f"本地模型: {model_manager.llm_model_path.name}"
                else:
                    models_info["llm_model"] = getattr(model_manager, 'llm_model_name', 'LLM-已加载')
                print(f"✅ 状态检查: LLM模型已加载")
            else:
                print(f"⚠️ 状态检查: LLM模型未加载")
            
            # 如果模型还没初始化，提供懒加载状态
            if not sensevoice_ready or not llm_ready:
                if not sensevoice_ready:
                    models_info["speech_model"] = "SenseVoice等待加载"
                if not llm_ready:
                    models_info["llm_model"] = "LLM等待加载"
                    
        except Exception as e:
            logger.warning(f"模型状态检查出错: {e}")
    
    status = {
        "system_online": True,
        "speech_model_loaded": sensevoice_ready,
        "llm_model_loaded": llm_ready,
        "models_ready": sensevoice_ready and llm_ready,
        "processing_method": "sensevoice_qwen2.5_7b",
        "model_manager_initialized": model_manager is not None,
        "models_info": models_info,
        "timestamp": datetime.now().isoformat()
    }
    return JSONResponse(status)

async def analyze_audio_file(processing_file_path: str, original_filename: str, progress_callback: Optional[callable] = None) -> dict:
    """使用SenseVoice中文语音识别 + Qwen2.5-7B-Instruct 分析音频文件"""
    logger = logging.getLogger(__name__)
    try:
        # 使用SenseVoice + Qwen2.5模型组合处理音频
        if progress_callback:
            progress_callback(0.0, "开始使用SenseVoice + Qwen2.5快速模型分析音频...")

        # 使用FastModelManager处理音频
        fast_result = model_manager.process_audio_complete(processing_file_path)
        
        if progress_callback:
            progress_callback(70.0, "音频转文字和风险分析完成...")

        # 提取结果
        text = fast_result.get("text", "")
        detected_language = fast_result.get("language", "zh")
        risk_analysis = fast_result.get("risk_analysis", {})
        
        if not text.strip():
            if progress_callback:
                progress_callback(100.0, "未检测到有效音频内容")
            return {
                "error": "未检测到有效音频内容",
                "transcription": {"text": "", "language": detected_language}
            }
            
        if progress_callback:
            progress_callback(95.0, "生成简洁结果...") 
        
        # 简化的结果格式 - 只包含关键的风险评估信息
        risk_level = risk_analysis.get("risk_level", "未知")
        risk_score = risk_analysis.get("risk_score", 0)
        key_issues = risk_analysis.get("key_issues", [])

        # 获取音频时长
        from utils.audio_utils import AudioProcessor
        audio_duration = AudioProcessor.get_audio_duration(processing_file_path)

        result = {
            "analysis_id": f"analysis_{int(datetime.now().timestamp())}",
            "original_filename": original_filename,
            "timestamp": datetime.now().isoformat(),
            "risk_assessment": {
                "risk_level": risk_level,
                "risk_score": risk_score,
                "key_issues": key_issues[:3]  # 只显示前3个关键问题
            },
            "processing_info": {
                "text_length": len(text),
                "detected_language": detected_language,
                "processing_method": "sensevoice_qwen2.5_7b",
                "transcription_time": fast_result.get("transcription_time", 0),
                "analysis_time": fast_result.get("analysis_time", 0),
                "total_time": fast_result.get("total_processing_time", 0),
                "audio_duration": audio_duration  # 音频时长（秒）
            }
        }
        
        # 保存分析结果
        analysis_results.append(result)
        
        # 保存到文件
        output_file = DATA_PATHS["output_dir"] / f"{result['analysis_id']}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        if progress_callback:
            progress_callback(100.0, "分析完成")
        
        return result
        
    except Exception as e:
        logging.error(f"音频分析失败: {e}")
        if progress_callback:
            progress_callback(-1.0, f"分析失败: {e}")
        raise

@app.post("/api/upload-audio")
async def upload_audio(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    """上传音频文件进行分析"""
    # 懒加载检查 - 如果模型未加载则尝试加载
    if not model_manager:
        raise HTTPException(status_code=503, detail="模型管理器未初始化")
    
    # 检查并加载SenseVoice模型
    if not model_manager.whisper_pipeline:
        print("🎙️  触发SenseVoice模型懒加载...")
        if not model_manager.load_whisper_model():
            raise HTTPException(status_code=503, detail="SenseVoice语音识别模型加载失败")
    
    # 检查并加载LLM模型
    if not model_manager.llm_model:
        print("🧠 触发LLM模型懒加载...")
        if not model_manager.load_llm_model():
            raise HTTPException(status_code=503, detail="LLM模型加载失败")
    
    processing_file_path = None
    try:
        # 验证文件类型
        if not file.filename.lower().endswith(('.wav', '.mp3', '.m4a', '.flac')):
            raise HTTPException(status_code=400, detail="不支持的音频格式")
        
        file_content = await file.read()
        
        # 使用tempfile创建一个临时文件来处理
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            tmp_file.write(file_content)
            processing_file_path = tmp_file.name
        
        # 使用后台任务进行分析
        task_id = f"task_{int(datetime.now().timestamp())}"
        
        if not hasattr(app.state, 'tasks_status'):
            app.state.tasks_status = {}
        
        app.state.tasks_status[task_id] = {
            "status": "processing", 
            "progress": 0, 
            "message": "任务已提交", 
            "original_filename": file.filename
        }

        def progress_update_for_task(task_id: str, progress: float, message: str):
            if task_id in app.state.tasks_status:
                app.state.tasks_status[task_id]["progress"] = progress
                app.state.tasks_status[task_id]["message"] = message
                if progress == 100.0 or progress == -1.0:
                     app.state.tasks_status[task_id]["status"] = "completed" if progress == 100.0 else "failed"

        async def analyze_and_cleanup(temp_path: str, original_filename: str, current_task_id: str):
            try:
                result = await analyze_audio_file(
                    temp_path, 
                    original_filename, 
                    progress_callback=lambda p, m: progress_update_for_task(current_task_id, p, m)
                )
                app.state.tasks_status[current_task_id]["result"] = result
                app.state.tasks_status[current_task_id]["status"] = "completed"
                app.state.tasks_status[current_task_id]["progress"] = 100.0
            except Exception as e:
                app.state.tasks_status[current_task_id]["status"] = "failed"
                app.state.tasks_status[current_task_id]["error"] = str(e)
                app.state.tasks_status[current_task_id]["progress"] = -1.0
            finally:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)
        
        background_tasks.add_task(analyze_and_cleanup, processing_file_path, file.filename, task_id)
        
        return JSONResponse({
            "message": "音频处理已开始", 
            "task_id": task_id, 
            "status_url": f"/api/task-status/{task_id}"
        })
        
    except Exception as e:
        logging.error(f"音频上传处理启动失败: {e}")
        if processing_file_path and os.path.exists(processing_file_path):
            os.remove(processing_file_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/task-status/{task_id}")
async def get_task_status(task_id: str):
    """获取后台任务的状态"""
    if not hasattr(app.state, 'tasks_status') or task_id not in app.state.tasks_status:
        raise HTTPException(status_code=404, detail="任务未找到")
    return JSONResponse(app.state.tasks_status[task_id])

@app.get("/api/list-server-audio-files")
async def list_server_audio_files():
    """列出预定义服务器目录中的音频文件"""
    logger = logging.getLogger(__name__)
    audio_files = []
    if not SERVER_AUDIO_FILES_BASE_DIR.exists() or not SERVER_AUDIO_FILES_BASE_DIR.is_dir():
        logger.error(f"指定的服务器音频目录不存在或不是一个目录: {SERVER_AUDIO_FILES_BASE_DIR}")
        return JSONResponse({"audio_files": []})

    try:
        for item in SERVER_AUDIO_FILES_BASE_DIR.iterdir():
            if item.is_file() and item.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS:
                audio_files.append(item.name)
        logger.info(f"找到 {len(audio_files)} 个音频文件在 {SERVER_AUDIO_FILES_BASE_DIR}")
        return JSONResponse({"audio_files": sorted(audio_files)})
    except Exception as e:
        logger.error(f"列出服务器音频文件时出错: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="无法列出服务器上的音频文件")

@app.post("/api/analyze-server-file")
async def analyze_server_file(request: ServerFileRequest, background_tasks: BackgroundTasks):
    """分析服务器上预定义目录中的指定音频文件"""
    logger = logging.getLogger(__name__)
    filename = request.filename

    if not filename:
        logger.warning("请求中未提供文件名")
        raise HTTPException(status_code=400, detail="未提供文件名")

    # 安全性：基本的文件名清理，防止路径遍历
    if "/" in filename or "\\" in filename or ".." in filename:
        logger.error(f"检测到非法文件名 (包含路径字符): {filename}")
        raise HTTPException(status_code=400, detail="文件名包含非法字符")

    # 构建完整的文件路径
    try:
        if not SERVER_AUDIO_FILES_BASE_DIR.is_absolute():
            logger.error(f"服务器音频文件基目录不是绝对路径: {SERVER_AUDIO_FILES_BASE_DIR}")
            raise HTTPException(status_code=500, detail="服务器内部配置错误")
        
        cleaned_filename = Path(filename).name
        if cleaned_filename != filename: 
            logger.warning(f"原始文件名 '{filename}' 可能包含路径，已清理为 '{cleaned_filename}'")
            raise HTTPException(status_code=400, detail="文件名格式不正确，可能包含路径")

        file_path = SERVER_AUDIO_FILES_BASE_DIR.joinpath(cleaned_filename).resolve()

        # 检查解析后的路径是否仍在预期的基目录下
        if not file_path.is_relative_to(SERVER_AUDIO_FILES_BASE_DIR.resolve()):
            logger.error(f"路径遍历尝试被阻止: 请求的文件 '{filename}' 解析为 '{file_path}'")
            raise HTTPException(status_code=400, detail="非法文件访问")

    except Exception as e: 
        logger.error(f"构建文件路径时出错 for filename '{filename}': {e}")
        raise HTTPException(status_code=500, detail="服务器处理文件路径时出错")

    if not file_path.exists() or not file_path.is_file():
        logger.error(f"请求分析的文件不存在或不是文件: {file_path}")
        raise HTTPException(status_code=404, detail=f"文件 '{filename}' 未找到或不是有效文件")
    
    if file_path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
        logger.warning(f"请求分析的文件类型不受支持: {filename} (后缀: {file_path.suffix})")
        raise HTTPException(status_code=400, detail=f"不支持的文件类型: {file_path.suffix}. 支持的类型: {', '.join(SUPPORTED_AUDIO_EXTENSIONS)}")

    # 懒加载检查 - 如果模型未加载则尝试加载
    if not model_manager:
        logger.error("模型管理器未初始化")
        raise HTTPException(status_code=503, detail="模型管理器未初始化")
    
    # 检查并加载SenseVoice模型
    if not model_manager.whisper_pipeline:
        logger.info("触发SenseVoice模型懒加载...")
        print("🎙️  触发SenseVoice模型懒加载...")
        if not model_manager.load_whisper_model():
            logger.error("SenseVoice语音识别模型加载失败")
            raise HTTPException(status_code=503, detail="SenseVoice语音识别模型加载失败")
    
    # 检查并加载LLM模型
    if not model_manager.llm_model:
        logger.info("触发LLM模型懒加载...")
        print("🧠 触发LLM模型懒加载...")
        if not model_manager.load_llm_model():
            logger.error("LLM模型加载失败")
            raise HTTPException(status_code=503, detail="LLM模型加载失败")

    try:
        task_id = f"task_server_{int(datetime.now().timestamp())}_{filename.replace('.', '_')}"
        
        if not hasattr(app.state, 'tasks_status'):
            app.state.tasks_status = {}
        
        app.state.tasks_status[task_id] = {
            "status": "pending", 
            "progress": 0, 
            "message": "任务已提交，等待处理", 
            "original_filename": filename,
            "task_type": "server_file_analysis"
        }

        def progress_update_for_task(current_task_id: str, progress: float, message: str):
            if current_task_id in app.state.tasks_status:
                app.state.tasks_status[current_task_id]["progress"] = progress
                app.state.tasks_status[current_task_id]["message"] = message
                if progress == 100.0 or progress == -1.0: 
                     app.state.tasks_status[current_task_id]["status"] = "completed" if progress == 100.0 else "failed"

        async def analyze_server_audio_task(audio_path: str, original_filename: str, current_task_id: str):
            logger.info(f"后台任务 {current_task_id} 开始分析服务器文件: {audio_path}")
            try:
                if current_task_id in app.state.tasks_status:
                    app.state.tasks_status[current_task_id]["status"] = "processing"
                    app.state.tasks_status[current_task_id]["message"] = "任务正在处理中..."
                
                result = await analyze_audio_file(
                    str(audio_path), 
                    original_filename, 
                    progress_callback=lambda p, m: progress_update_for_task(current_task_id, p, m)
                )
                app.state.tasks_status[current_task_id]["result"] = result
                app.state.tasks_status[current_task_id]["status"] = "completed"
                app.state.tasks_status[current_task_id]["progress"] = 100.0
                logger.info(f"后台任务 {current_task_id} 完成")
            except Exception as e:
                logger.error(f"后台任务 {current_task_id} 失败: {e}")
                app.state.tasks_status[current_task_id]["status"] = "failed"
                app.state.tasks_status[current_task_id]["error"] = str(e)
                app.state.tasks_status[current_task_id]["progress"] = -1.0

        background_tasks.add_task(analyze_server_audio_task, file_path, filename, task_id)
        
        return JSONResponse({
            "message": f"服务器文件 '{filename}' 分析已开始", 
            "task_id": task_id, 
            "status_url": f"/api/task-status/{task_id}"
        })
        
    except Exception as e:
        logger.error(f"分析服务器文件 '{filename}' 时发生意外错误: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"分析文件时发生内部错误: {str(e)}")

@app.get("/api/analysis-history")
async def get_analysis_history(limit: int = 50):
    """获取分析历史记录"""
    try:
        # 获取内存中的结果和文件系统中的历史记录
        all_results = []
        
        # 加载文件系统中的历史记录
        output_dir = DATA_PATHS["output_dir"]
        if output_dir.exists():
            for json_file in sorted(output_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        result_data = json.load(f)
                        all_results.append(result_data)
                except Exception as e:
                    logging.warning(f"读取历史记录文件失败 {json_file}: {e}")
        
        # 添加内存中的结果（避免重复）
        existing_ids = {r.get("analysis_id") for r in all_results}
        for result in analysis_results:
            if result.get("analysis_id") not in existing_ids:
                all_results.append(result)
        
        # 按时间戳排序，取最近的记录
        all_results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        recent_results = all_results[:limit]
        
        return JSONResponse({
            "total_count": len(all_results),
            "results": recent_results
        })
    except Exception as e:
        logging.error(f"获取历史记录失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/statistics")
async def get_statistics():
    """获取统计信息"""
    try:
        if not analysis_results:
            return JSONResponse({
                "total_analyses": 0,
                "risk_distribution": {},
                "average_score": 0
            })
        
        # 计算统计信息 - 适应新的简化格式
        risk_levels = [r["risk_assessment"]["risk_level"] for r in analysis_results if "risk_assessment" in r]
        risk_scores = [r["risk_assessment"]["risk_score"] for r in analysis_results if "risk_assessment" in r]
        
        from collections import Counter
        risk_distribution = dict(Counter(risk_levels))
        average_score = sum(risk_scores) / len(risk_scores) if risk_scores else 0
        
        return JSONResponse({
            "total_analyses": len(analysis_results),
            "risk_distribution": risk_distribution,
            "average_score": round(average_score, 2),
            "latest_analysis": analysis_results[-1]["timestamp"] if analysis_results else None
        })
    except Exception as e:
        logging.error(f"获取统计信息失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze-text")
async def analyze_text_direct(text_input: TextInput):
    """直接分析文本内容，优先使用LLM，失败则回退"""
    try:
        text = text_input.text
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="文本内容不能为空")

        # 获取风险分析器实例
        risk_analyzer = get_risk_analyzer()
        if not risk_analyzer:
            raise HTTPException(status_code=500, detail="风险分析器未初始化")

        # 使用统一的分析方法
        analysis_result = risk_analyzer.analyze(text)
        
        if not analysis_result or "error" in analysis_result:
            # 如果分析失败或返回错误，则使用备用结果或抛出异常
            logging.error(f"文本直接分析失败: {analysis_result.get('error', '未知错误')}")
            # 尝试从备用分析获取信息，如果analyze方法本身处理了回退，这里可能不需要
            # 但为了健壮性，可以保留一个最终的备用方案
            fallback_data = risk_analyzer._fallback_analysis(text) # 调用内部备用方法以获取结构
            risk_level = fallback_data.get("risk_level", "未知")
            risk_score = fallback_data.get("total_score", 0)
            key_issues = fallback_data.get("key_concerns", ["分析失败"])
            processing_method = fallback_data.get("analysis_method", "备用分析")
        else:
            risk_level = analysis_result.get("risk_level", "未知")
            risk_score = analysis_result.get("total_score", 0) # 确保使用 total_score
            # AI分析结果中的风险因素和关注点可能更丰富
            key_issues = analysis_result.get("key_concerns", []) or analysis_result.get("risk_factors", [])
            processing_method = analysis_result.get("analysis_method", "AI智能评估")

        # 简化的结果格式
        result = {
            "analysis_id": f"text_analysis_{int(datetime.now().timestamp())}",
            "timestamp": datetime.now().isoformat(),
            "risk_assessment": {
                "risk_level": risk_level,
                "risk_score": risk_score,
                "key_issues": key_issues[:3]  # 只显示前3个关键问题
            },
            "processing_info": {
                "text_length": len(text),
                "processing_method": processing_method # 使用从分析结果获取的方法
            }
        }
        
        # 保存结果到内存列表
        analysis_results.append(result) 
        
        # 保存结果到文件
        try:
            output_dir = DATA_PATHS["output_dir"]
            output_dir.mkdir(parents=True, exist_ok=True) # 确保目录存在
            output_file = output_dir / f"{result['analysis_id']}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            logging.info(f"文本分析结果已保存到: {output_file}")
        except Exception as e:
            logging.error(f"保存文本分析结果到文件失败: {e}")
            # 即使保存文件失败，也应返回结果给用户
        
        return JSONResponse(result)
        
    except Exception as e:
        logging.error(f"文本分析端点错误: {e}")
        import traceback
        traceback.print_exc() # 打印详细堆栈信息
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")
