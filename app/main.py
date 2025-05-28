from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form, Request # Added Request for app.state
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import JSONResponse, HTMLResponse
from contextlib import asynccontextmanager
import uvicorn
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
import logging
from typing import Optional
import tempfile
from pydantic import BaseModel
import uuid # Add this import

# 导入自定义模块
from models.simple_model_manager import SimpleModelManager
from models.risk_analyzer import PoliticalRiskAnalyzer
from utils.audio_utils import AudioRecorder, AudioProcessor, FileManager, Logger
from config.settings import WEB_CONFIG, DATA_PATHS

# Define the secure base directory for server-side audio files
SERVER_AUDIO_FILES_BASE_DIR = Path("/new_disk/cwh/audio_ai/server_audio_files/")
SUPPORTED_AUDIO_EXTENSIONS = ['.wav', '.mp3', '.m4a', '.flac']

# 全局变量
model_manager: SimpleModelManager = None
risk_analyzer: PoliticalRiskAnalyzer = None
audio_recorder: AudioRecorder = None
analysis_results = [] # This is an in-memory list, primarily for very short-term caching if any.

ANALYSIS_TASKS = {} # Initialize a global dictionary to store task statuses and results

async def save_analysis_to_history(result_data: dict):
    """将分析结果保存到JSON文件作为历史记录"""
    logger = logging.getLogger(__name__)
    try:
        output_dir = DATA_PATHS["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        
        analysis_id = result_data.get("analysis_id")
        if not analysis_id:
            logger.error("无法保存历史记录: analysis_id 在 result_data 中缺失。")
            return

        file_path = output_dir / f"{analysis_id}.json"
        
        # 使用 asyncio.to_thread 执行阻塞的文件写入操作
        def _write_file():
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, ensure_ascii=False, indent=4)
            logger.info(f"分析结果已保存到历史记录: {file_path}")

        await asyncio.to_thread(_write_file)
        
        # 可选：如果 analysis_results 仍被积极用于某些即时UI更新，则可以添加到内存列表
        # global analysis_results # Ensure we are modifying the global list
        # analysis_results.insert(0, result_data) # Add to the beginning
        # analysis_results = analysis_results[:100] # Keep it bounded if it grows too large

    except Exception as e:
        logger.error(f"保存分析结果到历史记录失败: {e}", exc_info=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # global model_manager, risk_analyzer, audio_recorder # Globals are already defined at module level
    logger = logging.getLogger(__name__)
    logger.info("应用启动 - 初始化组件...")
    
    try:
        # 初始化模型管理器（懒加载，不立即加载模型）
        app.state.model_manager = SimpleModelManager()
        logger.info("模型管理器(SimpleModelManager)初始化成功")
        
        # 初始化风险分析器
        app.state.risk_analyzer = PoliticalRiskAnalyzer()
        logger.info("风险分析器初始化成功")
        
        # 尝试初始化音频录制器（可选）
        try:
            app.state.audio_recorder = AudioRecorder()
            logger.info("音频录制功能已启用")
        except Exception as e:
            logger.warning(f"音频录制功能不可用: {e}")
            app.state.audio_recorder = None
            
        logger.info("系统组件初始化完成，模型将在首次请求时加载")
        
    except Exception as e:
        logger.error(f"应用启动时初始化组件失败: {e}", exc_info=True)
        # 确保即使初始化失败，这些变量也被设置为None
        app.state.model_manager = None
        app.state.risk_analyzer = None
        app.state.audio_recorder = None

    yield # 服务运行

    logger.info("应用关闭 - 清理资源...")
    # 在这里添加任何必要的清理代码
    if hasattr(app.state, 'model_manager') and app.state.model_manager:
        # 如果模型管理器有清理方法，可以在这里调用
        # app.state.model_manager.cleanup() 
        pass
    logger.info("资源清理完成")

# 将 lifespan 函数与 FastAPI 应用关联
app = FastAPI(
    lifespan=lifespan,
    title="AI语音政治风险监测系统",
    description="基于SimpleModelManager(Whisper-large-v3 + Qwen2.5-7B-Instruct)的快速语音政治风险分析系统",
    version="2.1.0"
)

# 获取 app/main.py 文件所在的目录
APP_MAIN_PY_DIR = Path(__file__).resolve().parent

# 更新静态文件和模板目录的路径
app.mount("/static", StaticFiles(directory=APP_MAIN_PY_DIR.parent / "static"), name="static")
templates = Jinja2Templates(directory=APP_MAIN_PY_DIR.parent / "templates")

async def load_models_background():
    """后台加载AI模型"""
    logger = logging.getLogger(__name__)
    try:
        logger.info("开始后台加载AI模型...")
        # 确保 model_manager 已在 app.state 中设置
        if hasattr(app.state, 'model_manager') and app.state.model_manager:
            # FastModelManager.initialize_models() 是同步的
            success = app.state.model_manager.initialize_models()
            if success:
                logger.info("AI模型加载完成")
            else:
                logger.error("AI模型加载失败")
        else:
            logger.error("模型管理器在 app.state 中未找到，无法在后台加载模型。")
            
    except Exception as e:
        logger.error(f"后台加载模型出错: {e}")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """主页"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/status")
async def get_system_status():
    """获取系统状态"""
    status = {
        "system_online": True,
        "whisper_model_loaded": model_manager and model_manager.whisper_pipeline is not None,
        "llm_model_loaded": model_manager and model_manager.llm_model is not None,
        "fast_processing_ready": model_manager and model_manager.whisper_pipeline is not None and model_manager.llm_model is not None,
        "recording_available": audio_recorder is not None,
        "processing_method": "whisper_large_v3_qwen2.5_7b",
        "timestamp": datetime.now().isoformat()
    }
    return JSONResponse(status)

async def analyze_audio_file(audio_path: Path, original_filename: str):
    """
    分析单个音频文件（本地上传或服务器文件），提取文本并进行政治风险评估。
    此函数现在也负责保存到历史记录。
    """
    logger = logging.getLogger(__name__)
    # request_app = app # 'app' is global, can use app.state directly
    if not hasattr(app.state, 'model_manager') or app.state.model_manager is None or \
       not app.state.model_manager.whisper_pipeline or not app.state.model_manager.llm_model: # More specific check
        logger.error("模型管理器或其内部模型未完全加载。无法处理请求。")
        raise HTTPException(status_code=503, detail="模型服务不可用或未完全初始化，请稍后再试")

    try:
        logger.info(f"开始分析音频文件: {original_filename} (路径: {audio_path})")
        
        # 使用 app.state.model_manager 的 process_audio_complete 方法
        # process_audio_complete 是同步的，所以使用 asyncio.to_thread
        analysis_output = await asyncio.to_thread(
            app.state.model_manager.process_audio_complete, str(audio_path)
        )

        # 从 analysis_output 中提取所需信息
        text = analysis_output.get("text", "")
        risk_assessment_data = analysis_output.get("risk_analysis", {})
        risk_level = risk_assessment_data.get("risk_level", "未知")
        risk_score = risk_assessment_data.get("risk_score", 0)
        key_issues = risk_assessment_data.get("key_issues", [])
        
        # 构建 processing_details，可以从 analysis_output 中获取更多信息
        processing_details = {
            "transcription_time": analysis_output.get("transcription_time"),
            "analysis_time": analysis_output.get("analysis_time"),
            "total_processing_time": analysis_output.get("total_processing_time"),
            "processing_method": analysis_output.get("processing_method")
        }

        result_data = {
            "analysis_id": f"analysis_{int(datetime.now().timestamp())}_{uuid.uuid4().hex[:8]}",
            "original_filename": original_filename,
            "timestamp": datetime.now().isoformat(),
            "text": text,
            "risk_assessment": {
                "risk_level": risk_level,
                "risk_score": risk_score,
                "key_issues": key_issues[:3] # 确保只取前三个关键问题，与UI一致
            },
            "processing_info": processing_details
        }
        await save_analysis_to_history(result_data) # 调用新定义的保存函数
        return result_data

    except Exception as e:
        logger.error(f"分析文件 {original_filename} 时出错: {e}", exc_info=True)
        # Re-raise as HTTPException or a custom exception to be handled by process_audio_task
        raise # Or return a specific error structure if preferred

async def process_audio_task(task_id: str, audio_path: Path, original_filename: str):
    """后台处理音频文件的任务"""
    logger = logging.getLogger(__name__)
    try:
        logger.info(f"任务 {task_id}: 开始处理 {original_filename}")
        ANALYSIS_TASKS[task_id]["status"] = "PROCESSING"
        ANALYSIS_TASKS[task_id]["timestamp_processing_start"] = datetime.now().isoformat()

        # Call the main analysis function
        # This function (analyze_audio_file) is defined in main.py and can access app.state.model_manager
        analysis_result_obj = await analyze_audio_file(audio_path, original_filename)

        ANALYSIS_TASKS[task_id]["status"] = "COMPLETED"
        ANALYSIS_TASKS[task_id]["result"] = analysis_result_obj
        ANALYSIS_TASKS[task_id]["timestamp_completed"] = datetime.now().isoformat()
        logger.info(f"任务 {task_id}: 处理完成 {original_filename}")

    except Exception as e:
        logger.error(f"任务 {task_id}: 处理失败 {original_filename} - {e}", exc_info=True)
        ANALYSIS_TASKS[task_id]["status"] = "FAILED"
        ANALYSIS_TASKS[task_id]["error"] = str(e)
        ANALYSIS_TASKS[task_id]["timestamp_failed"] = datetime.now().isoformat()

@app.post("/api/upload-audio")
async def upload_audio(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    """上传音频文件进行分析"""
    if not model_manager or not model_manager.whisper_pipeline:
        raise HTTPException(status_code=503, detail="Whisper模型未加载")
    
    if not model_manager.llm_model:
        raise HTTPException(status_code=503, detail="LLM模型未加载")
    
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
    """获取指定任务ID的当前状态和结果（如果可用）"""
    logger = logging.getLogger(__name__)
    # logger.debug(f"查询任务状态: {task_id}. 当前所有任务: {list(ANALYSIS_TASKS.keys())}") # For debugging
    task_info = ANALYSIS_TASKS.get(task_id)
    if not task_info:
        logger.warning(f"任务ID未找到: {task_id}")
        raise HTTPException(status_code=404, detail="Task ID not found")
    
    return task_info

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
async def analyze_server_file(background_tasks: BackgroundTasks, filename: str = Form(...)):
    """分析服务器上预定义目录中的指定音频文件"""
    logger = logging.getLogger(__name__)

    if not filename:
        logger.warning("API /api/analyze-server-file: 文件名为空")
        raise HTTPException(status_code=400, detail="文件名不能为空")

    # 使用在文件顶部定义的全局常量 SERVER_AUDIO_FILES_BASE_DIR
    server_audio_files_dir = SERVER_AUDIO_FILES_BASE_DIR
    
    # 检查配置的服务器音频目录是否存在且是一个目录
    if not server_audio_files_dir.exists() or not server_audio_files_dir.is_dir():
        logger.error(f"配置的服务器音频目录不存在或不是一个目录: {server_audio_files_dir}")
        raise HTTPException(status_code=500, detail="服务器配置错误：无法访问音频文件存储目录。")
        
    server_file_path = server_audio_files_dir / filename
    if not server_file_path.exists():
        logger.error(f"请求分析服务器文件，但文件未找到: {server_file_path}")
        # 提供更清晰的错误消息，指明在哪个目录中未找到文件
        raise HTTPException(status_code=404, detail=f"文件 '{filename}' 未在服务器目录 '{server_audio_files_dir.name}' 中找到。")

    task_id = f"task_{uuid.uuid4()}"
    ANALYSIS_TASKS[task_id] = {
        "status": "QUEUED",
        "filename": filename,
        "original_filename": filename, 
        "result": None,
        "error": None,
        "timestamp_queued": datetime.now().isoformat()
    }
    
    background_tasks.add_task(process_audio_task, task_id, server_file_path, filename)
    
    logger.info(f"分析任务 {task_id} 已加入队列: {filename}")
    return {"task_id": task_id, "status": "QUEUED", "message": f"文件 {filename} 的分析任务已启动"}

@app.post("/api/start-recording")
async def start_recording():
    """开始录音"""
    if not audio_recorder:
        raise HTTPException(status_code=503, detail="录音功能不可用")
    
    try:
        success = audio_recorder.start_recording()
        if success:
            return JSONResponse({"status": "recording_started", "timestamp": datetime.now().isoformat()})
        else:
            raise HTTPException(status_code=500, detail="录音启动失败")
    except Exception as e:
        logging.error(f"开始录音失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/stop-recording")
async def stop_recording():
    """停止录音并分析"""
    if not audio_recorder:
        raise HTTPException(status_code=503, detail="录音功能不可用")
    
    try:
        file_path = audio_recorder.stop_recording()
        if file_path:
            # 分析录音
            result = await analyze_audio_file(file_path, os.path.basename(file_path))
            # 清理临时文件
            os.remove(file_path)
            return JSONResponse(result)
        else:
            raise HTTPException(status_code=500, detail="录音文件保存失败")
    except Exception as e:
        logging.error(f"停止录音失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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
async def analyze_text_direct(request: Request):
    """直接分析文本"""
    try:
        data = await request.json()
        text = data.get("text", "").strip()
        
        if not text:
            raise HTTPException(status_code=400, detail="文本内容不能为空")
        
        # 使用FastModelManager的LLM分析
        llm_analysis = {}
        if model_manager and model_manager.llm_model:
            try:
                llm_analysis = model_manager.analyze_text_risk(text)
            except Exception as e:
                logging.warning(f"LLM模型分析失败: {e}")
                llm_analysis = {"error": "LLM模型分析不可用"}
        
        # 基于规则的分析作为备用
        rule_analysis = {}
        try:
            rule_analysis = risk_analyzer.analyze_text(text)
        except Exception as e:
            logging.warning(f"规则分析失败: {e}")
            rule_analysis = {"error": "规则分析不可用"}
        
        # 使用LLM分析结果，如果不可用则使用规则分析
        if llm_analysis and "error" not in llm_analysis:
            risk_level = llm_analysis.get("risk_level", "未知")
            risk_score = llm_analysis.get("risk_score", 0)
            key_issues = llm_analysis.get("key_issues", [])
        else:
            risk_level = rule_analysis.get("risk_level", "未知")
            risk_score = rule_analysis.get("total_score", 0)
            key_issues = rule_analysis.get("keywords", [])
        
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
                "processing_method": "text_direct_analysis"
            }
        }
        
        # 保存结果
        analysis_results.append(result)
        
        return JSONResponse(result)
        
    except Exception as e:
        logging.error(f"文本分析失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
