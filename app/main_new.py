from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import JSONResponse, HTMLResponse
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

# 导入自定义模块
from models.fast_model_manager import FastModelManager
from models.risk_analyzer import PoliticalRiskAnalyzer
from utils.audio_utils import AudioRecorder, AudioProcessor, FileManager, Logger
from config.settings import WEB_CONFIG, DATA_PATHS

# Define the secure base directory for server-side audio files
SERVER_AUDIO_FILES_BASE_DIR = Path("/new_disk/cwh/audio_ai/server_audio_files/")
SUPPORTED_AUDIO_EXTENSIONS = ['.wav', '.mp3', '.m4a', '.flac']

# 初始化FastAPI应用
app = FastAPI(
    title="AI语音政治风险监测系统",
    description="基于FastModelManager(Whisper-large-v3 + Qwen2.5-7B-Instruct)的快速语音政治风险分析系统",
    version="2.0.0"
)

# 获取 app/main.py 文件所在的目录
APP_MAIN_PY_DIR = Path(__file__).resolve().parent

# 更新静态文件和模板目录的路径
app.mount("/static", StaticFiles(directory=APP_MAIN_PY_DIR.parent / "static"), name="static")
templates = Jinja2Templates(directory=APP_MAIN_PY_DIR.parent / "templates")

class ServerFileRequest(BaseModel):
    filename: str

# 全局变量
model_manager: FastModelManager = None
risk_analyzer: PoliticalRiskAnalyzer = None
audio_recorder: AudioRecorder = None
analysis_results = []

@app.on_event("startup")
async def startup_event():
    """应用启动时初始化"""
    global model_manager, risk_analyzer, audio_recorder
    
    logger = logging.getLogger(__name__)
    logger.info("正在启动AI语音政治风险监测系统...")
    
    try:
        # 初始化组件
        model_manager = FastModelManager()
        risk_analyzer = PoliticalRiskAnalyzer()
        
        # 尝试初始化音频录制器（可选）
        try:
            audio_recorder = AudioRecorder()
            logger.info("音频录制功能已启用")
        except ImportError as e:
            logger.warning(f"音频录制功能不可用: {e}")
            audio_recorder = None
        
        # 在后台加载AI模型
        asyncio.create_task(load_models_background())
        
        logger.info("系统启动成功")
    except Exception as e:
        logger.error(f"系统启动失败: {e}")
        raise

async def load_models_background():
    """后台加载AI模型"""
    logger = logging.getLogger(__name__)
    try:
        logger.info("开始后台加载AI模型...")
        success = model_manager.initialize_models()
        if success:
            logger.info("AI模型加载完成")
        else:
            logger.error("AI模型加载失败")
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

async def analyze_audio_file(processing_file_path: str, original_filename: str, progress_callback: Optional[callable] = None) -> dict:
    """使用FastModelManager (Whisper-large-v3 + Qwen2.5-7B-Instruct) 分析音频文件"""
    logger = logging.getLogger(__name__)
    try:
        # 使用快速模型组合处理音频
        if progress_callback:
            progress_callback(0.0, "开始使用Whisper + Qwen2.5快速模型分析音频...")

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
                "processing_method": "whisper_large_v3_qwen2.5_7b",
                "transcription_time": fast_result.get("transcription_time", 0),
                "analysis_time": fast_result.get("analysis_time", 0),
                "total_time": fast_result.get("total_processing_time", 0)
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

    if not model_manager or not model_manager.whisper_pipeline:
        logger.error("Whisper模型未加载，无法分析服务器文件")
        raise HTTPException(status_code=503, detail="Whisper模型未加载")
    
    if not model_manager.llm_model:
        logger.error("LLM模型未加载，无法分析服务器文件")
        raise HTTPException(status_code=503, detail="LLM模型未加载")

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
                app.state.tasks_status[current_task_id]["message"] = "分析完成"
                logger.info(f"后台任务 {current_task_id} 完成分析: {audio_path}")
            except Exception as e:
                logger.error(f"后台任务 {current_task_id} 分析服务器文件 {audio_path} 失败: {e}")
                app.state.tasks_status[current_task_id]["status"] = "failed"
                app.state.tasks_status[current_task_id]["error"] = str(e)
                app.state.tasks_status[current_task_id]["progress"] = -1.0
                app.state.tasks_status[current_task_id]["message"] = f"分析失败: {str(e)[:100]}..."
        
        background_tasks.add_task(analyze_server_audio_task, str(file_path), filename, task_id)
        
        logger.info(f"已为文件 '{filename}' 创建后台分析任务: {task_id}")
        return JSONResponse({
            "message": f"文件 '{filename}' 的分析任务已提交", 
            "task_id": task_id, 
            "status_url": f"/api/task-status/{task_id}"
        })
        
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"分析服务器文件 '{filename}' 时发生意外错误: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"分析文件时发生内部错误: {str(e)}")

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
