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

# 导入自定义模块
from models.model_manager import ModelManager
from models.risk_analyzer import PoliticalRiskAnalyzer
from utils.audio_utils import AudioRecorder, AudioProcessor, FileManager, Logger
from config.settings import WEB_CONFIG, DATA_PATHS

# 初始化FastAPI应用
app = FastAPI(
    title="AI语音政治风险监测系统",
    description="基于本地AI大模型的实时语音政治风险分析系统",
    version="1.0.0"
)

# 设置静态文件和模板
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# 全局变量
model_manager: ModelManager = None
risk_analyzer: PoliticalRiskAnalyzer = None
audio_recorder: AudioRecorder = None
analysis_results = []

@app.on_event("startup")
async def startup_event():
    """应用启动时初始化"""
    global model_manager, risk_analyzer, audio_recorder
    
    # 设置日志
    Logger.setup_logging(DATA_PATHS["logs_dir"])
    logger = logging.getLogger(__name__)
    
    logger.info("正在启动AI语音政治风险监测系统...")
    
    try:
        # 初始化组件
        model_manager = ModelManager()
        risk_analyzer = PoliticalRiskAnalyzer()
        audio_recorder = AudioRecorder()
        
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
        "whisper_loaded": model_manager and model_manager.whisper_model is not None,
        "llm_loaded": model_manager and model_manager.llm_model is not None,
        "recording_available": audio_recorder is not None,
        "timestamp": datetime.now().isoformat()
    }
    return JSONResponse(status)

@app.post("/api/upload-audio")
async def upload_audio(file: UploadFile = File(...)):
    """上传音频文件进行分析"""
    if not model_manager or not model_manager.whisper_model:
        raise HTTPException(status_code=503, detail="语音识别模型未加载")
    
    try:
        # 验证文件类型
        if not file.filename.lower().endswith(('.wav', '.mp3', '.m4a', '.flac')):
            raise HTTPException(status_code=400, detail="不支持的音频格式")
        
        # 保存上传的文件
        file_content = await file.read()
        file_path = FileManager.save_uploaded_file(
            file_content, 
            file.filename, 
            DATA_PATHS["upload_dir"]
        )
        
        # 分析音频
        result = await analyze_audio_file(file_path)
        
        # 清理临时文件
        os.remove(file_path)
        
        return JSONResponse(result)
        
    except Exception as e:
        logging.error(f"音频上传分析失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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
            result = await analyze_audio_file(file_path)
            
            # 清理临时文件
            os.remove(file_path)
            
            return JSONResponse(result)
        else:
            raise HTTPException(status_code=500, detail="录音文件保存失败")
    except Exception as e:
        logging.error(f"停止录音失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def analyze_audio_file(file_path: str) -> dict:
    """分析音频文件"""
    try:
        # 1. 语音转文本
        transcription = model_manager.transcribe_audio(file_path)
        text = transcription["text"]
        
        if not text.strip():
            return {
                "error": "未检测到语音内容",
                "transcription": transcription
            }
        
        # 2. 政治风险分析
        risk_analysis = risk_analyzer.analyze_text(text)
        
        # 3. AI模型深度分析（如果可用）
        llm_analysis = {}
        if model_manager.llm_model:
            try:
                llm_analysis = model_manager.analyze_text_risk(text)
            except Exception as e:
                logging.warning(f"AI模型分析失败: {e}")
                llm_analysis = {"error": "AI模型分析不可用"}
        
        # 4. 提取音频特征
        audio_features = AudioProcessor.extract_audio_features(file_path)
        
        # 5. 综合结果
        result = {
            "analysis_id": f"analysis_{int(datetime.now().timestamp())}",
            "timestamp": datetime.now().isoformat(),
            "transcription": transcription,
            "risk_analysis": risk_analysis,
            "llm_analysis": llm_analysis,
            "audio_features": audio_features,
            "summary": {
                "text_length": len(text),
                "risk_level": risk_analysis["risk_level"],
                "risk_score": risk_analysis["total_score"],
                "audio_duration": audio_features.get("duration", 0)
            }
        }
        
        # 保存分析结果
        analysis_results.append(result)
        
        # 保存到文件
        output_file = DATA_PATHS["output_dir"] / f"{result['analysis_id']}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        return result
        
    except Exception as e:
        logging.error(f"音频分析失败: {e}")
        raise

@app.get("/api/analysis-history")
async def get_analysis_history(limit: int = 50):
    """获取分析历史记录"""
    try:
        # 返回最近的分析记录
        recent_results = analysis_results[-limit:] if analysis_results else []
        
        return JSONResponse({
            "total_count": len(analysis_results),
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
        
        # 计算统计信息
        risk_levels = [r["risk_analysis"]["risk_level"] for r in analysis_results]
        risk_scores = [r["risk_analysis"]["total_score"] for r in analysis_results]
        
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
        
        # 政治风险分析
        risk_analysis = risk_analyzer.analyze_text(text)
        
        # AI模型分析（如果可用）
        llm_analysis = {}
        if model_manager and model_manager.llm_model:
            try:
                llm_analysis = model_manager.analyze_text_risk(text)
            except Exception as e:
                logging.warning(f"AI模型分析失败: {e}")
                llm_analysis = {"error": "AI模型分析不可用"}
        
        result = {
            "analysis_id": f"text_analysis_{int(datetime.now().timestamp())}",
            "timestamp": datetime.now().isoformat(),
            "input_text": text,
            "risk_analysis": risk_analysis,
            "llm_analysis": llm_analysis,
            "summary": {
                "text_length": len(text),
                "risk_level": risk_analysis["risk_level"],
                "risk_score": risk_analysis["total_score"]
            }
        }
        
        # 保存结果
        analysis_results.append(result)
        
        return JSONResponse(result)
        
    except Exception as e:
        logging.error(f"文本分析失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=WEB_CONFIG["host"],
        port=WEB_CONFIG["port"],
        reload=WEB_CONFIG["reload"]
    )
