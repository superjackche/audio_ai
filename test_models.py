"""
测试模型下载和加载
"""
import os
import sys
sys.path.append('.')

from config.settings import MODEL_CONFIG
from models.model_manager import ModelManager
from models.risk_analyzer import PoliticalRiskAnalyzer
import logging

def test_model_download():
    """测试模型下载和初始化"""
    
    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 50)
    logger.info("开始测试模型下载和加载")
    logger.info("=" * 50)
    
    # 显示配置信息
    logger.info(f"模型配置: {MODEL_CONFIG}")
    logger.info(f"HF镜像源: {os.environ.get('HF_ENDPOINT', 'default')}")
    logger.info(f"缓存目录: {os.environ.get('HF_HOME', 'default')}")
    
    # 初始化风险分析器
    logger.info("初始化政治风险分析器...")
    risk_analyzer = PoliticalRiskAnalyzer()
    
    # 测试基础分析功能
    test_text = "今天我们讨论中西方教育制度的差异，需要客观分析各自的优势和不足。"
    basic_analysis = risk_analyzer.analyze_text(test_text)
    logger.info(f"基础分析测试通过，风险等级: {basic_analysis['risk_level']}")
    
    # 初始化模型管理器
    logger.info("初始化AI模型管理器...")
    model_manager = ModelManager()
    
    # 测试Whisper模型加载
    logger.info("测试Whisper语音识别模型加载...")
    whisper_success = model_manager.load_whisper_model()
    if whisper_success:
        logger.info("✅ Whisper模型加载成功")
    else:
        logger.error("❌ Whisper模型加载失败")
    
    # 测试大语言模型加载
    logger.info("测试大语言模型加载...")
    logger.info("⚠️  注意：首次下载模型可能需要较长时间，请耐心等待...")
    llm_success = model_manager.load_llm_model()
    if llm_success:
        logger.info("✅ 大语言模型加载成功")
        
        # 测试大模型分析
        try:
            logger.info("测试大模型分析功能...")
            llm_analysis = model_manager.analyze_text_risk(test_text)
            logger.info(f"✅ 大模型分析成功: {llm_analysis.get('risk_level', '未知')}")
        except Exception as e:
            logger.error(f"❌ 大模型分析测试失败: {e}")
    else:
        logger.warning("⚠️  大语言模型加载失败，系统将以基础模式运行")
    
    # 总结
    logger.info("=" * 50)
    if whisper_success:
        logger.info("🎉 系统可以正常运行！")
        logger.info("✅ 语音识别功能可用")
        if llm_success:
            logger.info("✅ AI深度分析功能可用")
        else:
            logger.info("⚠️  仅基础分析功能可用")
    else:
        logger.error("❌ 系统无法正常运行，请检查环境配置")
    
    logger.info("=" * 50)
    
    return whisper_success, llm_success

if __name__ == "__main__":
    test_model_download()
