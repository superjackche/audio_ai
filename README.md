# AI语音政治风险监测系统

这是一个基于本地AI大模型的语音政治风险监测系统，用于实时分析授课内容中的意识形态问题。

## 功能特性

- 🎤 实时麦克风语音监测
- 📁 音频文件上传分析
- 🤖 本地AI大模型加载（支持GPU加速）
- 🔍 政治风险关键词检测
- 📊 语义行为分析和加权评分
- 🌐 Web界面可视化
- 📈 实时监控和历史数据分析

## 系统架构

```
audio_ai/
├── app/                    # Web应用
├── models/                 # AI模型相关
├── data/                   # 数据和训练集
├── static/                 # 静态文件
├── templates/              # HTML模板
├── utils/                  # 工具函数
└── config/                 # 配置文件
```

## 快速开始

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 启动系统：
```bash
python main.py
```

3. 访问 http://localhost:8000

## 技术栈

- **语音识别**: OpenAI Whisper
- **大语言模型**: ChatGLM3-6B（本地部署）
- **Web框架**: FastAPI
- **前端**: HTML5 + JavaScript
- **GPU加速**: CUDA + PyTorch

## 注意事项

- 确保已安装CUDA环境（GPU加速）
- 首次运行会自动下载模型文件
- 支持中英文语音识别和分析
