# AI语音政治风险监测系统

这是一个基于本地AI大模型的语音政治风险监测系统，用于实时分析授课内容中的意识形态问题。

## 🚀 功能特性

- 🎤 **实时语音监测** - 麦克风实时录音和分析
- 📁 **文件分析** - 支持多格式音频文件上传
- 📝 **文本分析** - 直接输入文本进行政治风险评估
- 🤖 **本地AI模型** - 支持ChatGLM3、Qwen等大模型
- 🔍 **智能风险检测** - 多维度政治敏感内容识别
- 📊 **量化评分** - 语义分析和加权评分系统
- 🌐 **Web可视化** - 直观的操作界面和结果展示
- 📈 **历史记录** - 分析结果存储和查看

## 🛠 环境要求

- **Python**: 3.8+
- **操作系统**: Windows 10/11, Linux, macOS
- **GPU**: CUDA 11.8+ (可选，用于GPU加速)
- **内存**: 建议8GB+ (模型加载需要)

## 📦 快速安装

### 自动安装
```bash
# 克隆项目
git clone <repository-url>
cd audio_ai

# 直接运行主程序（会自动安装依赖）
python main.py
```

### 手动安装
```bash
# 1. 创建虚拟环境
python -m venv .venv

# 2. 激活虚拟环境
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 启动系统
python main.py
```

## 🎯 使用方法

1. **启动系统**: 运行 `python main.py`
2. **访问界面**: 打开浏览器访问 http://localhost:8000
3. **功能测试**:
   - **实时监测**: 点击"开始录音"测试麦克风功能
   - **文件分析**: 上传音频文件进行分析
   - **文本分析**: 直接输入文本进行政治风险评估
   - **历史记录**: 查看之前的分析结果

## 🏗 系统架构

```
audio_ai/
├── app/                    # Web应用主程序
│   └── main.py            # FastAPI服务器
├── models/                 # AI模型相关
│   ├── model_manager.py   # 模型管理器
│   ├── risk_analyzer.py   # 风险分析器
│   ├── training_data.py   # 训练数据生成
│   └── fine_tuning.py     # 模型微调
├── config/                 # 配置文件
│   └── settings.py        # 系统配置
├── utils/                  # 工具函数
│   └── audio_utils.py     # 音频处理工具
├── static/                 # 静态文件
│   ├── css/              # 样式文件
│   └── js/               # JavaScript文件
├── templates/              # HTML模板
├── data/                   # 数据目录
│   ├── uploads/          # 上传文件
│   ├── outputs/          # 分析结果
│   └── logs/             # 日志文件
└── requirements.txt        # 依赖包列表
```

## 🔧 技术栈

- **语音识别**: OpenAI Whisper
- **大语言模型**: ChatGLM3-6B, Qwen系列（本地部署）
- **Web框架**: FastAPI
- **前端**: HTML5 + JavaScript + CSS3
- **自然语言处理**: jieba分词, transformers
- **GPU加速**: CUDA + PyTorch

## ⚙️ 高级配置

### 模型微调
```bash
python models/fine_tuning.py
```

### 自定义风险关键词
编辑 `config/settings.py` 文件中的 `RISK_KEYWORDS` 配置

### 调整评分权重
修改 `config/settings.py` 文件中的 `SCORING_WEIGHTS` 配置

### 端口配置
如果8000端口被占用，可修改 `config/settings.py` 中的端口号

## 🐛 常见问题

### 问题1: CUDA相关错误
- **解决方案**: 如果没有NVIDIA GPU或CUDA环境，系统会自动切换到CPU模式运行

### 问题2: 模型下载失败
- **原因**: 首次运行需要下载AI模型文件
- **解决方案**: 
  - 使用VPN或代理
  - 手动下载模型文件到 `models/cache/` 目录

### 问题3: 麦克风权限
- **检查项目**:
  - Windows麦克风权限设置
  - 防病毒软件是否阻止麦克风访问

### 问题4: 端口占用
- **解决方案**: 修改 `config/settings.py` 中的端口号

## 📋 开发说明

### 日志查看
系统日志文件位于: `data/logs/audio_ai.log`

### 依赖管理
确保所有依赖包正确安装，检查Python版本兼容性

### 数据存储
- 上传文件: `data/uploads/`
- 分析结果: `data/outputs/`
- 训练数据: `data/train_dataset.json`

## ⚠️ 注意事项

- 确保Python版本为3.8+
- 首次运行会自动下载模型文件（需要网络连接）
- GPU加速需要CUDA环境，但不是必需的
- 支持中英文语音识别和分析
- 请确保音频设备正常工作

## 📄 许可证

本项目仅供学习和研究使用。
