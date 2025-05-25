# AI语音政治风险监测系统 - 快速安装指南

## 安装步骤

### 1. 环境要求
- Python 3.8+
- CUDA 11.8+ (GPU加速，可选)
- Windows 10/11

### 2. 自动安装
直接运行 `start.bat` 文件，脚本会自动完成以下操作：
- 创建Python虚拟环境
- 安装所需依赖包
- 启动系统

### 3. 手动安装
如果自动安装失败，可以手动执行以下步骤：

```bash
# 1. 创建虚拟环境
python -m venv audio_ai

# 2. 激活虚拟环境
audio_ai\Scripts\activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 启动系统
python main.py
```

### 4. 可能遇到的问题

#### 问题1: CUDA相关错误
如果没有NVIDIA GPU或CUDA环境，系统会自动切换到CPU模式运行。

#### 问题2: 模型下载失败
首次运行时需要下载AI模型文件，如果网络连接问题导致下载失败，可以：
- 使用VPN或代理
- 手动下载模型文件到 `models/` 目录

#### 问题3: 麦克风权限
如果录音功能不可用，请检查：
- Windows麦克风权限设置
- 防病毒软件是否阻止麦克风访问

#### 问题4: 端口占用
如果8000端口被占用，可以修改 `config/settings.py` 中的端口号。

### 5. 系统功能测试

启动成功后，访问 http://localhost:8000 进行功能测试：

1. **实时监测**: 点击"开始录音"测试麦克风功能
2. **文件分析**: 上传音频文件进行分析
3. **文本分析**: 直接输入文本进行政治风险评估
4. **历史记录**: 查看之前的分析结果

### 6. 高级配置

#### 模型微调
如需对AI模型进行微调，运行：
```bash
python models/fine_tuning.py
```

#### 自定义风险关键词
编辑 `config/settings.py` 文件中的 `RISK_KEYWORDS` 配置。

#### 调整评分权重
修改 `config/settings.py` 文件中的 `SCORING_WEIGHTS` 配置。

### 7. 技术支持

如遇到技术问题，请检查：
- 系统日志文件: `data/logs/audio_ai.log`
- 确保所有依赖包正确安装
- Python版本兼容性

### 8. 系统架构说明

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
