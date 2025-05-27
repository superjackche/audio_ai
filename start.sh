#!/bin/bash

# 🚀 AI语音政治风险监控系统 - 生产启动脚本
# =============================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ASCII艺术
echo -e "${PURPLE}"
cat << "EOF"
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║    🚀 AI语音政治风险监控系统 - 生产启动器                    ║
    ║                                                               ║
    ║    ⚡ 自动优化配置 | 🎯 最大化性能 | 🔧 智能调优            ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

# 函数：打印彩色消息
print_status() {
    echo -e "${GREEN}[✅]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[ℹ️ ]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[⚠️ ]${NC} $1"
}

print_error() {
    echo -e "${RED}[❌]${NC} $1"
    exit 1
}

print_progress() {
    echo -e "${CYAN}[🔄]${NC} $1"
}

# 检查Python环境
check_python_env() {
    print_progress "检查Python环境..."
    
    if ! command -v python &> /dev/null; then
        print_error "Python未安装或不在PATH中"
    fi
    
    PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
    print_info "Python版本: $PYTHON_VERSION"
    
    # 检查conda环境
    if [[ "$CONDA_DEFAULT_ENV" == "audio_ai" ]]; then
        print_status "已在audio_ai conda环境中"
    else
        print_warning "未在audio_ai环境中，尝试激活..."
        if command -v conda &> /dev/null; then
            source "$(conda info --base)/etc/profile.d/conda.sh"
            conda activate audio_ai
            print_status "已激活audio_ai环境"
        else
            print_warning "Conda未找到，继续使用当前Python环境"
        fi
    fi
}

# 检查依赖包
check_dependencies() {
    print_progress "检查核心依赖包..."
    
    REQUIRED_PACKAGES=(
        "torch"
        "transformers"
        "fastapi"
        "uvicorn"
        "numpy"
        "psutil"
        "GPUtil"
    )
    
    MISSING_PACKAGES=()
    
    for package in "${REQUIRED_PACKAGES[@]}"; do
        if python -c "import $package" &> /dev/null; then
            print_status "$package ✓"
        else
            print_warning "$package ✗ (缺失)"
            MISSING_PACKAGES+=("$package")
        fi
    done
    
    if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
        print_error "缺少必要依赖包: ${MISSING_PACKAGES[*]}. 请运行: pip install -r requirements.txt"
    fi
}

# 检查GPU环境
check_gpu() {
    print_progress "检查GPU环境..."
    
    if command -v nvidia-smi &> /dev/null; then
        GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
        print_status "检测到 $GPU_COUNT 个GPU设备"
        nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits | while read line; do
            print_info "GPU: $line"
        done
    else
        print_warning "未检测到NVIDIA GPU或nvidia-smi未安装"
    fi
    
    # 检查PyTorch CUDA支持
    if python -c "import torch; print(f'PyTorch CUDA: {torch.cuda.is_available()}')" 2>/dev/null; then
        print_status "PyTorch CUDA支持已启用"
    else
        print_warning "PyTorch CUDA支持未启用"
    fi
}

# 检查模型文件
check_models() {
    print_progress "检查模型文件..."
    
    # 检查Whisper模型缓存
    WHISPER_CACHE="./models/cache/models--openai--whisper-large-v3"
    if [[ -d "$WHISPER_CACHE" ]] && [[ -f "$WHISPER_CACHE/snapshots"*"/pytorch_model.bin" || -f "$WHISPER_CACHE/snapshots"*"/model.safetensors" ]]; then
        print_status "Whisper模型已缓存"
    else
        print_warning "Whisper模型未找到，首次运行时将自动下载"
    fi
    
    # 检查数据目录
    if [[ ! -d "./data" ]]; then
        mkdir -p ./data/audio_files ./data/analysis_results
        print_status "创建数据目录"
    fi
}

# 系统优化
optimize_system() {
    print_progress "应用系统优化..."
    
    # 设置HuggingFace镜像
    export HF_ENDPOINT="https://hf-mirror.com"
    print_status "设置HuggingFace镜像源"
    
    # 设置PyTorch优化
    export OMP_NUM_THREADS=$(nproc)
    export MKL_NUM_THREADS=$(nproc)
    print_status "设置多线程优化"
    
    # GPU内存优化
    export CUDA_VISIBLE_DEVICES="0"
    print_status "设置GPU设备"
    
    print_status "系统优化完成"
}

# 启动服务
start_service() {
    print_progress "启动AI语音政治风险监控系统..."
    
    # 检查端口是否被占用
    PORT=8000
    if lsof -i:$PORT &> /dev/null; then
        print_warning "端口 $PORT 已被占用，尝试终止现有进程..."
        pkill -f "uvicorn.*main_new:app" || true
        sleep 2
    fi
    
    # 启动FastAPI服务
    print_status "在端口 $PORT 启动服务..."
    echo -e "${CYAN}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🌐 服务地址: http://localhost:$PORT"
    echo "📚 API文档: http://localhost:$PORT/docs"
    echo "🔧 健康检查: http://localhost:$PORT/health"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${NC}"
    
    # 启动服务
    cd "$(dirname "$0")"
    
    if [[ -f "app/main_new.py" ]]; then
        uvicorn app.main_new:app --host 0.0.0.0 --port $PORT --reload
    elif [[ -f "main.py" ]]; then
        uvicorn main:app --host 0.0.0.0 --port $PORT --reload
    else
        print_error "未找到主应用文件 (app/main_new.py 或 main.py)"
    fi
}

# 安装依赖
install_dependencies() {
    print_progress "安装Python依赖..."
    
    if [[ -f "requirements.txt" ]]; then
        pip install -r requirements.txt
        print_status "依赖安装完成"
    else
        print_error "requirements.txt文件未找到"
    fi
}

# 主函数
main() {
    # 切换到脚本目录
    cd "$(dirname "$0")"
    
    # 安装依赖（如果指定）
    if [[ "$1" == "--install" ]] || [[ "$1" == "-i" ]]; then
        install_dependencies
    fi
    
    # 环境检查
    check_python_env
    check_dependencies
    check_gpu
    
    # 模型检查
    check_models
    
    # 系统优化
    optimize_system
    
    # 启动服务
    start_service
}

# 显示帮助
show_help() {
    echo -e "${YELLOW}用法:${NC}"
    echo "  ./start.sh              # 启动系统"
    echo "  ./start.sh --install    # 安装依赖并启动"
    echo "  ./start.sh --help       # 显示帮助"
    echo ""
    echo -e "${YELLOW}选项:${NC}"
    echo "  -i, --install     安装依赖包"
    echo "  -h, --help        显示帮助信息"
    echo ""
    echo -e "${YELLOW}系统要求:${NC}"
    echo "  • Python 3.8+"
    echo "  • PyTorch with CUDA"
    echo "  • 至少8GB GPU显存（推荐16GB+）"
    echo "  • 16GB+ 系统内存"
}

# 优雅退出处理
cleanup() {
    echo -e "\n${YELLOW}正在关闭服务...${NC}"
    pkill -f "uvicorn.*main_new:app" || true
    print_status "服务已关闭"
    exit 0
}

trap cleanup SIGINT SIGTERM

# 参数处理
case "${1:-}" in
    -h|--help)
        show_help
        exit 0
        ;;
    *)
        main "$@"
        ;;
esac
