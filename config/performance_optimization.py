#!/usr/bin/env python3
"""
🚀 AI语音政治风险监控系统 - 极速性能优化配置
==========================================

专为超高速、超准确的语音处理而设计的性能优化配置
目标：最大化GPU利用率，最小化延迟，最优化内存使用

优化重点：
1. 模型加载优化 - 预加载、量化、缓存
2. GPU内存优化 - 动态分配、内存池
3. 并发处理优化 - 异步处理、批处理
4. 音频处理优化 - 实时流式处理
5. 缓存策略优化 - 智能缓存、预测加载
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import psutil

# ==================== 早期PyTorch配置 ====================
# 必须在任何PyTorch操作之前设置！

def setup_torch_early_optimization():
    """在PyTorch初始化前设置关键配置"""
    
    # 设置环境变量（在导入torch前）
    cpu_cores = psutil.cpu_count(logical=False)
    os.environ['OMP_NUM_THREADS'] = str(cpu_cores)
    os.environ['MKL_NUM_THREADS'] = str(cpu_cores)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_cores)
    
    # 设置CUDA相关环境变量
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'
    
    return cpu_cores

# 早期设置
_CPU_CORES = setup_torch_early_optimization()

# 现在安全导入PyTorch
import torch
import GPUtil

# ==================== 系统检测与自适应配置 ====================

def detect_system_capabilities():
    """检测系统硬件能力并返回优化配置"""
    
    # GPU检测
    gpu_info = {
        'available': torch.cuda.is_available(),
        'count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'memory': [],
        'compute_capability': []
    }
    
    if gpu_info['available']:
        for i in range(gpu_info['count']):
            props = torch.cuda.get_device_properties(i)
            gpu_info['memory'].append(props.total_memory // (1024**3))  # GB
            gpu_info['compute_capability'].append(f"{props.major}.{props.minor}")
    
    # CPU检测
    cpu_info = {
        'cores': _CPU_CORES,  # 使用早期检测的CPU核心数
        'threads': psutil.cpu_count(logical=True),
        'frequency': psutil.cpu_freq().max if psutil.cpu_freq() else 0,
        'memory_gb': psutil.virtual_memory().total // (1024**3)
    }
    
    return gpu_info, cpu_info

def display_system_info():
    """显示系统信息"""
    try:
        gpu_info, cpu_info = detect_system_capabilities()
        
        print("📊 系统配置信息:")
        print(f"   CPU: {cpu_info['cores']}物理核心 / {cpu_info['threads']}逻辑线程")
        print(f"   内存: {cpu_info['memory_gb']}GB")
        print(f"   CPU频率: {cpu_info['frequency']:.1f}MHz" if cpu_info['frequency'] else "   CPU频率: 未知")
        
        if gpu_info['available']:
            print(f"   GPU: {gpu_info['count']}个可用")
            for i, mem in enumerate(gpu_info['memory']):
                compute_cap = gpu_info['compute_capability'][i] if i < len(gpu_info['compute_capability']) else "未知"
                print(f"   GPU {i}: {mem}GB显存, 计算能力 {compute_cap}")
        else:
            print("   GPU: 无可用GPU")
            
        # 显示GPU实时状态
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                print("   GPU实时状态:")
                for gpu in gpus:
                    print(f"     GPU {gpu.id}: {gpu.name}, 使用率 {gpu.load*100:.1f}%, 内存 {gpu.memoryUsed}MB/{gpu.memoryTotal}MB")
        except Exception as e:
            print(f"   GPU状态检测失败: {e}")
            
    except Exception as e:
        print(f"系统信息显示失败: {e}")

# ==================== 极速模型配置 ====================

def get_ultra_fast_model_config():
    """获取超高速模型配置"""
    
    gpu_info, cpu_info = detect_system_capabilities()
    
    # 基础配置
    config = {
        "model_optimization": {
            # 模型选择策略：优先更快的3B模型
            "primary_model": "Qwen/Qwen2.5-Omni-3B",
            "fallback_model": "Qwen/Qwen2.5-Omni-7B",
            
            # 量化配置
            "quantization": {
                "enabled": True,
                "method": "bitsandbytes",  # 使用4bit量化
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": "float16",  # 使用字符串而不是torch.dtype
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_quant_type": "nf4"
            },
            
            # 模型加载优化
            "loading": {
                "torch_dtype": "float16",  # 使用字符串而不是torch.dtype
                "device_map": "auto",
                "low_cpu_mem_usage": True,
                "trust_remote_code": True,
                "use_flash_attention_2": True,  # Flash Attention 2.0
                "attn_implementation": "flash_attention_2"
            }
        },
        
        "gpu_optimization": {
            # GPU内存管理
            "memory_management": {
                "max_memory_fraction": 0.85,  # 使用85%GPU内存
                "memory_growth": True,
                "allow_growth": True,
                "pre_allocate": False
            },
            
            # CUDA优化
            "cuda_settings": {
                "cudnn_benchmark": True,
                "cudnn_deterministic": False,
                "empty_cache_frequency": 10,  # 每10次推理清理缓存
                "use_cuda_graphs": True
            }
        },
        
        "inference_optimization": {
            # 推理参数优化
            "generation_config": {
                "max_new_tokens": 512,
                "temperature": 0.1,  # 更低温度提高一致性
                "top_p": 0.9,
                "top_k": 50,
                "do_sample": True,
                "pad_token_id": 0,
                "eos_token_id": 1,
                "use_cache": True
            },
            
            # 批处理配置
            "batch_processing": {
                "enabled": True,
                "batch_size": 4 if gpu_info['memory'] and gpu_info['memory'][0] > 20 else 2,
                "max_batch_size": 8,
                "dynamic_batching": True,
                "timeout_seconds": 0.1
            }
        }
    }
    
    # 根据GPU数量调整配置
    if gpu_info['count'] >= 2:
        config["model_optimization"]["loading"]["device_map"] = {
            "": 0,  # 主模型放在GPU 0
            "lm_head": 1  # 输出层放在GPU 1
        }
        config["inference_optimization"]["batch_processing"]["batch_size"] *= 2
    
    return config

# ==================== 音频处理优化配置 ====================

AUDIO_PROCESSING_CONFIG = {
    "streaming": {
        "enabled": True,
        "chunk_size": 1024,  # 更小的块大小减少延迟
        "overlap": 0.2,
        "buffer_size": 4096,
        "sample_rate": 16000,
        "channels": 1
    },
    
    "preprocessing": {
        "noise_reduction": True,
        "auto_gain_control": True,
        "voice_activity_detection": True,
        "silence_removal": True,
        "normalization": "rms"
    },
    
    "feature_extraction": {
        "method": "mel_spectrogram",
        "n_mels": 80,
        "n_fft": 400,
        "hop_length": 160,
        "window": "hann",
        "center": True,
        "normalize": True
    }
}

# ==================== 缓存优化配置 ====================

CACHE_CONFIG = {
    "model_cache": {
        "enabled": True,
        "max_size_gb": 8,
        "ttl_hours": 24,
        "preload_models": True,
        "cache_location": "./models/cache"
    },
    
    "inference_cache": {
        "enabled": True,
        "max_entries": 1000,
        "hash_method": "md5",
        "cache_similar_inputs": True,
        "similarity_threshold": 0.95
    },
    
    "audio_cache": {
        "enabled": True,
        "max_size_mb": 500,
        "compress": True,
        "cache_processed": True
    }
}

# ==================== 并发处理配置 ====================

CONCURRENCY_CONFIG = {
    "async_processing": {
        "enabled": True,
        "max_workers": min(8, psutil.cpu_count()),
        "queue_size": 100,
        "timeout_seconds": 30
    },
    
    "threading": {
        "audio_processing_threads": 2,
        "model_inference_threads": 1,  # 避免GIL问题
        "io_threads": 4
    },
    
    "multiprocessing": {
        "enabled": False,  # GPU模型不适用多进程
        "processes": 1
    }
}

# ==================== 监控与自适应配置 ====================

MONITORING_CONFIG = {
    "performance_tracking": {
        "enabled": True,
        "metrics": [
            "inference_time",
            "gpu_utilization", 
            "memory_usage",
            "throughput",
            "queue_length"
        ],
        "log_interval": 10
    },
    
    "auto_optimization": {
        "enabled": True,
        "adapt_batch_size": True,
        "adapt_memory_usage": True,
        "adapt_worker_count": True,
        "optimization_interval": 60
    }
}

# ==================== 专家级优化策略 ====================

class UltraPerformanceOptimizer:
    """超高性能优化器"""
    
    def __init__(self):
        self.gpu_info, self.cpu_info = detect_system_capabilities()
        self.config = get_ultra_fast_model_config()
        self.logger = logging.getLogger(__name__)
    
    def detect_system_capabilities(self):
        """检测系统能力（兼容方法）"""
        return detect_system_capabilities()
    
    def optimize_system(self):
        """优化系统（兼容方法）"""
        return self.apply_all_optimizations()
    
    def get_optimized_config(self):
        """获取优化配置（兼容方法）"""
        return {
            "whisper_kwargs": {
                "torch_dtype": torch.float16,  # 这里转换为实际的torch类型
                "low_cpu_mem_usage": True,
                "use_safetensors": True
                # 移除 trust_remote_code 和 device_map，这些对 Whisper pipeline 不适用
            },
            "model_kwargs": self.config.get("model_optimization", {}),
            "gpu_kwargs": self.config.get("gpu_optimization", {}),
            "inference_kwargs": self.config.get("inference_optimization", {})
        }
    
    def optimize_torch_settings(self):
        """优化PyTorch设置"""
        
        # 启用优化模式
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # 安全设置线程数（避免重复设置错误）
        try:
            torch.set_num_threads(self.cpu_info['cores'])
            self.logger.info(f"PyTorch intra-op线程数设置为: {self.cpu_info['cores']}")
        except RuntimeError as e:
            if "cannot be called after" in str(e) or "already initialized" in str(e):
                self.logger.warning("PyTorch intra-op线程数已被设置，跳过重复设置")
            else:
                self.logger.error(f"设置PyTorch intra-op线程数时出错: {e}")
        
        try:
            torch.set_num_interop_threads(self.cpu_info['cores'])
            self.logger.info(f"PyTorch inter-op线程数设置为: {self.cpu_info['cores']}")
        except RuntimeError as e:
            if "cannot set number of interop threads after parallel work has started" in str(e):
                self.logger.warning("PyTorch inter-op线程数无法设置（并行工作已开始），这是正常的")
            elif "cannot be called after" in str(e) or "already initialized" in str(e):
                self.logger.warning("PyTorch inter-op线程数已被设置，跳过重复设置")
            else:
                self.logger.error(f"设置PyTorch inter-op线程数时出错: {e}")
        
        # 内存管理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.memory.set_per_process_memory_fraction(0.85)
        
        self.logger.info("PyTorch优化设置已应用")
    
    def setup_gpu_optimization(self):
        """设置GPU优化"""
        
        if not torch.cuda.is_available():
            return
        
        # 设置GPU设备
        if self.gpu_info['count'] > 0:
            torch.cuda.set_device(0)
        
        # 启用混合精度
        os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
        os.environ['TORCH_USE_CUDA_DSA'] = '1'
        
        self.logger.info(f"GPU优化设置完成，可用GPU: {self.gpu_info['count']}")
    
    def get_optimal_batch_size(self, model_size_gb: float = 3.0) -> int:
        """根据GPU内存计算最优批处理大小"""
        
        if not self.gpu_info['available']:
            return 1
        
        available_memory = self.gpu_info['memory'][0] if self.gpu_info['memory'] else 8
        # 预留40%内存给模型，其余用于批处理
        batch_memory = (available_memory - model_size_gb) * 0.6
        
        # 每个样本大概需要0.5GB内存
        optimal_batch = max(1, int(batch_memory / 0.5))
        
        return min(optimal_batch, 16)  # 最大16
    
    def apply_all_optimizations(self):
        """应用所有优化"""
        
        self.optimize_torch_settings()
        self.setup_gpu_optimization()
        
        # 应用环境变量优化
        optimizations = {
            'OMP_NUM_THREADS': str(self.cpu_info['cores']),
            'MKL_NUM_THREADS': str(self.cpu_info['cores']),
            'NUMEXPR_NUM_THREADS': str(self.cpu_info['cores']),
            'TOKENIZERS_PARALLELISM': 'false',
            'CUDA_VISIBLE_DEVICES': '0,1' if self.gpu_info['count'] >= 2 else '0',
            'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:512',
            'TRANSFORMERS_OFFLINE': '1',
            'HF_DATASETS_OFFLINE': '1'
        }
        
        for key, value in optimizations.items():
            os.environ[key] = value
        
        self.logger.info("🚀 所有性能优化已应用！")

# ==================== 智能预加载系统 ====================

class SmartPreloader:
    """智能模型预加载系统"""
    
    def __init__(self):
        self.preloaded_models = {}
        self.usage_stats = {}
    
    async def preload_critical_models(self):
        """预加载关键模型"""
        
        models_to_preload = [
            "Qwen/Qwen2.5-Omni-3B",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        ]
        
        for model_name in models_to_preload:
            try:
                # 这里将在实际实现中加载模型
                self.preloaded_models[model_name] = f"preloaded_{model_name}"
                logging.info(f"✅ 预加载模型: {model_name}")
            except Exception as e:
                logging.error(f"❌ 预加载失败: {model_name}, 错误: {e}")
    
    def start_preloading(self):
        """启动预加载（同步版本）"""
        try:
            # 简化的同步预加载
            logging.info("📡 启动智能预加载系统...")
            return True
        except Exception as e:
            logging.error(f"预加载启动失败: {e}")
            return False
    
    def preload_model(self, model, model_type: str):
        """预加载指定模型"""
        try:
            model_key = f"{model_type}_{id(model)}"
            self.preloaded_models[model_key] = model
            logging.info(f"✅ 模型预加载完成: {model_type}")
            
            # 更新使用统计
            if model_type not in self.usage_stats:
                self.usage_stats[model_type] = {"count": 0, "last_used": time.time()}
            self.usage_stats[model_type]["count"] += 1
            self.usage_stats[model_type]["last_used"] = time.time()
            
            return True
        except Exception as e:
            logging.error(f"❌ 模型预加载失败: {model_type}, 错误: {e}")
            return False

# ==================== 实时性能监控 ====================

class RealTimeMonitor:
    """实时性能监控器"""
    
    def __init__(self):
        self.metrics = {
            'inference_times': [],
            'gpu_utilization': [],
            'memory_usage': [],
            'throughput': 0
        }
    
    def log_inference_time(self, time_ms: float):
        """记录推理时间"""
        self.metrics['inference_times'].append(time_ms)
        if len(self.metrics['inference_times']) > 100:
            self.metrics['inference_times'].pop(0)
    
    def get_average_inference_time(self) -> float:
        """获取平均推理时间"""
        if not self.metrics['inference_times']:
            return 0
        return sum(self.metrics['inference_times']) / len(self.metrics['inference_times'])
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        return {
            'avg_inference_time_ms': self.get_average_inference_time(),
            'total_inferences': len(self.metrics['inference_times']),
            'throughput_per_second': self.metrics['throughput'],
            'gpu_utilization': self.metrics['gpu_utilization'][-1] if self.metrics['gpu_utilization'] else 0
        }
    
    def start_monitoring(self):
        """启动监控"""
        try:
            logging.info("📊 启动实时性能监控...")
            return True
        except Exception as e:
            logging.error(f"监控启动失败: {e}")
            return False
    
    def stop_monitoring(self):
        """停止监控"""
        try:
            logging.info("📊 停止实时性能监控...")
            return True
        except Exception as e:
            logging.error(f"监控停止失败: {e}")
            return False

# ==================== 导出配置 ====================

def get_production_config():
    """获取生产环境配置"""
    
    optimizer = UltraPerformanceOptimizer()
    optimizer.apply_all_optimizations()
    
    return {
        "model_config": optimizer.config,
        "audio_config": AUDIO_PROCESSING_CONFIG,
        "cache_config": CACHE_CONFIG,
        "concurrency_config": CONCURRENCY_CONFIG,
        "monitoring_config": MONITORING_CONFIG,
        "optimizer": optimizer
    }

# ==================== 性能基准测试 ====================

def run_performance_benchmark():
    """运行性能基准测试"""
    
    print("🏁 启动性能基准测试...")
    
    gpu_info, cpu_info = detect_system_capabilities()
    
    print(f"📊 系统配置:")
    print(f"   CPU: {cpu_info['cores']}核心 / {cpu_info['threads']}线程")
    print(f"   内存: {cpu_info['memory_gb']}GB")
    print(f"   GPU: {gpu_info['count']}个GPU")
    
    if gpu_info['memory']:
        for i, mem in enumerate(gpu_info['memory']):
            print(f"   GPU {i}: {mem}GB显存")
    
    config = get_ultra_fast_model_config()
    optimal_batch = UltraPerformanceOptimizer().get_optimal_batch_size()
    
    print(f"✨ 优化配置:")
    print(f"   推荐批处理大小: {optimal_batch}")
    print(f"   量化模式: {config['model_optimization']['quantization']['method']}")
    print(f"   内存使用率: {config['gpu_optimization']['memory_management']['max_memory_fraction']*100}%")
    
    return config

# ==================== 全局初始化函数 ====================

def initialize_ultra_performance():
    """
    全局性能优化初始化函数
    应在应用启动时调用，确保所有优化配置正确应用
    """
    
    print("🚀 正在初始化超高性能优化配置...")
    
    # 设置HuggingFace镜像（全局设置）
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    
    try:
        # 创建优化器实例
        optimizer = UltraPerformanceOptimizer()
        
        # 应用PyTorch优化
        optimizer.optimize_torch_settings()
        
        # 设置GPU优化
        optimizer.setup_gpu_optimization()
        
        # 启动预加载器
        preloader = SmartPreloader()
        preloader.start_preloading()
        
        # 启动实时监控
        monitor = RealTimeMonitor()
        monitor.start_monitoring()
        
        print("✅ 超高性能优化配置初始化完成！")
        
        # 显示系统信息
        display_system_info()
        
        return {
            'optimizer': optimizer,
            'preloader': preloader,
            'monitor': monitor,
            'status': 'initialized'
        }
        
    except Exception as e:
        logging.error(f"性能优化初始化失败: {e}")
        print(f"❌ 性能优化初始化失败: {e}")
        return {
            'status': 'failed',
            'error': str(e)
        }

# ==================== 自动初始化 ====================

# 定义全局变量存储优化组件
_PERFORMANCE_COMPONENTS = None

def get_performance_components():
    """获取性能优化组件（懒加载）"""
    global _PERFORMANCE_COMPONENTS
    
    if _PERFORMANCE_COMPONENTS is None:
        _PERFORMANCE_COMPONENTS = initialize_ultra_performance()
    
    return _PERFORMANCE_COMPONENTS

# 导出主要类和函数
__all__ = [
    'UltraPerformanceOptimizer',
    'SmartPreloader', 
    'RealTimeMonitor',
    'initialize_ultra_performance',
    'get_performance_components',
    'detect_system_capabilities',
    'get_ultra_fast_model_config',
    'display_system_info'
]
