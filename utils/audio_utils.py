import os
import logging
import wave
import pyaudio
import threading
import time
from pathlib import Path
from typing import Optional, Callable, Any
import tempfile

class AudioRecorder:
    """音频录制器"""
    
    def __init__(self, sample_rate: int = 16000, channels: int = 1, chunk_size: int = 1024):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.format = pyaudio.paInt16
        
        self.audio = pyaudio.PyAudio()
        self.stream: Optional[Any] = None
        self.frames = []
        self.is_recording = False
        self.recording_thread: Optional[threading.Thread] = None
        
        self.logger = logging.getLogger(__name__)
    
    def start_recording(self) -> bool:
        """开始录音"""
        try:
            self.frames = []
            self.is_recording = True
            
            self.stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            self.recording_thread = threading.Thread(target=self._record_audio)
            self.recording_thread.start()
            
            self.logger.info("开始录音")
            return True
        except Exception as e:
            self.logger.error(f"开始录音失败: {e}")
            return False
    
    def stop_recording(self) -> Optional[str]:
        """停止录音并保存文件"""
        try:
            self.is_recording = False
            
            if self.recording_thread:
                self.recording_thread.join()
            
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            
            # 保存录音文件
            if self.frames:
                timestamp = int(time.time())
                filename = f"recording_{timestamp}.wav"
                filepath = Path(tempfile.gettempdir()) / filename
                
                with wave.open(str(filepath), 'wb') as wf:
                    wf.setnchannels(self.channels)
                    wf.setsampwidth(self.audio.get_sample_size(self.format))
                    wf.setframerate(self.sample_rate)
                    wf.writeframes(b''.join(self.frames))
                
                self.logger.info(f"录音保存到: {filepath}")
                return str(filepath)
            
            return None
        except Exception as e:
            self.logger.error(f"停止录音失败: {e}")
            return None
    
    def _record_audio(self):
        """录音线程函数"""
        while self.is_recording:
            try:
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                self.frames.append(data)
            except Exception as e:
                self.logger.error(f"录音错误: {e}")
                break
    
    def cleanup(self):
        """清理资源"""
        if self.stream:
            self.stream.close()
        self.audio.terminate()

class AudioProcessor:
    """音频处理工具"""
    
    @staticmethod
    def validate_audio_file(file_path: str) -> bool:
        """验证音频文件"""
        try:
            import librosa
            # 尝试加载音频文件
            audio, sr = librosa.load(file_path, sr=None)
            return len(audio) > 0
        except Exception:
            return False
    
    @staticmethod
    def convert_audio_format(input_path: str, output_path: str, target_sr: int = 16000) -> bool:
        """转换音频格式"""
        try:
            import librosa
            import soundfile as sf
            
            # 加载音频
            audio, sr = librosa.load(input_path, sr=target_sr)
            
            # 保存为WAV格式
            sf.write(output_path, audio, target_sr)
            return True
        except Exception as e:
            logging.error(f"音频格式转换失败: {e}")
            return False
    
    @staticmethod
    def get_audio_duration(file_path: str) -> Optional[float]:
        """获取音频时长（秒）"""
        try:
            import librosa
            audio, sr = librosa.load(file_path, sr=None)
            return len(audio) / sr
        except Exception:
            return None
    
    @staticmethod
    def extract_audio_features(file_path: str) -> dict:
        """提取音频特征"""
        try:
            import librosa
            import numpy as np
            
            audio, sr = librosa.load(file_path, sr=16000)
            
            # 提取基础特征
            features = {
                "duration": len(audio) / sr,
                "sample_rate": sr,
                "rms_energy": float(np.sqrt(np.mean(audio**2))),
                "zero_crossing_rate": float(np.mean(librosa.feature.zero_crossing_rate(audio))),
                "spectral_centroid": float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)))
            }
            
            return features
        except Exception as e:
            logging.error(f"音频特征提取失败: {e}")
            return {}

class FileManager:
    """文件管理工具"""
    
    @staticmethod
    def save_uploaded_file(file_content: bytes, filename: str, upload_dir: Path) -> str:
        """保存上传的文件"""
        try:
            # 确保上传目录存在
            upload_dir.mkdir(parents=True, exist_ok=True)
            
            # 生成唯一文件名
            timestamp = int(time.time())
            file_extension = Path(filename).suffix
            unique_filename = f"{timestamp}_{filename}"
            file_path = upload_dir / unique_filename
            
            # 保存文件
            with open(file_path, 'wb') as f:
                f.write(file_content)
            
            return str(file_path)
        except Exception as e:
            logging.error(f"文件保存失败: {e}")
            raise
    
    @staticmethod
    def cleanup_old_files(directory: Path, max_age_hours: int = 24):
        """清理旧文件"""
        try:
            current_time = time.time()
            for file_path in directory.glob("*"):
                if file_path.is_file():
                    file_age = current_time - file_path.stat().st_mtime
                    if file_age > max_age_hours * 3600:
                        file_path.unlink()
                        logging.info(f"删除旧文件: {file_path}")
        except Exception as e:
            logging.error(f"清理文件失败: {e}")
    
    @staticmethod
    def get_file_size(file_path: str) -> int:
        """获取文件大小（字节）"""
        try:
            return os.path.getsize(file_path)
        except Exception:
            return 0

class Logger:
    """日志工具"""
    
    @staticmethod
    def setup_logging(log_dir: Path, log_level: str = "INFO"):
        """设置日志"""
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建日志格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # 文件处理器
        file_handler = logging.FileHandler(
            log_dir / "audio_ai.log",
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        # 配置根日志器
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level.upper()))
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        return root_logger
