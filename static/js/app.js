// AI语音政治风险监测系统 - 前端JavaScript

class AudioMonitorApp {
    constructor() {
        this.isRecording = false;
        this.analysisHistory = [];
        this.currentModal = null;
        
        this.init();
    }

    init() {
        this.bindEvents();
        this.checkSystemStatus();
        this.loadStatistics();
        this.setupFileUpload();
        
        // 定期检查系统状态
        setInterval(() => this.checkSystemStatus(), 30000);
    }

    bindEvents() {
        // 录音控制
        document.getElementById('start-recording').addEventListener('click', () => this.startRecording());
        document.getElementById('stop-recording').addEventListener('click', () => this.stopRecording());
        
        // 文件上传
        document.getElementById('audio-file').addEventListener('change', (e) => this.handleFileSelect(e));
        
        // 文本分析
        document.getElementById('analyze-text').addEventListener('click', () => this.analyzeText());
        
        // 历史记录刷新
        document.getElementById('refresh-history').addEventListener('click', () => this.loadHistory());
        
        // 拖拽上传
        this.setupDragAndDrop();
        
        // 标签页切换事件
        document.querySelectorAll('[data-bs-toggle="tab"]').forEach(tab => {
            tab.addEventListener('shown.bs.tab', (e) => {
                if (e.target.id === 'history-tab') {
                    this.loadHistory();
                }
            });
        });
    }

    async checkSystemStatus() {
        try {
            const response = await fetch('/api/status');
            const status = await response.json();
            
            this.updateStatusIndicators(status);
            this.updateSystemStatusText(status);
        } catch (error) {
            console.error('检查系统状态失败:', error);
            this.showError('无法连接到服务器');
        }
    }

    updateStatusIndicators(status) {
        // 更新Whisper状态
        const whisperStatus = document.getElementById('whisper-status');
        this.updateStatusElement(whisperStatus, status.whisper_loaded, '语音识别', '已就绪', '未加载');
        
        // 更新LLM状态
        const llmStatus = document.getElementById('llm-status');
        this.updateStatusElement(llmStatus, status.llm_loaded, 'AI大模型', '已就绪', '未加载');
        
        // 更新录音状态
        const recordingStatus = document.getElementById('recording-status');
        this.updateStatusElement(recordingStatus, status.recording_available, '录音功能', '可用', '不可用');
        
        // 更新录音按钮状态
        document.getElementById('start-recording').disabled = !status.recording_available || this.isRecording;
    }

    updateStatusElement(element, isOnline, feature, onlineText, offlineText) {
        element.className = `status-indicator ${isOnline ? 'status-online' : 'status-offline'}`;
        const small = element.querySelector('small');
        small.textContent = isOnline ? onlineText : offlineText;
        small.className = isOnline ? 'text-success' : 'text-danger';
    }

    updateSystemStatusText(status) {
        const statusText = document.getElementById('system-status');
        if (status.whisper_loaded && status.llm_loaded) {
            statusText.innerHTML = '<i class="fas fa-circle text-success"></i> 系统运行正常';
        } else if (status.whisper_loaded || status.llm_loaded) {
            statusText.innerHTML = '<i class="fas fa-circle text-warning"></i> 部分功能可用';
        } else {
            statusText.innerHTML = '<i class="fas fa-circle text-danger"></i> 系统初始化中';
        }
    }    async startRecording() {
        try {
            const startButton = document.getElementById('start-recording');
            const stopButton = document.getElementById('stop-recording');
            
            // 禁用开始按钮，防止重复点击
            startButton.disabled = true;
            startButton.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>启动中...';
            
            const response = await fetch('/api/start-recording', { method: 'POST' });
            const result = await response.json();
            
            if (response.ok) {
                this.isRecording = true;
                startButton.style.display = 'none';
                stopButton.disabled = false;
                stopButton.style.display = 'inline-block';
                document.getElementById('recording-indicator').style.display = 'flex';
                
                this.showSuccess('录音已开始');
                
                // 清空之前的结果
                document.getElementById('realtime-result').innerHTML = '';
            } else {
                this.showError(result.detail || '录音启动失败');
                // 恢复按钮状态
                startButton.disabled = false;
                startButton.innerHTML = '<i class="fas fa-microphone me-2"></i>开始录音';
            }
        } catch (error) {
            console.error('开始录音失败:', error);
            this.showError('录音启动失败，请检查麦克风权限');
            
            // 恢复按钮状态
            const startButton = document.getElementById('start-recording');
            startButton.disabled = false;
            startButton.innerHTML = '<i class="fas fa-microphone me-2"></i>开始录音';
        }
    }    async stopRecording() {
        try {
            const startButton = document.getElementById('start-recording');
            const stopButton = document.getElementById('stop-recording');
            
            // 禁用停止按钮，防止重复点击
            stopButton.disabled = true;
            stopButton.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>处理中...';
            
            // 隐藏录音指示器，显示分析进度条
            document.getElementById('recording-indicator').style.display = 'none';
            this.showProgressBar('recording-analysis-progress', '正在处理录音数据...');
            
            // 模拟录音分析进度
            const progressStages = [
                { progress: 25, message: '正在保存录音文件...' },
                { progress: 50, message: '正在进行语音识别...' },
                { progress: 75, message: '正在分析文本内容...' },
                { progress: 90, message: '正在生成分析报告...' }
            ];

            let stageIndex = 0;
            const progressInterval = setInterval(() => {
                if (stageIndex < progressStages.length) {
                    const stage = progressStages[stageIndex];
                    this.updateProgressBar('recording-analysis-progress', stage.progress, stage.message);
                    stageIndex++;
                }
            }, 800);
            
            const response = await fetch('/api/stop-recording', { method: 'POST' });
            const result = await response.json();
            
            // 清除进度模拟
            clearInterval(progressInterval);
            this.updateProgressBar('recording-analysis-progress', 100, '分析完成！');
            
            this.isRecording = false;
            startButton.style.display = 'inline-block';
            startButton.disabled = false;
            stopButton.style.display = 'none';
            stopButton.innerHTML = '<i class="fas fa-stop me-2"></i>停止录音';
            
            if (response.ok) {
                // 延迟一下让用户看到100%完成
                setTimeout(() => {
                    this.hideProgressBar('recording-analysis-progress');
                    this.displayAnalysisResult(result, 'realtime-result');
                    this.showSuccess('录音分析完成');
                    this.loadStatistics();
                }, 500);
            } else {
                this.hideProgressBar('recording-analysis-progress');
                this.showError(result.detail || '录音处理失败');
                document.getElementById('realtime-result').innerHTML = '';
            }
        } catch (error) {
            console.error('停止录音失败:', error);
            this.hideProgressBar('recording-analysis-progress');
            this.showError('录音处理失败，请重试');
            document.getElementById('realtime-result').innerHTML = '';
            
            // 重置状态
            this.isRecording = false;
            const startButton = document.getElementById('start-recording');
            const stopButton = document.getElementById('stop-recording');
            startButton.style.display = 'inline-block';
            startButton.disabled = false;
            stopButton.style.display = 'none';
            stopButton.innerHTML = '<i class="fas fa-stop me-2"></i>停止录音';
        }
    }

    setupFileUpload() {
        const fileInput = document.getElementById('audio-file');
        const uploadArea = document.getElementById('upload-area');
        
        fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
        
        uploadArea.addEventListener('click', () => {
            fileInput.click();
        });
    }

    setupDragAndDrop() {
        const uploadArea = document.getElementById('upload-area');
        
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, this.preventDefaults, false);
        });
        
        ['dragenter', 'dragover'].forEach(eventName => {
            uploadArea.addEventListener(eventName, () => uploadArea.classList.add('dragover'), false);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, () => uploadArea.classList.remove('dragover'), false);
        });
        
        uploadArea.addEventListener('drop', (e) => this.handleDrop(e), false);
    }

    preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    handleDrop(e) {
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.processFile(files[0]);
        }
    }

    handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            this.processFile(file);
        }
    }    async processFile(file) {
        // 验证文件类型
        const allowedTypes = ['audio/wav', 'audio/mpeg', 'audio/mp4', 'audio/flac'];
        if (!allowedTypes.includes(file.type) && !file.name.match(/\.(wav|mp3|m4a|flac)$/i)) {
            this.showError('不支持的文件格式，请上传 WAV、MP3、M4A 或 FLAC 文件');
            return;
        }
        
        // 检查文件大小 (50MB限制)
        if (file.size > 50 * 1024 * 1024) {
            this.showError('文件太大，请上传小于50MB的文件');
            return;
        }
        
        const uploadArea = document.getElementById('upload-area');
        const fileInput = document.getElementById('audio-file');
          try {
            // 禁用上传区域，防止重复上传
            this.setUploadAreaState(false);
            
            // 显示上传进度条
            this.showProgressBar('upload-progress', '正在上传音频文件...');
            this.updateProgressBar('upload-progress', 10, '正在上传音频文件...');
            this.showLoadingInElement('upload-result', '正在上传音频文件...');
            
            const formData = new FormData();
            formData.append('file', file);
            
            // 模拟上传进度
            const uploadProgressInterval = setInterval(() => {
                const currentProgress = this.getCurrentProgress('upload-progress');
                if (currentProgress < 30) {
                    this.updateProgressBar('upload-progress', currentProgress + 5, '正在上传音频文件...');
                }
            }, 200);
            
            const response = await fetch('/api/upload-audio', {
                method: 'POST',
                body: formData
            });
            
            clearInterval(uploadProgressInterval);
            const result = await response.json();
            
            if (response.ok && result.task_id) {
                // 上传成功，开始分析
                this.updateProgressBar('upload-progress', 35, '文件上传成功，开始分析...');
                this.showLoadingInElement('upload-result', '文件上传成功，正在分析中...');
                await this.pollTaskStatus(result.task_id, 'upload-result');
            } else {
                this.hideProgressBar('upload-progress');
                this.showError(result.detail || '文件上传失败');
                document.getElementById('upload-result').innerHTML = '';
            }
        } catch (error) {
            console.error('文件上传失败:', error);
            this.hideProgressBar('upload-progress');
            this.showError('网络错误，文件上传失败，请检查连接后重试');
            document.getElementById('upload-result').innerHTML = '';
        } finally {
            // 确保重新启用上传区域
            this.setUploadAreaState(true);
        }
    }

    setUploadAreaState(enabled) {
        const uploadArea = document.getElementById('upload-area');
        const fileInput = document.getElementById('audio-file');
        
        if (enabled) {
            uploadArea.style.pointerEvents = 'auto';
            uploadArea.style.opacity = '1';
            uploadArea.classList.remove('uploading');
            fileInput.disabled = false;
            fileInput.value = ''; // 清空文件选择
        } else {
            uploadArea.style.pointerEvents = 'none';
            uploadArea.style.opacity = '0.6';
            uploadArea.classList.add('uploading');
            fileInput.disabled = true;
        }
    }    async pollTaskStatus(taskId, resultContainerId) {
        const maxAttempts = 60; // 最多轮询60次（5分钟）
        let attempts = 0;
        
        const poll = async () => {
            try {
                const response = await fetch(`/api/task-status/${taskId}`);
                const status = await response.json();
                
                if (response.ok) {
                    // 更新进度信息
                    if (status.progress >= 0) {
                        const progressPercent = Math.round(status.progress);
                        
                        // 更新上传进度条
                        this.updateProgressBar('upload-progress', progressPercent, status.message || '处理中...');
                        
                        // 也更新结果容器中的加载信息
                        this.showLoadingInElement(resultContainerId, 
                            `${status.message} (${progressPercent}%)`);
                    }
                    
                    if (status.status === 'completed' && status.result) {
                        // 任务完成，显示结果
                        this.updateProgressBar('upload-progress', 100, '分析完成！');
                        
                        setTimeout(() => {
                            this.hideProgressBar('upload-progress');
                            this.displayAnalysisResult(status.result, resultContainerId);
                            this.showSuccess('文件分析完成');
                            this.loadStatistics();
                        }, 500);
                        return;
                    } else if (status.status === 'failed') {
                        // 任务失败
                        this.hideProgressBar('upload-progress');
                        this.showError(status.error || '分析失败');
                        document.getElementById(resultContainerId).innerHTML = '';
                        return;
                    } else if (status.status === 'processing') {
                        // 继续轮询
                        attempts++;
                        if (attempts < maxAttempts) {
                            setTimeout(poll, 2000); // 2秒后再次检查
                        } else {
                            this.hideProgressBar('upload-progress');
                            this.showError('分析超时，请重试');
                            document.getElementById(resultContainerId).innerHTML = '';
                        }
                    }
                } else {
                    this.hideProgressBar('upload-progress');
                    this.showError('获取任务状态失败');
                    document.getElementById(resultContainerId).innerHTML = '';
                }
            } catch (error) {
                console.error('轮询任务状态失败:', error);
                this.hideProgressBar('upload-progress');
                this.showError('获取分析状态失败');
                document.getElementById(resultContainerId).innerHTML = '';
            }
        };
        
        // 开始轮询
        poll();
    }async analyzeText() {
        const textInput = document.getElementById('text-input');
        const analyzeButton = document.getElementById('analyze-text');
        const text = textInput.value.trim();
        
        if (!text) {
            this.showError('请输入要分析的文本内容');
            textInput.focus();
            return;
        }
        
        // 立即提供视觉反馈
        this.setAnalyzeTextState(false);
        this.showProgressBar('text-analysis-progress', '准备开始分析...');
        
        // 启动模拟进度条
        const progressInterval = this.simulateTextAnalysisProgress();
        
        try {
            const response = await fetch('/api/analyze-text', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text: text })
            });
            
            // 清除模拟进度条
            clearInterval(progressInterval);
            this.updateProgressBar('text-analysis-progress', 100, '分析完成！');
            
            const result = await response.json();
            
            if (response.ok) {
                // 延迟一下让用户看到100%完成
                setTimeout(() => {
                    this.hideProgressBar('text-analysis-progress');
                    this.displayAnalysisResult(result, 'text-result');
                    this.showSuccess('文本分析完成');
                    this.loadStatistics();
                }, 500);
            } else {
                this.hideProgressBar('text-analysis-progress');
                this.showError(result.detail || '文本分析失败');
                document.getElementById('text-result').innerHTML = '';
            }
        } catch (error) {
            console.error('文本分析失败:', error);
            clearInterval(progressInterval);
            this.hideProgressBar('text-analysis-progress');
            this.showError('网络错误，文本分析失败，请检查连接后重试');
            document.getElementById('text-result').innerHTML = '';
        } finally {
            // 恢复按钮和输入框状态
            this.setAnalyzeTextState(true);
        }
    }

    setAnalyzeTextState(enabled) {
        const textInput = document.getElementById('text-input');
        const analyzeButton = document.getElementById('analyze-text');
        
        if (enabled) {
            analyzeButton.disabled = false;
            analyzeButton.innerHTML = '<i class="fas fa-search me-2"></i>分析文本';
            textInput.disabled = false;
            textInput.classList.remove('analyzing');
        } else {
            analyzeButton.disabled = true;
            analyzeButton.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>分析中...';
            textInput.disabled = true;
            textInput.classList.add('analyzing');
        }
    }

    showLoadingInElement(elementId, message) {
        const element = document.getElementById(elementId);
        element.innerHTML = `
            <div class="d-flex align-items-center justify-content-center p-4">
                <div class="spinner-border text-primary me-3" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <div class="text-muted">${message}</div>
            </div>
        `;
    }

    showUploadProgress(show) {
        const progressElement = document.getElementById('upload-progress');
        if (progressElement) {
            progressElement.style.display = show ? 'block' : 'none';
        }
    }

    showSuccess(message) {
        this.showToast(message, 'success');
    }

    showError(message) {
        this.showToast(message, 'error');
        console.error('Error:', message);
    }

    showToast(message, type = 'info') {
        // 创建toast容器（如果不存在）
        let toastContainer = document.getElementById('toast-container');
        if (!toastContainer) {
            toastContainer = document.createElement('div');
            toastContainer.id = 'toast-container';
            toastContainer.className = 'toast-container position-fixed top-0 end-0 p-3';
            toastContainer.style.zIndex = '1050';
            document.body.appendChild(toastContainer);
        }

        // 创建toast
        const toastId = 'toast_' + Date.now();
        const bgClass = type === 'success' ? 'bg-success' : type === 'error' ? 'bg-danger' : 'bg-info';
        const iconClass = type === 'success' ? 'fa-check-circle' : type === 'error' ? 'fa-exclamation-circle' : 'fa-info-circle';

        const toastHtml = `
            <div id="${toastId}" class="toast ${bgClass} text-white" role="alert" aria-live="assertive" aria-atomic="true">
                <div class="toast-header ${bgClass} text-white border-0">
                    <i class="fas ${iconClass} me-2"></i>
                    <strong class="me-auto">${type === 'success' ? '成功' : type === 'error' ? '错误' : '信息'}</strong>
                    <button type="button" class="btn-close btn-close-white" data-bs-dismiss="toast"></button>
                </div>
                <div class="toast-body">
                    ${message}
                </div>
            </div>
        `;

        toastContainer.insertAdjacentHTML('beforeend', toastHtml);

        // 显示toast
        const toastElement = document.getElementById(toastId);
        const toast = new bootstrap.Toast(toastElement, {
            autohide: true,
            delay: type === 'error' ? 5000 : 3000
        });
        toast.show();

        // 清理：在toast隐藏后移除DOM元素
        toastElement.addEventListener('hidden.bs.toast', () => {
            toastElement.remove();
        });
    }

    displayAnalysisResult(result, containerId) {
        const container = document.getElementById(containerId);
        
        if (result.error) {
            container.innerHTML = `
                <div class="alert alert-warning">
                    <i class="fas fa-exclamation-triangle me-2"></i>
                    ${result.error}
                </div>
            `;
            return;
        }
        
        const riskAnalysis = result.risk_analysis;
        const summary = result.summary;
        
        const html = `
            <div class="analysis-card card">
                <div class="card-header d-flex justify-content-between align-items-center">
                    <h6 class="mb-0">
                        <i class="fas fa-search me-2"></i>分析结果
                    </h6>
                    <button class="btn btn-outline-primary btn-sm" onclick="app.showDetailedResult('${result.analysis_id}')">
                        <i class="fas fa-eye me-1"></i>详细信息
                    </button>
                </div>
                <div class="card-body">
                    <div class="row">
                        <div class="col-md-4">
                            <div class="score-display">
                                <div class="score-circle ${this.getScoreClass(summary.risk_score)}">
                                    ${summary.risk_score}
                                </div>
                                <span class="risk-badge ${summary.risk_level}">${summary.risk_level}</span>
                            </div>
                        </div>
                        <div class="col-md-8">
                            <div class="analysis-details">
                                <p><strong>分析时间:</strong> ${new Date(result.timestamp).toLocaleString('zh-CN')}</p>
                                ${summary.text_length ? `<p><strong>文本长度:</strong> ${summary.text_length} 字符</p>` : ''}
                                ${summary.audio_duration ? `<p><strong>音频时长:</strong> ${summary.audio_duration.toFixed(2)} 秒</p>` : ''}
                                <p><strong>关键词匹配:</strong> ${riskAnalysis.keyword_analysis.total_keyword_matches} 个</p>
                                <p><strong>改进建议:</strong></p>
                                <ul class="mb-0">
                                    ${riskAnalysis.recommendations.slice(0, 3).map(rec => `<li>${rec}</li>`).join('')}
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        container.innerHTML = html;
        
        // 保存结果到历史记录
        this.analysisHistory.unshift(result);
    }    showDetailedResult(analysisId) {
        // 首先从历史记录中查找
        let result = this.analysisHistory.find(r => r.analysis_id === analysisId);
        
        // 如果在内存中没找到，尝试从服务器获取
        if (!result) {
            this.fetchDetailedResult(analysisId);
            return;
        }
        
        const modalBody = document.getElementById('modal-body-content');
        modalBody.innerHTML = this.generateDetailedResultHTML(result);
        
        const modal = new bootstrap.Modal(document.getElementById('result-modal'));
        modal.show();
    }

    async fetchDetailedResult(analysisId) {
        try {
            // 从服务器获取历史记录
            const response = await fetch('/api/analysis-history');
            const data = await response.json();
            
            const result = data.results.find(r => r.analysis_id === analysisId);
            if (result) {
                const modalBody = document.getElementById('modal-body-content');
                modalBody.innerHTML = this.generateDetailedResultHTML(result);
                
                const modal = new bootstrap.Modal(document.getElementById('result-modal'));
                modal.show();
            } else {
                this.showError('未找到对应的分析记录');
            }
        } catch (error) {
            console.error('获取详细分析结果失败:', error);
            this.showError('获取详细分析结果失败');
        }
    }    generateDetailedResultHTML(result) {
        const risk = result.risk_analysis || {};
        const riskLevel = risk.risk_level || '未知';
        const riskScore = risk.risk_score || risk.total_score || 0;
        
        let html = `
            <div class="detail-section">
                <h6><i class="fas fa-info-circle me-2"></i>基本信息</h6>
                <div class="row">
                    <div class="col-md-6">
                        <p><strong>分析ID:</strong> ${result.analysis_id}</p>
                        <p><strong>分析时间:</strong> ${new Date(result.timestamp).toLocaleString('zh-CN')}</p>
                        <p><strong>风险等级:</strong> <span class="risk-badge ${riskLevel}">${riskLevel}</span></p>
                    </div>
                    <div class="col-md-6">
                        <p><strong>风险分数:</strong> ${riskScore}</p>
                        <p><strong>文本长度:</strong> ${result.transcription ? result.transcription.text.length : (result.input_text ? result.input_text.length : 0)} 字符</p>
                        ${result.original_filename ? `<p><strong>原始文件:</strong> ${result.original_filename}</p>` : ''}
                    </div>
                </div>
            </div>
        `;
        
        // 转录文本或输入文本
        const textContent = result.transcription ? result.transcription.text : result.input_text;
        if (textContent) {
            html += `
                <div class="detail-section">
                    <h6><i class="fas fa-${result.transcription ? 'microphone' : 'keyboard'} me-2"></i>${result.transcription ? '语音转录' : '输入文本'}</h6>
                    <div class="p-3 bg-light rounded">
                        <p class="mb-0">${this.truncateText(textContent, 500)}</p>
                        ${textContent.length > 500 ? '<small class="text-muted">（内容已截断，显示前500字符）</small>' : ''}
                    </div>
                </div>
            `;
        }
        
        // 风险分析详情
        if (risk.key_issues && risk.key_issues.length > 0) {
            html += `
                <div class="detail-section">
                    <h6><i class="fas fa-exclamation-triangle me-2"></i>关键问题</h6>
                    <ul>
                        ${risk.key_issues.map(issue => `<li>${issue}</li>`).join('')}
                    </ul>
                </div>
            `;
        }
        
        // 改进建议
        if (risk.suggestions && risk.suggestions.length > 0) {
            html += `
                <div class="detail-section">
                    <h6><i class="fas fa-lightbulb me-2"></i>改进建议</h6>
                    <ul>
                        ${risk.suggestions.map(suggestion => `<li>${suggestion}</li>`).join('')}
                    </ul>
                </div>
            `;
        }
        
        // 详细分析
        if (risk.detailed_analysis) {
            html += `
                <div class="detail-section">
                    <h6><i class="fas fa-brain me-2"></i>详细分析</h6>
                    <div class="p-3 bg-light rounded">
                        <p class="mb-0">${risk.detailed_analysis}</p>
                    </div>
                </div>
            `;
        }
        
        // 如果是语音转录，显示检测的语言
        if (result.transcription && result.transcription.language) {
            html += `
                <div class="detail-section">
                    <h6><i class="fas fa-language me-2"></i>语言信息</h6>
                    <p><strong>检测语言:</strong> ${result.transcription.language}</p>
                </div>
            `;
        }
        
        return html;
    }

    truncateText(text, maxLength) {
        if (text.length <= maxLength) return text;
        return text.substring(0, maxLength) + '...';
    }

    async loadHistory() {
        try {
            const response = await fetch('/api/analysis-history');
            const data = await response.json();
            
            this.displayHistoryTable(data.results);
        } catch (error) {
            console.error('加载历史记录失败:', error);
            this.showError('加载历史记录失败');
        }
    }

    displayHistoryTable(results) {
        const container = document.getElementById('history-content');
        
        if (!results || results.length === 0) {
            container.innerHTML = `
                <div class="text-center text-muted">
                    <i class="fas fa-clock fa-2x mb-3"></i>
                    <p>暂无分析记录</p>
                </div>
            `;
            return;
        }
        
        let html = `
            <div class="table-responsive">
                <table class="table table-hover history-table">
                    <thead>
                        <tr>
                            <th>时间</th>
                            <th>类型</th>
                            <th>风险等级</th>
                            <th>风险分数</th>
                            <th>文本长度</th>
                            <th>操作</th>
                        </tr>
                    </thead>
                    <tbody>
        `;
          results.forEach(result => {
            const type = result.transcription ? '语音' : '文本';
            const time = new Date(result.timestamp).toLocaleString('zh-CN');
            const riskAnalysis = result.risk_analysis || {};
            
            html += `
                <tr>
                    <td>${time}</td>
                    <td><i class="fas fa-${result.transcription ? 'microphone' : 'keyboard'} me-1"></i>${type}</td>
                    <td><span class="risk-badge ${riskAnalysis.risk_level || '未知'}">${riskAnalysis.risk_level || '未知'}</span></td>
                    <td>${riskAnalysis.risk_score || riskAnalysis.total_score || 0}</td>
                    <td>${result.transcription ? result.transcription.text.length : (result.input_text ? result.input_text.length : 0)}</td>
                    <td>
                        <button class="btn btn-outline-primary btn-sm" onclick="app.showDetailedResult('${result.analysis_id}')">
                            <i class="fas fa-eye"></i>
                        </button>
                    </td>
                </tr>
            `;
        });
        
        html += `
                    </tbody>
                </table>
            </div>
        `;
        
        container.innerHTML = html;
    }

    async loadStatistics() {
        try {
            const response = await fetch('/api/statistics');
            const stats = await response.json();
            
            document.getElementById('total-analyses').textContent = stats.total_analyses;
        } catch (error) {
            console.error('加载统计信息失败:', error);
        }
    }

    getScoreClass(score) {
        if (score < 30) return 'low';
        if (score < 70) return 'medium';
        return 'high';
    }

    showUploadProgress(show) {
        const progressDiv = document.getElementById('upload-progress');
        if (show) {
            progressDiv.style.display = 'block';
            progressDiv.querySelector('.progress-bar').style.width = '100%';
        } else {
            progressDiv.style.display = 'none';
            progressDiv.querySelector('.progress-bar').style.width = '0%';
        }
    }

    showLoadingInElement(elementId, message) {
        const element = document.getElementById(elementId);
        element.innerHTML = `
            <div class="text-center p-4">
                <div class="loading-spinner me-2"></div>
                <span>${message}</span>
            </div>
        `;
    }

    showSuccess(message) {
        this.showToast(message, 'success');
    }

    showError(message) {
        this.showToast(message, 'danger');
    }

    showToast(message, type) {
        // 创建简单的提示消息
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
        alertDiv.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
        alertDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        document.body.appendChild(alertDiv);
          // 自动移除
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.parentNode.removeChild(alertDiv);
            }
        }, 5000);
    }

    // 进度条控制函数
    updateProgressBar(containerId, percentage, message = '') {
        const container = document.getElementById(containerId);
        if (container) {
            const progressBar = container.querySelector('.progress-bar');
            const progressText = container.querySelector('small');
            
            if (progressBar) {
                progressBar.style.width = `${Math.max(0, Math.min(100, percentage))}%`;
                progressBar.setAttribute('aria-valuenow', percentage);
            }
            
            if (progressText && message) {
                progressText.textContent = message;
            }
        }
    }

    showProgressBar(containerId, message = '') {
        const container = document.getElementById(containerId);
        if (container) {
            container.style.display = 'block';
            this.updateProgressBar(containerId, 0, message);
        }
    }    hideProgressBar(containerId) {
        const container = document.getElementById(containerId);
        if (container) {
            container.style.display = 'none';
        }
    }

    getCurrentProgress(containerId) {
        const container = document.getElementById(containerId);
        if (container) {
            const progressBar = container.querySelector('.progress-bar');
            if (progressBar) {
                return parseInt(progressBar.getAttribute('aria-valuenow') || '0');
            }
        }
        return 0;
    }

    showUploadProgress(show) {
        // 兼容旧代码，实际使用showProgressBar/hideProgressBar
        if (show) {
            this.showProgressBar('upload-progress', '正在准备上传...');
        } else {
            this.hideProgressBar('upload-progress');
        }
    }

    // 模拟文本分析进度（用于提供用户反馈）
    simulateTextAnalysisProgress() {
        let progress = 0;
        const stages = [
            { progress: 20, message: '正在加载AI模型...' },
            { progress: 40, message: '开始文本分析...' },
            { progress: 60, message: 'AI正在深度思考...' },
            { progress: 80, message: '生成分析结果...' },
            { progress: 95, message: '整理输出格式...' }
        ];

        let currentStage = 0;
        const interval = setInterval(() => {
            if (currentStage < stages.length) {
                const stage = stages[currentStage];
                this.updateProgressBar('text-analysis-progress', stage.progress, stage.message);
                currentStage++;
            } else {
                clearInterval(interval);
            }
        }, 800); // 每0.8秒更新一次

        return interval;
    }
}

// 初始化应用
const app = new AudioMonitorApp();
