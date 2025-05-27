// AI语音政治风险监测系统 - 前端JavaScript

class AudioMonitorApp {
    constructor() {
        this.isRecording = false;
        this.analysisHistory = [];
        this.currentModal = null;
        
        this.init();
    }

    log(message) {
        console.log("[AudioMonitorApp]", message);
    }

    init() {
        this.bindEvents();
        this.checkSystemStatus();
        this.loadStatistics();
        this.loadServerFiles(); // Load server files on init
        
        // 定期检查系统状态
        setInterval(() => this.checkSystemStatus(), 30000);
    }

    bindEvents() {
        // 录音控制
        document.getElementById('start-recording').addEventListener('click', () => this.startRecording());
        document.getElementById('stop-recording').addEventListener('click', () => this.stopRecording());
          // 服务器文件分析
        document.getElementById('refresh-server-files-btn').addEventListener('click', () => this.loadServerFiles());
        document.getElementById('analyze-server-file-btn').addEventListener('click', () => this.analyzeSelectedServerFile());

        // 文本分析
        document.getElementById('analyze-text').addEventListener('click', () => this.analyzeText());
        
        // 历史记录刷新
        document.getElementById('refresh-history').addEventListener('click', () => this.loadHistory());
        
        // 标签页切换事件
        document.querySelectorAll('[data-bs-toggle="tab"]').forEach(tab => {
            tab.addEventListener('shown.bs.tab', (e) => {
                const targetTabId = e.target.id;
                if (targetTabId === 'history-tab') {
                    this.loadHistory();
                } else if (targetTabId === 'upload-tab') {                    // Automatically load server files when switching to the 'File Analysis' tab
                    // if the list is empty or hasn't been loaded yet.
                    const selectElement = document.getElementById('server-audio-files-select');
                    if (!selectElement.options.length || 
                        (selectElement.options.length === 1 && selectElement.options[0].disabled)) {
                        this.loadServerFiles();
                    }
                }
            });
        });

        // Initial call to load server files if the 'File Analysis' tab is active on page load
        // (though by default 'realtime-tab' is active)
        if (document.getElementById('upload-tab')?.classList.contains('active')) {
            this.loadServerFiles();
        }
    }

    async checkSystemStatus() {
        try {
            const response = await fetch('/api/status');
            const status = await response.json();
            
            this.updateStatusIndicators(status);
            this.updateSystemStatusText(status);
            this.displayCriticalWarnings(status); // Display critical warnings

        } catch (error) {
            console.error('检查系统状态失败:', error);
            this.showError('无法连接到服务器');
            this.displayCriticalWarnings({ system_online: false }); // Show server offline warning
        }
    }    updateStatusIndicators(status) {
        // 更新Whisper状态
        const whisperStatus = document.getElementById('whisper-status');
        this.updateStatusElement(whisperStatus, status.whisper_model_loaded, '语音识别', '已就绪', '未加载');
        
        // 更新LLM状态
        const llmStatus = document.getElementById('llm-status');
        this.updateStatusElement(llmStatus, status.llm_model_loaded, 'AI大模型', '已就绪', '未加载');
        
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
        if (status.whisper_model_loaded && status.llm_model_loaded) {
            statusText.innerHTML = '<i class="fas fa-circle text-success"></i> 系统运行正常';
        } else if (status.whisper_model_loaded || status.llm_model_loaded) {
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
            stopButton.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>停止中...';
            
            const response = await fetch('/api/stop-recording', { method: 'POST' });
            const result = await response.json();
            
            if (response.ok) {
                this.isRecording = false;
                startButton.disabled = false;
                startButton.style.display = 'inline-block';
                startButton.innerHTML = '<i class="fas fa-microphone me-2"></i>开始录音';
                stopButton.disabled = true;
                stopButton.style.display = 'none';
                document.getElementById('recording-indicator').style.display = 'none';
                
                this.showSuccess('录音已停止');
                
                // 处理实时结果
                this.displayRealtimeResult(result);
            } else {
                this.showError(result.detail || '录音停止失败');
                // 恢复按钮状态
                stopButton.disabled = false;
                stopButton.innerHTML = '<i class="fas fa-stop me-2"></i>停止录音';
            }
        } catch (error) {
            console.error('停止录音失败:', error);
            this.showError('录音停止失败');
            
            // 恢复按钮状态
            const stopButton = document.getElementById('stop-recording');
            stopButton.disabled = false;
            stopButton.innerHTML = '<i class="fas fa-stop me-2"></i>停止录音';
        }
    }    displayRealtimeResult(result) {
        const resultContainer = document.getElementById('realtime-result');
        
        // 清空当前结果
        resultContainer.innerHTML = '';
        
        if (result.error) {
            resultContainer.innerHTML = `<div class="alert alert-danger">${result.error}</div>`;
            return;
        }
        
        // 处理并显示每个分析结果
        result.segments.forEach(segment => {
            const card = document.createElement('div');
            card.className = 'card mb-3 analysis-card';
            card.innerHTML = `
                <div class="card-body">
                    <h5 class="card-title">分析结果</h5>
                    <p class="card-text">${segment.text}</p>
                    <p class="card-text"><small class="text-muted">时间戳: ${segment.start} - ${segment.end}</small></p>
                </div>
            `;
            resultContainer.appendChild(card);
        });
        
        // 自动滚动到结果区域
        resultContainer.scrollIntoView({ behavior: 'smooth', block: 'end' });
    }    isElementInViewport(el) {
        if (!el) return false;
        const rect = el.getBoundingClientRect();
        return (
            rect.top >= 0 &&
            rect.left >= 0 &&
            rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
            rect.right <= (window.innerWidth || document.documentElement.clientWidth)
        );
    }

    displayAnalysisResult(result, containerId) {
        const container = document.getElementById(containerId);
        
        // 清空当前结果
        container.innerHTML = '';
        
        if (result.error) {
            container.innerHTML = `<div class="alert alert-danger">${result.error}</div>`;
            return;
        }
        
        // 记录分析历史
        this.analysisHistory.unshift(result);
        if (this.analysisHistory.length > 100) this.analysisHistory.pop(); // Limit memory history

        // 提示用户滚动查看结果
        setTimeout(() => { // setTimeout to allow DOM to update
            const resultCard = container.querySelector('.analysis-card');
            if (resultCard && !this.isElementInViewport(resultCard)) {
                this.showToast('分析结果已生成，请向下滚动查看。', 'info');
            }
        }, 100);
    }

    updateProgressBar(containerId, percentage, message = '') {
        const progressContainer = document.getElementById(containerId);
        if (progressContainer) {
            const progressBar = progressContainer.querySelector('.progress-bar');
            const progressText = progressContainer.querySelector('small'); // Assuming a small tag for text

            if (progressBar) {
                progressBar.style.width = `${percentage}%`;
                progressBar.setAttribute('aria-valuenow', percentage);
                if (message) {
                    progressBar.textContent = message; // Show message on bar itself if no separate text element
                }
            }
            if (progressText && message) {
                progressText.textContent = message; // Or update a specific text element
            }
            if (percentage > 0 && percentage < 100) {
                progressContainer.style.display = 'block';
            } else if (percentage === 0 || percentage === 100) {
                 // Optionally hide or reset
                 // progressContainer.style.display = 'none'; 
            }
        } else {
            this.log(`Progress container with ID '${containerId}' not found.`);
        }
    }

    async loadStatistics() {
        try {
            const response = await fetch('/api/statistics');
            const stats = await response.json();
            
            // 更新统计信息 - Gracefully handle if elements don't exist
            const totalFilesEl = document.getElementById('total-files');
            if (totalFilesEl) totalFilesEl.textContent = stats.total_files;
            
            const analyzedFilesEl = document.getElementById('analyzed-files');
            if (analyzedFilesEl) analyzedFilesEl.textContent = stats.analyzed_files;
            
            const pendingFilesEl = document.getElementById('pending-files');
            if (pendingFilesEl) pendingFilesEl.textContent = stats.pending_files;
            
            const errorFilesEl = document.getElementById('error-files');
            if (errorFilesEl) errorFilesEl.textContent = stats.error_files;

            // Example for an existing element, if you want to use total-analyses for something
            const totalAnalysesEl = document.getElementById('total-analyses');
            if (totalAnalysesEl && stats.total_analyses !== undefined) {
                 totalAnalysesEl.textContent = stats.total_analyses;
            }

        } catch (error) {
            this.log('加载统计信息失败: ' + error); // Use this.log
        }
    }    async loadServerFiles() {
        this.log("加载服务器文件列表...");
        const selectElement = document.getElementById('server-audio-files-select');
        // Set a loading message
        selectElement.innerHTML = '<option value="" selected disabled>正在加载文件列表...</option>';
        try {
            const response = await fetch('/api/list-server-audio-files');
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json(); // data should be { audio_files: ["file1.wav", ...] }
            
            selectElement.innerHTML = ''; // 清空现有选项 (包括loading message)
            
            if (data && data.audio_files && Array.isArray(data.audio_files)) {
                if (data.audio_files.length === 0) {
                    const option = document.createElement('option');
                    option.value = "";
                    option.textContent = "服务器上没有找到音频文件";
                    option.disabled = true;
                    selectElement.appendChild(option);
                } else {
                    data.audio_files.forEach(fileName => {
                        const option = document.createElement('option');
                        option.value = fileName; // Use fileName as value, as expected by analyzeSelectedServerFile
                        option.textContent = fileName; // Display fileName
                        selectElement.appendChild(option);
                    });
                }
                // Only show success if files were actually loaded or an empty list was confirmed
                if (data.audio_files.length > 0) {
                    this.showToast('服务器文件列表已更新', 'success');
                } else {
                    this.showToast('服务器上没有音频文件', 'info');
                }
            } else {
                 throw new Error('服务器返回的文件列表格式不正确');
            }
        } catch (error) {
            console.error('加载服务器文件失败:', error);
            this.showError('无法加载服务器文件列表: ' + error.message);
            // Clear and set error message in select
            selectElement.innerHTML = '<option value="" selected disabled>加载失败，请刷新</option>';
        }
    }    async analyzeSelectedServerFile() {
        const selectElement = document.getElementById('server-audio-files-select'); // 修正ID
        const selectedFile = selectElement.value;
        if (!selectedFile) {
            this.showToast("请选择一个文件进行分析。", "warning");
            return;
        }

        this.log(`开始分析服务器文件: ${selectedFile}`);
        
        // 显示分析状态区域并重置进度条
        const statusArea = document.getElementById('server-file-analysis-status-area');
        statusArea.style.display = 'block';
        
        // 更新按钮状态
        this.updateButtonState('analyze-server-file-btn', true, '<i class="fas fa-spinner fa-spin me-2"></i>分析中...');
        
        // 重置进度条和结果区域
        const progressBar = document.getElementById('server-file-analysis-progress-bar');
        progressBar.style.width = '0%';
        progressBar.setAttribute('aria-valuenow', 0);
        progressBar.textContent = '0%';
        
        const resultContent = document.getElementById('server-file-analysis-result-content');
        resultContent.innerHTML = '<div class="text-muted">正在准备分析...</div>';

        try {
            const formData = new FormData();
            formData.append('filename', selectedFile);

            const response = await fetch('/api/analyze-server-file', {
                method: 'POST',
                body: formData
            });
            
            // Before parsing JSON, check if response is ok and content-type is application/json
            if (!response.ok) {
                let errorDetail = `HTTP error! status: ${response.status} (${response.statusText})`;
                try {
                    const errorResult = await response.json();
                    if (errorResult.detail) {
                        if (Array.isArray(errorResult.detail) && errorResult.detail.length > 0) {
                            // FastAPI often returns an array of error objects.
                            // We'll take the message from the first one, or stringify the whole detail.
                            errorDetail = errorResult.detail.map(err => err.msg || JSON.stringify(err)).join('; ');
                        } else if (typeof errorResult.detail === 'string') {
                            errorDetail = errorResult.detail;
                        } else {
                            errorDetail = JSON.stringify(errorResult.detail);
                        }
                    }
                } catch (e) {
                    // If response.json() fails or no .detail, stick with statusText
                    this.log("Failed to parse JSON error response or extract detail: " + e);
                }
                throw new Error(errorDetail);
            }

            const result = await response.json(); // This is the task submission result, not the analysis result
            
            if (result && result.task_id) {
                this.showToast(`文件 '${selectedFile}' 分析任务已提交。`, 'info');
                // 开始轮询任务状态
                this.pollTaskStatus(result.task_id, selectedFile);
            } else {
                throw new Error(result.detail || '提交分析任务失败，未返回任务ID');
            }
        } catch (error) {
            console.error('分析服务器文件失败:', error); // Log the full error object for inspection
            this.showError('文件分析请求失败: ' + (error.message || "未知错误")); // Ensure error.message is used
            this.updateButtonState('analyze-server-file-btn', false, '<i class="fas fa-play-circle me-2"></i>分析选定文件');
            
            // 显示错误在结果区域
            const resultContent = document.getElementById('server-file-analysis-result-content');
            resultContent.innerHTML = `<div class="alert alert-danger">分析请求失败: ${error.message || "请查看控制台了解详情"}</div>`;
        }
    }    async pollTaskStatus(taskId, filename) {
        this.log(`开始轮询任务状态: ${taskId} for ${filename}`);
        const startTime = Date.now();
        const maxDuration = 300 * 1000; // 5 minutes timeout for polling
        const pollInterval = 3000; // Poll every 3 seconds

        // Update UI to show processing started for server file
        const resultContent = document.getElementById('server-file-analysis-result-content');
        if (resultContent) {
            resultContent.innerHTML = ''; // Clear previous results
        }
        this.updateServerFileProgressBar(0, `正在处理文件: ${filename}...`);
          const intervalId = setInterval(async () => {
            if (Date.now() - startTime > maxDuration) {
                clearInterval(intervalId);
                this.showError(`轮询任务 ${taskId} 超时。`);
                this.updateServerFileProgressBar(100, `处理超时: ${filename}`);
                this.updateButtonState('analyze-server-file-btn', false, '<i class="fas fa-play-circle me-2"></i>分析选定文件');
                return;
            }

            try {
                const response = await fetch(`/api/task-status/${taskId}`);
                if (!response.ok) {
                    // Stop polling on server error, but allow for 404 if task is just not ready
                    if (response.status !== 404) {
                        clearInterval(intervalId);
                        this.showError(`获取任务状态失败: ${response.statusText}`);
                        this.updateServerFileProgressBar(100, `处理失败: ${filename}`);
                        this.updateButtonState('analyze-server-file-btn', false, '<i class="fas fa-play-circle me-2"></i>分析选定文件');
                    } else {
                        this.log(`任务 ${taskId} 状态暂不可用 (404)，将继续轮询...`);
                        // Update progress bar to show it's still working but waiting
                        const elapsedPercentage = Math.min(90, ((Date.now() - startTime) / (maxDuration/2)) * 100); // Cap at 90% while waiting
                        this.updateServerFileProgressBar(elapsedPercentage, `正在处理: ${filename} (等待服务器响应...)`);
                    }
                    return;
                }

                const task = await response.json();
                this.log(`任务 ${taskId} 状态: ${task.status}`);                if (task.status === 'PROCESSING' || task.status === 'QUEUED') {
                    // Update progress bar - make it dynamic or a generic "processing" message
                    const elapsedPercentage = Math.min(90, ((Date.now() - startTime) / (maxDuration/2)) * 100);
                    this.updateServerFileProgressBar(elapsedPercentage, `正在处理: ${filename} (状态: ${task.status})...`);
                } else if (task.status === 'COMPLETED') {
                    clearInterval(intervalId);
                    this.updateServerFileProgressBar(100, `处理完成: ${filename}`);
                    this.showToast(`文件 ${filename} 分析完成。`);
                    // The result from the task should be in task.result
                    // Display the result in the server file analysis result content area
                    this.displayServerFileAnalysisResult(task.result);
                    this.updateButtonState('analyze-server-file-btn', false, '<i class="fas fa-play-circle me-2"></i>分析选定文件');
                    // Optionally, refresh history and statistics
                    this.loadHistory();
                    this.loadStatistics();
                } else if (task.status === 'FAILED') {
                    clearInterval(intervalId);
                    this.showError(`文件 ${filename} 分析失败: ${task.error || '未知错误'}`);
                    this.updateServerFileProgressBar(100, `处理失败: ${filename}`);
                    this.updateButtonState('analyze-server-file-btn', false, '<i class="fas fa-play-circle me-2"></i>分析选定文件');
                }
            } catch (error) {
                // Log network errors or JSON parsing errors during polling
                console.error('轮询任务状态时出错:', error);
                // Don't necessarily stop polling for all errors, could be transient network issues
                // but if it persists, the timeout will catch it.
            }
        }, pollInterval);
    }

    async analyzeText() {
        const text = document.getElementById('input-text').value.trim();
        
        if (!text) {
            return this.showError('请输入要分析的文本');
        }
        
        try {
            const response = await fetch('/api/analyze-text', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text })
            });
            const result = await response.json();
            
            if (response.ok) {
                this.showSuccess('文本分析完成');
                this.displayAnalysisResult(result, 'text-analysis-result');
            } else {
                this.showError(result.detail || '文本分析失败');
            }
        } catch (error) {
            console.error('分析文本失败:', error);
            this.showError('文本分析失败，请稍后重试');
        }
    }

    async loadHistory() {
        this.log("加载历史记录...");
        try {
            const response = await fetch('/api/analysis-history'); // Corrected path
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const history = await response.json();
            
            const container = document.getElementById('history-results');
            container.innerHTML = ''; // 清空现有历史
            
            history.forEach(item => {
                const card = document.createElement('div');
                card.className = 'card mb-3 analysis-card';
                card.innerHTML = `
                    <div class="card-body">
                        <h5 class="card-title">历史记录 - ${item.timestamp}</h5>
                        <p class="card-text">${item.text}</p>
                        <p class="card-text"><small class="text-muted">分析时间: ${item.analysis_time}秒</small></p>
                    </div>
                `;
                container.appendChild(card);
            });
        } catch (error) {
            console.error('加载历史记录失败:', error);
        }
    }

    showSuccess(message) {
        this.showToast(message, 'success');
    }

    showError(message) {
        this.showToast(message, 'error');
    }

    showToast(message, type) {
        const toastContainer = document.getElementById('toast-container');
        if (!toastContainer) {
            console.error("Toast container #toast-container not found. Message:", message);
            // Fallback to alert if toast container is missing
            alert(`${type.toUpperCase()}: ${message}`);
            return;
        }
        const toast = document.createElement('div');
        toast.className = `toast align-items-center text-bg-${type} border-0`;
        toast.role = 'alert';
        toast.ariaLive = 'assertive';
        toast.ariaAtomic = 'true';
        toast.innerHTML = `
            <div class="d-flex">
                <div class="toast-body">
                    ${message}
                </div>
                <button type="button" class="btn-close btn-close-white" data-bs-dismiss="toast" aria-label="Close"></button>
            </div>
        `;
        
        toastContainer.appendChild(toast);
        const bsToast = new bootstrap.Toast(toast);
        bsToast.show();
        
        // 自动移除toast
        setTimeout(() => {
            toastContainer.removeChild(toast);
        }, 5000);
    }

    displayCriticalWarnings(status) {
        const warningContainer = document.getElementById('critical-system-warnings'); // Corrected ID
        if (!warningContainer) {
            this.log("Critical warning container 'critical-system-warnings' not found.");
            return;
        }
        warningContainer.innerHTML = ''; // 清空现有警告
        
        if (!status.system_online) {
            const warning = document.createElement('div');
            warning.className = 'alert alert-danger';
            warning.textContent = '系统离线，请检查服务器状态';
            warningContainer.appendChild(warning);
        }
        
        if (!status.whisper_model_loaded) {
            const warning = document.createElement('div');
            warning.className = 'alert alert-warning';
            warning.textContent = '语音识别模型未加载';
            warningContainer.appendChild(warning);
        }
        
        if (!status.llm_model_loaded) {
            const warning = document.createElement('div');
            warning.className = 'alert alert-warning';
            warning.textContent = 'AI大模型未加载';
            warningContainer.appendChild(warning);
        }
    }

    showDetailedResult(analysisId) {
        // 首先从历史记录中查找
        let result = this.analysisHistory.find(r => r.analysis_id === analysisId);
        
        // 如果在内存中没找到，尝试从服务器获取
        if (!result) {
            this.fetchDetailedResult(analysisId);
            return;
        }
        
        const modalBody = document.getElementById('modal-body-content');
        modalBody.innerHTML = this.generateDetailedResultHTML(result);
        
        const modalElement = document.getElementById('resultModal'); // Corrected ID
        if (modalElement) {
            const modal = new bootstrap.Modal(modalElement);
            modal.show();
        } else {
            console.error("Modal element #resultModal not found");
            this.showError("无法显示详细结果：模态框未找到。");
        }
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
                
                const modalElement = document.getElementById('resultModal'); // Corrected ID
                if (modalElement) {
                    const modal = new bootstrap.Modal(modalElement);
                    modal.show();
                } else {
                    console.error("Modal element #resultModal not found during fetch");
                    this.showError("无法显示详细结果：模态框未找到。");
                }
            } else {
                this.showError('未找到对应的分析记录');
            }
        } catch (error) {
            console.error('获取详细分析结果失败:', error);
            this.showError('获取详细分析结果失败');
        }    }

    updateServerFileProgressBar(percentage, message = '') {
        const progressBar = document.getElementById('server-file-analysis-progress-bar');
        if (progressBar) {
            progressBar.style.width = `${percentage}%`;
            progressBar.setAttribute('aria-valuenow', percentage);
            progressBar.textContent = `${percentage}%`;
            
            if (message) {
                // 在进度条上显示消息
                progressBar.title = message;
            }
        } else {
            this.log(`Server file progress bar not found.`);
        }
    }

    displayServerFileAnalysisResult(result) {
        const resultContent = document.getElementById('server-file-analysis-result-content');
        if (!resultContent) {
            this.log('Server file analysis result content area not found.');
            return;
        }
        
        // 清空当前结果
        resultContent.innerHTML = '';
        
        if (result.error) {
            resultContent.innerHTML = `<div class="alert alert-danger">
                <h5>分析失败</h5>
                <p>${result.error}</p>
            </div>`;
            return;
        }
        
        // 创建分析结果卡片
        const resultCard = document.createElement('div');
        resultCard.className = 'card analysis-card';
        
        let cardContent = `
            <div class="card-body">
                <h5 class="card-title">
                    <i class="fas fa-file-audio me-2"></i>文件分析结果
                </h5>
        `;
        
        // 显示转录文本
        if (result.transcription) {
            cardContent += `
                <div class="mb-3">
                    <h6 class="text-primary">
                        <i class="fas fa-microphone me-2"></i>语音转录
                    </h6>
                    <p class="card-text">${result.transcription}</p>
                </div>
            `;
        }
        
        // 显示分析结果
        if (result.analysis) {
            cardContent += `
                <div class="mb-3">
                    <h6 class="text-success">
                        <i class="fas fa-brain me-2"></i>智能分析
                    </h6>
                    <p class="card-text">${result.analysis}</p>
                </div>
            `;
        }
        
        // 显示风险评估
        if (result.risk_score !== undefined) {
            const riskLevel = this.getRiskLevel(result.risk_score);
            cardContent += `
                <div class="mb-3">
                    <h6 class="text-warning">
                        <i class="fas fa-exclamation-triangle me-2"></i>风险评估
                    </h6>
                    <div class="d-flex align-items-center">
                        <span class="badge bg-${riskLevel.color} me-2">${riskLevel.label}</span>
                        <span>风险评分: ${result.risk_score}/100</span>
                    </div>
                </div>
            `;
        }
        
        // 显示关键词
        if (result.keywords && result.keywords.length > 0) {
            cardContent += `
                <div class="mb-3">
                    <h6 class="text-info">
                        <i class="fas fa-tags me-2"></i>关键词
                    </h6>
                    <div>
                        ${result.keywords.map(keyword => 
                            `<span class="badge bg-secondary me-1">${keyword}</span>`
                        ).join('')}
                    </div>
                </div>
            `;
        }
        
        // 显示时间信息
        if (result.timestamp) {
            cardContent += `
                <div class="text-muted">
                    <small>
                        <i class="fas fa-clock me-1"></i>
                        分析时间: ${new Date(result.timestamp).toLocaleString()}
                    </small>
                </div>
            `;
        }
        
        cardContent += `
            </div>
        `;
        
        resultCard.innerHTML = cardContent;
        resultContent.appendChild(resultCard);
        
        // 记录到分析历史
        this.analysisHistory.unshift(result);
        if (this.analysisHistory.length > 100) this.analysisHistory.pop();
        
        // 提示用户查看结果
        this.showToast('分析结果已生成，请查看下方内容。', 'success');
    }

    getRiskLevel(score) {
        if (score >= 80) return { label: '高风险', color: 'danger' };
        if (score >= 60) return { label: '中风险', color: 'warning' };
        if (score >= 40) return { label: '低风险', color: 'info' };
        return { label: '安全', color: 'success' };
    }

    updateButtonState(buttonId, disabled, htmlContent = null) {
        const button = document.getElementById(buttonId);
        if (button) {
            button.disabled = disabled;
            if (htmlContent !== null) {
                button.innerHTML = htmlContent;
            }
        } else {            this.log(`Button with ID '${buttonId}' not found.`);
        }
    }

    // 专门用于服务器文件分析的进度条更新
    updateServerFileProgressBar(percentage, message = '') {
        const statusArea = document.getElementById('server-file-analysis-status-area');
        const progressBar = document.getElementById('server-file-analysis-progress-bar');
        
        if (statusArea && progressBar) {
            // 显示状态区域
            statusArea.style.display = 'block';
            
            // 更新进度条
            progressBar.style.width = `${percentage}%`;
            progressBar.setAttribute('aria-valuenow', percentage);
            progressBar.textContent = `${percentage}%`;
            
            // 如果有消息，可以在这里显示
            if (message) {
                this.log(`Progress: ${message}`);
            }
        } else {
            this.log('服务器文件分析进度条元素未找到');
        }
    }

    // 专门用于显示服务器文件分析结果
    displayServerFileAnalysisResult(result) {
        const resultContent = document.getElementById('server-file-analysis-result-content');
        if (!resultContent) {
            this.log('服务器文件分析结果容器未找到');
            return;
        }

        try {
            if (!result) {
                resultContent.innerHTML = '<div class="alert alert-warning">未收到分析结果</div>';
                return;
            }

            // 构建结果HTML
            let html = '<div class="analysis-results">';
            
            // 风险等级
            if (result.risk_level) {
                const riskClass = this.getRiskLevelClass(result.risk_level);
                html += `<div class="alert ${riskClass}">
                    <h5><i class="fas fa-exclamation-triangle me-2"></i>风险等级: ${result.risk_level}</h5>
                </div>`;
            }

            // 分析结果
            if (result.analysis_result) {
                html += `<div class="card mb-3">
                    <div class="card-header"><h6>分析结果</h6></div>
                    <div class="card-body">${result.analysis_result}</div>
                </div>`;
            }

            // 检测到的风险
            if (result.detected_risks && result.detected_risks.length > 0) {
                html += '<div class="card mb-3">';
                html += '<div class="card-header"><h6>检测到的风险</h6></div>';
                html += '<div class="card-body"><ul class="list-group list-group-flush">';
                result.detected_risks.forEach(risk => {
                    html += `<li class="list-group-item">${risk}</li>`;
                });
                html += '</ul></div></div>';
            }

            // 转录文本
            if (result.transcription) {
                html += `<div class="card mb-3">
                    <div class="card-header"><h6>转录文本</h6></div>
                    <div class="card-body"><pre style="white-space: pre-wrap;">${result.transcription}</pre></div>
                </div>`;
            }

            // 分析时间
            if (result.analysis_time) {
                html += `<div class="text-muted small">分析时间: ${result.analysis_time}</div>`;
            }

            html += '</div>';
            resultContent.innerHTML = html;

        } catch (error) {
            console.error('显示分析结果时出错:', error);
            resultContent.innerHTML = `<div class="alert alert-danger">显示结果时出错: ${error.message}</div>`;
        }
    }

    // 获取风险等级对应的CSS类
    getRiskLevelClass(riskLevel) {
        switch (riskLevel?.toLowerCase()) {
            case 'high':
            case '高':
                return 'alert-danger';
            case 'medium':
            case '中':
                return 'alert-warning';
            case 'low':
            case '低':
                return 'alert-success';
            default:
                return 'alert-info';
        }
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.app = new AudioMonitorApp();
});
