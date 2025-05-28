// AI语音政治风险监测系统 - 前端JavaScript

class AudioMonitorApp {
    constructor() {
        this.isMonitoring = false; // 改为监控状态
        this.analysisHistory = [];
        this.currentModal = null;
        this.currentClassroom = null;
        this.monitoringInterval = null;
        
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
    }    bindEvents() {
        // 移除录音相关按钮事件绑定，因为已改为纯展示的监控界面
        
        // 服务器文件分析
        document.getElementById('refresh-server-files').addEventListener('click', () => this.loadServerFiles());
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
        // 更新SenseVoice状态
        const speechStatus = document.getElementById('whisper-status');
        this.updateStatusElement(speechStatus, status.speech_model_loaded, 'SenseVoice语音识别', '已就绪', '未加载');
        
        // 更新LLM状态
        const llmStatus = document.getElementById('llm-status');
        this.updateStatusElement(llmStatus, status.llm_model_loaded, 'AI大模型', '已就绪', '未加载');
        
        // 更新监控状态 (改为固定显示可用状态)
        const recordingStatus = document.getElementById('recording-status');
        this.updateStatusElement(recordingStatus, true, '音频监控接入', '系统就绪', '离线');
        
        // 移除按钮状态更新，因为已经删除了录音按钮
    }

    updateStatusElement(element, isOnline, feature, onlineText, offlineText) {
        element.className = `status-indicator ${isOnline ? 'status-online' : 'status-offline'}`;
        const small = element.querySelector('small');
        small.textContent = isOnline ? onlineText : offlineText;
        small.className = isOnline ? 'text-success' : 'text-danger';
    }    updateSystemStatusText(status) {
        const statusText = document.getElementById('system-status');
        // 修复字段名不匹配问题：使用 speech_model_loaded 而不是 whisper_model_loaded
        if (status.speech_model_loaded && status.llm_model_loaded) {
            statusText.innerHTML = '<i class="fas fa-circle text-success"></i> 系统运行正常';
        } else if (status.speech_model_loaded || status.llm_model_loaded) {
            statusText.innerHTML = '<i class="fas fa-circle text-warning"></i> 部分功能可用';
        } else {
            statusText.innerHTML = '<i class="fas fa-circle text-danger"></i> 系统初始化中';
        }
    }// 学校音频监控系统（纯展示界面，无实际录音功能）
    // 这些函数已被移除，因为新的监控界面是纯展示性的

    // 显示实时分析结果
    displayRealtimeResult(result) {
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
        resultContent.innerHTML = '<div class="text-muted">正在准备分析...</div>';        try {
            const response = await fetch('/api/analyze-server-file', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    filename: selectedFile
                })
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
        const pollInterval = 2000; // 更频繁的轮询，每2秒检查一次
          // 进度阶段定义 - 优化时间分配
        const progressStages = [
            { percentage: 8, message: '正在初始化任务...', duration: 3000 },
            { percentage: 18, message: '正在加载音频文件...', duration: 8000 },
            { percentage: 30, message: '正在预处理音频...', duration: 12000 },
            { percentage: 55, message: '正在进行语音识别...', duration: 25000 },
            { percentage: 75, message: '正在分析文本内容...', duration: 15000 },
            { percentage: 90, message: '正在生成风险评估...', duration: 8000 },
            { percentage: 98, message: '正在整理分析结果...', duration: 4000 }
        ];
        
        let currentStageIndex = 0;
        let simulatedProgress = 0;
        let lastProgressUpdate = startTime;
        let stageStartTime = startTime;

        // Update UI to show processing started for server file
        const resultContent = document.getElementById('server-file-analysis-result-content');
        if (resultContent) {
            resultContent.innerHTML = ''; // Clear previous results
        }
        
        // 立即显示第一个阶段
        this.updateServerFileProgressBar(progressStages[0].percentage, progressStages[0].message);
        
        const intervalId = setInterval(async () => {
            const elapsedTime = Date.now() - startTime;
            
            if (elapsedTime > maxDuration) {
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
                        this.updateButtonState('analyze-server-file-btn', false, '<i class="fas fa-play-circle me-2"></i>分析选定文件');                    } else {
                        // 任务还没开始，智能显示初始化进度
                        if (elapsedTime < 5000) {
                            // 前5秒显示任务初始化
                            const initProgress = Math.min(8, (elapsedTime / 5000) * 8);
                            this.updateServerFileProgressBar(initProgress, `正在初始化分析任务 ${filename}...`);
                        } else if (elapsedTime < 15000) {
                            // 5-15秒显示文件加载
                            const loadProgress = 8 + Math.min(10, ((elapsedTime - 5000) / 10000) * 10);
                            this.updateServerFileProgressBar(loadProgress, `正在准备音频文件 ${filename}...`);
                        } else {
                            // 15秒后显示等待服务器响应
                            this.updateServerFileProgressBar(18, `正在等待服务器响应...`);
                        }
                    }
                    return;
                }

                const task = await response.json();
                this.log(`任务 ${taskId} 状态: ${task.status}`);
                  if (task.status === 'processing' || task.status === 'pending') {
                    // 智能进度计算算法
                    const currentTime = Date.now();
                    const timeSinceStageStart = currentTime - stageStartTime;
                    const currentStage = progressStages[currentStageIndex];
                    
                    // 检查是否应该进入下一个阶段
                    if (currentStageIndex < progressStages.length - 1) {
                        const expectedStageEnd = stageStartTime + currentStage.duration;
                        
                        if (currentTime >= expectedStageEnd) {
                            // 进入下一个阶段
                            currentStageIndex++;
                            stageStartTime = currentTime;
                            this.log(`进入阶段 ${currentStageIndex + 1}: ${progressStages[currentStageIndex].message}`);
                        }
                    }
                    
                    // 计算当前阶段内的进度
                    const stageProgress = Math.min(1, timeSinceStageStart / progressStages[currentStageIndex].duration);
                    
                    // 计算总体进度
                    const prevStagePercentage = currentStageIndex > 0 ? progressStages[currentStageIndex - 1].percentage : 0;
                    const currentStagePercentage = progressStages[currentStageIndex].percentage;
                    const stageRange = currentStagePercentage - prevStagePercentage;
                    
                    // 使用缓动函数使进度更平滑
                    const easedProgress = this.easeOutCubic(stageProgress);
                    simulatedProgress = prevStagePercentage + (easedProgress * stageRange);
                    
                    // 确保进度不会倒退
                    simulatedProgress = Math.max(simulatedProgress, simulatedProgress);
                    
                    // 添加小幅随机波动，使进度看起来更真实
                    const randomVariation = (Math.random() - 0.5) * 0.5; // ±0.25%
                    const displayProgress = Math.min(97, Math.max(simulatedProgress + randomVariation, simulatedProgress - 1));
                    
                    this.updateServerFileProgressBar(
                        displayProgress,
                        progressStages[currentStageIndex].message
                    );
                } else if (task.status === 'completed') {
                    clearInterval(intervalId);
                    // 快速完成最后的进度
                    this.updateServerFileProgressBar(100, `分析完成: ${filename}`);
                    this.showToast(`文件 ${filename} 分析完成。`);
                    // The result from the task should be in task.result
                    // Display the result in the server file analysis result content area
                    this.displayServerFileAnalysisResult(task.result);
                    this.updateButtonState('analyze-server-file-btn', false, '<i class="fas fa-play-circle me-2"></i>分析选定文件');
                    // Optionally, refresh history and statistics
                    this.loadHistory();
                    this.loadStatistics();
                } else if (task.status === 'failed') {
                    clearInterval(intervalId);
                    this.showError(`文件 ${filename} 分析失败: ${task.error || '未知错误'}`);
                    this.updateServerFileProgressBar(100, `处理失败: ${filename}`);
                    this.updateButtonState('analyze-server-file-btn', false, '<i class="fas fa-play-circle me-2"></i>分析选定文件');
                }            } catch (error) {
                // Log network errors or JSON parsing errors during polling
                console.error('轮询任务状态时出错:', error);
                
                // 智能网络错误处理
                const networkRetryMessage = this.getNetworkRetryMessage(error);
                
                // 在网络错误时，保持当前进度但显示重连状态
                if (currentStageIndex < progressStages.length - 1) {
                    // 保持当前进度，但添加网络状态指示
                    const currentProgress = Math.max(simulatedProgress, progressStages[currentStageIndex].percentage);
                    this.updateServerFileProgressBar(
                        currentProgress,
                        `${progressStages[currentStageIndex].message} ${networkRetryMessage}`
                    );
                } else {
                    // 如果已经是最后阶段，显示即将完成但网络重连中
                    this.updateServerFileProgressBar(
                        95,
                        `即将完成... ${networkRetryMessage}`
                    );
                }
            }
        }, pollInterval);
    }async analyzeText() {
        const text = document.getElementById('text-input').value.trim();
        
        if (!text) {
            return this.showError('请输入要分析的文本');
        }

        const progressContainer = document.getElementById('text-analysis-progress');
        const resultContainer = document.getElementById('text-result');
        
        try {
            // 显示进度条
            if (progressContainer) {
                progressContainer.style.display = 'block';
                const progressBar = progressContainer.querySelector('.progress-bar');
                const progressText = document.getElementById('text-progress-text');
                
                if (progressBar) {
                    progressBar.style.width = '20%';
                    progressBar.setAttribute('aria-valuenow', '20');
                }
                if (progressText) {
                    progressText.textContent = '正在分析文本...';
                }
            }

            const response = await fetch('/api/analyze-text', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text })
            });
            
            // 更新进度
            if (progressContainer) {
                const progressBar = progressContainer.querySelector('.progress-bar');
                if (progressBar) {
                    progressBar.style.width = '80%';
                    progressBar.setAttribute('aria-valuenow', '80');
                }
            }

            const result = await response.json();
            
            if (response.ok) {
                // 完成进度
                if (progressContainer) {
                    const progressBar = progressContainer.querySelector('.progress-bar');
                    const progressText = document.getElementById('text-progress-text');
                    if (progressBar) {
                        progressBar.style.width = '100%';
                        progressBar.setAttribute('aria-valuenow', '100');
                    }
                    if (progressText) {
                        progressText.textContent = '分析完成';
                    }
                    
                    // 隐藏进度条
                    setTimeout(() => {
                        progressContainer.style.display = 'none';
                    }, 1000);                }
                
                this.showSuccess('文本分析完成');
                this.displayTextAnalysisResult(result, resultContainer);
                
                // 自动刷新历史记录和统计信息
                this.loadHistory();
                this.loadStatistics();
            } else {
                this.showError(result.detail || '文本分析失败');
                if (progressContainer) {
                    progressContainer.style.display = 'none';
                }
            }
        } catch (error) {
            console.error('分析文本失败:', error);
            this.showError('文本分析失败，请稍后重试');
            if (progressContainer) {
                progressContainer.style.display = 'none';
            }
        }
    }

    displayTextAnalysisResult(result, container) {
        if (!container) {
            this.log('Text result container not found');
            return;
        }

        container.innerHTML = '';

        if (result.error) {
            container.innerHTML = `<div class="alert alert-danger">${result.error}</div>`;
            return;
        }

        // 创建结果卡片
        const card = document.createElement('div');
        card.className = 'card analysis-card';
        
        let cardContent = `
            <div class="card-body">
                <h5 class="card-title">
                    <i class="fas fa-brain me-2"></i>文本分析结果
                </h5>
        `;

        // 显示风险评估
        if (result.risk_assessment) {
            const risk = result.risk_assessment;
            const riskClass = this.getRiskLevelClass(risk.risk_level);
            
            cardContent += `
                <div class="mb-3">
                    <h6 class="text-primary">
                        <i class="fas fa-exclamation-triangle me-2"></i>风险评估
                    </h6>
                    <div class="alert ${riskClass}">
                        <strong>风险等级：</strong>${risk.risk_level}<br>
                        <strong>风险评分：</strong>${risk.risk_score}/100
                    </div>
                </div>
            `;

            // 显示关键问题
            if (risk.key_issues && risk.key_issues.length > 0) {
                cardContent += `
                    <div class="mb-3">
                        <h6 class="text-warning">
                            <i class="fas fa-list me-2"></i>关键问题
                        </h6>
                        <ul class="list-group list-group-flush">
                `;
                risk.key_issues.forEach(issue => {
                    cardContent += `<li class="list-group-item">${issue}</li>`;
                });
                cardContent += `</ul></div>`;
            }
        }

        // 显示处理信息
        if (result.processing_info) {
            const info = result.processing_info;
            cardContent += `
                <div class="mb-3">
                    <h6 class="text-info">
                        <i class="fas fa-info-circle me-2"></i>处理信息
                    </h6>
                    <div class="row">
                        <div class="col-md-6">
                            <small class="text-muted">
                                文本长度：${info.text_length} 字符<br>
                                处理方法：${info.processing_method}
                            </small>
                        </div>
                    </div>
                </div>
            `;
        }

        // 显示时间戳
        if (result.timestamp) {
            cardContent += `
                <div class="text-muted">
                    <small>
                        <i class="fas fa-clock me-1"></i>
                        分析时间：${new Date(result.timestamp).toLocaleString()}
                    </small>
                </div>
            `;
        }

        cardContent += '</div>';
        card.innerHTML = cardContent;
        container.appendChild(card);

        // 记录到分析历史
        this.analysisHistory.unshift(result);
        if (this.analysisHistory.length > 100) this.analysisHistory.pop();

        // 提示用户查看结果
        this.showToast('文本分析结果已生成，请查看下方内容。', 'success');
    }    async loadHistory() {
        this.log("加载历史记录...");
        const container = document.getElementById('history-results');
        if (!container) {
            this.log('History results container not found');
            return;
        }

        try {
            // 显示加载状态
            container.innerHTML = '<div class="text-center"><i class="fas fa-spinner fa-spin me-2"></i>正在加载历史记录...</div>';
            
            const response = await fetch('/api/analysis-history');
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            
            container.innerHTML = ''; // 清空现有历史
            
            if (!data.results || data.results.length === 0) {
                container.innerHTML = '<div class="alert alert-info">暂无分析历史记录</div>';
                return;
            }
            
            // 显示历史记录
            data.results.forEach(item => {
                const card = document.createElement('div');
                card.className = 'card mb-3 analysis-card';
                
                let cardContent = `
                    <div class="card-body">
                        <h6 class="card-title">
                            <i class="fas fa-history me-2"></i>
                            ${item.analysis_id || '未知ID'}
                        </h6>
                `;

                // 显示风险评估
                if (item.risk_assessment) {
                    const risk = item.risk_assessment;
                    const riskClass = this.getRiskLevelClass(risk.risk_level);
                    cardContent += `
                        <div class="mb-2">
                            <span class="badge ${riskClass.replace('alert-', 'bg-')}">${risk.risk_level}</span>
                            <small class="text-muted ms-2">评分: ${risk.risk_score}/100</small>
                        </div>
                    `;
                }                // 显示处理信息
                if (item.processing_info) {
                    let processingInfoText = `
                        文件: ${item.original_filename || '未知文件'}<br>
                        处理方法: ${item.processing_info.processing_method || '未知方法'}<br>
                        文本长度: ${item.processing_info.text_length || 0} 字符
                    `;
                    
                    // 添加音频时长显示
                    if (item.processing_info.audio_duration !== undefined && item.processing_info.audio_duration !== null) {
                        const durationMinutes = Math.floor(item.processing_info.audio_duration / 60);
                        const durationSeconds = Math.floor(item.processing_info.audio_duration % 60);
                        processingInfoText += `<br>音频时长: ${durationMinutes}:${durationSeconds.toString().padStart(2, '0')}`;
                    }
                    
                    cardContent += `
                        <p class="card-text">
                            <small class="text-muted">
                                ${processingInfoText}
                            </small>
                        </p>
                    `;
                }

                // 显示时间
                if (item.timestamp) {
                    cardContent += `
                        <p class="card-text">
                            <small class="text-muted">
                                <i class="fas fa-clock me-1"></i>
                                ${new Date(item.timestamp).toLocaleString()}
                            </small>
                        </p>
                    `;
                }

                cardContent += `
                        <button class="btn btn-sm btn-outline-primary" onclick="app.showDetailedResult('${item.analysis_id}')">
                            查看详情
                        </button>
                    </div>
                `;
                
                card.innerHTML = cardContent;
                container.appendChild(card);
            });

            this.showToast(`已加载 ${data.results.length} 条历史记录`, 'success');
            
        } catch (error) {
            console.error('加载历史记录失败:', error);
            container.innerHTML = `<div class="alert alert-danger">加载历史记录失败: ${error.message}</div>`;
            this.showError('加载历史记录失败');
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
    }    displayCriticalWarnings(status) {
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
        
        // 修复字段名不匹配问题：使用 speech_model_loaded 而不是 whisper_model_loaded
        if (!status.speech_model_loaded) {
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
            `;        }
        
        // 显示处理信息（包括音频时长）
        if (result.processing_info) {
            const info = result.processing_info;
            cardContent += `
                <div class="mb-3">
                    <h6 class="text-info">
                        <i class="fas fa-info-circle me-2"></i>处理信息
                    </h6>
                    <div class="row">
                        <div class="col-md-6">
                            <small class="text-muted">
                                文本长度：${info.text_length || 0} 字符<br>
                                处理方法：${info.processing_method || '未知'}
                            </small>
                        </div>
                        <div class="col-md-6">
                            <small class="text-muted">
            `;
            
            // 添加音频时长显示
            if (info.audio_duration !== undefined && info.audio_duration !== null) {
                const durationMinutes = Math.floor(info.audio_duration / 60);
                const durationSeconds = Math.floor(info.audio_duration % 60);
                cardContent += `音频时长：${durationMinutes}:${durationSeconds.toString().padStart(2, '0')}<br>`;
            }
            
            if (info.total_time) {
                cardContent += `处理时间：${info.total_time.toFixed(2)} 秒`;
            }
            
            cardContent += `
                            </small>
                        </div>
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
    }    // 专门用于服务器文件分析的进度条更新 - 优化版
    updateServerFileProgressBar(percentage, message = '') {
        const statusArea = document.getElementById('server-file-analysis-status-area');
        const progressBar = document.getElementById('server-file-analysis-progress-bar');
        
        if (statusArea && progressBar) {
            // 显示状态区域
            statusArea.style.display = 'block';
            
            // 添加平滑动画效果
            progressBar.style.transition = 'width 0.3s ease-in-out';
            
            // 根据进度状态设置不同的颜色和样式
            progressBar.classList.remove('bg-success', 'bg-warning', 'bg-danger');
            if (percentage >= 100) {
                progressBar.classList.add('bg-success');
                progressBar.classList.remove('progress-bar-animated', 'progress-bar-striped');
            } else if (percentage >= 80) {
                progressBar.classList.add('bg-warning');
            } else {
                // 保持默认的蓝色，确保动画效果
                if (!progressBar.classList.contains('progress-bar-animated')) {
                    progressBar.classList.add('progress-bar-animated', 'progress-bar-striped');
                }
            }
            
            // 更新进度条宽度和值
            const roundedPercentage = Math.round(percentage);
            progressBar.style.width = `${roundedPercentage}%`;
            progressBar.setAttribute('aria-valuenow', roundedPercentage);
            
            // 进度条文本显示优化
            if (roundedPercentage < 100) {
                progressBar.textContent = `${roundedPercentage}%`;
            } else {
                progressBar.textContent = '完成';
            }
            
            // 状态消息显示和动画
            if (message) {
                // 查找或创建状态消息元素
                let statusMessage = document.getElementById('server-file-status-message');
                if (!statusMessage) {
                    statusMessage = document.createElement('div');
                    statusMessage.id = 'server-file-status-message';
                    statusMessage.className = 'text-muted small mt-2 fade-in';
                    statusMessage.style.transition = 'opacity 0.3s ease-in-out';
                    statusArea.appendChild(statusMessage);
                }
                
                // 添加状态指示图标
                let icon = 'fas fa-spinner fa-spin';
                if (percentage >= 100) {
                    icon = 'fas fa-check-circle text-success';
                } else if (message.includes('失败') || message.includes('错误') || message.includes('超时')) {
                    icon = 'fas fa-exclamation-triangle text-warning';
                } else if (message.includes('网络重连')) {
                    icon = 'fas fa-wifi text-warning';
                }
                
                statusMessage.innerHTML = `<i class="${icon} me-1"></i>${message}`;
                statusMessage.style.opacity = '1';
                this.log(`Progress: ${roundedPercentage}% - ${message}`);
            }
            
            // 添加进度完成时的特殊效果
            if (percentage >= 100) {
                setTimeout(() => {
                    if (progressBar.classList.contains('bg-success')) {
                        // 添加一个短暂的闪烁效果表示完成
                        progressBar.style.boxShadow = '0 0 10px rgba(40, 167, 69, 0.5)';
                        setTimeout(() => {
                            progressBar.style.boxShadow = '';
                        }, 500);
                    }
                }, 100);
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
        if (!riskLevel) return 'alert-info';
        
        const level = riskLevel.toString().toLowerCase();
        if (level.includes('高') || level.includes('high') || level.includes('danger')) {
            return 'alert-danger';
        } else if (level.includes('中') || level.includes('medium') || level.includes('warning')) {
            return 'alert-warning';
        } else if (level.includes('低') || level.includes('low') || level.includes('success')) {
            return 'alert-success';
        } else {
            return 'alert-info';
        }
    }

    generateDetailedResultHTML(result) {
        if (!result) return '<p>无详细结果可显示</p>';

        let html = '<div class="detailed-analysis-result">';
        
        // 基本信息
        html += `<h6>分析ID: ${result.analysis_id || '未知'}</h6>`;
        html += `<p><strong>文件名:</strong> ${result.original_filename || '未知文件'}</p>`;
        html += `<p><strong>分析时间:</strong> ${result.timestamp ? new Date(result.timestamp).toLocaleString() : '未知时间'}</p>`;

        // 风险评估
        if (result.risk_assessment) {
            const risk = result.risk_assessment;
            html += `<div class="mb-3">
                <h6>风险评估</h6>
                <div class="alert ${this.getRiskLevelClass(risk.risk_level)}">
                    <strong>风险等级:</strong> ${risk.risk_level}<br>
                    <strong>风险评分:</strong> ${risk.risk_score}/100
                </div>
            </div>`;

            if (risk.key_issues && risk.key_issues.length > 0) {
                html += '<div class="mb-3"><h6>关键问题</h6><ul>';
                risk.key_issues.forEach(issue => {
                    html += `<li>${issue}</li>`;
                });
                html += '</ul></div>';
            }
        }        // 处理信息
        if (result.processing_info) {
            const info = result.processing_info;
            html += `<div class="mb-3">
                <h6>处理信息</h6>
                <p><strong>文本长度:</strong> ${info.text_length || 0} 字符</p>
                <p><strong>处理方法:</strong> ${info.processing_method || '未知'}</p>
            `;
            
            // 添加音频时长显示
            if (info.audio_duration !== undefined && info.audio_duration !== null) {
                const durationMinutes = Math.floor(info.audio_duration / 60);
                const durationSeconds = Math.floor(info.audio_duration % 60);
                html += `<p><strong>音频时长:</strong> ${durationMinutes}:${durationSeconds.toString().padStart(2, '0')} (${info.audio_duration.toFixed(2)} 秒)</p>`;
            }
            
            if (info.total_time) {
                html += `<p><strong>处理时间:</strong> ${info.total_time.toFixed(2)} 秒</p>`;
            }
            html += '</div>';
        }        html += '</div>';
        return html;
    }

    // 缓动函数：平滑的立方缓出效果
    easeOutCubic(t) {
        return 1 - Math.pow(1 - t, 3);
    }    // 防抖函数：防止进度更新过于频繁
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    // 获取网络重试消息
    getNetworkRetryMessage(error) {
        if (error.name === 'TypeError' && error.message.includes('fetch')) {
            return '(网络连接中断，正在重试...)';
        } else if (error.message.includes('timeout')) {
            return '(请求超时，正在重连...)';
        } else if (error.message.includes('404')) {
            return '(任务准备中...)';
        } else {
            return '(网络重连中...)';
        }
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.app = new AudioMonitorApp();
});
