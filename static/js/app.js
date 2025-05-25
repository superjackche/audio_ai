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
    }

    async startRecording() {
        try {
            const response = await fetch('/api/start-recording', { method: 'POST' });
            const result = await response.json();
            
            if (response.ok) {
                this.isRecording = true;
                document.getElementById('start-recording').disabled = true;
                document.getElementById('stop-recording').disabled = false;
                document.getElementById('recording-indicator').style.display = 'flex';
                
                this.showSuccess('录音已开始');
            } else {
                this.showError(result.detail || '录音启动失败');
            }
        } catch (error) {
            console.error('开始录音失败:', error);
            this.showError('录音启动失败');
        }
    }

    async stopRecording() {
        try {
            this.showLoadingInElement('realtime-result', '正在处理录音...');
            
            const response = await fetch('/api/stop-recording', { method: 'POST' });
            const result = await response.json();
            
            this.isRecording = false;
            document.getElementById('start-recording').disabled = false;
            document.getElementById('stop-recording').disabled = true;
            document.getElementById('recording-indicator').style.display = 'none';
            
            if (response.ok) {
                this.displayAnalysisResult(result, 'realtime-result');
                this.showSuccess('录音分析完成');
                this.loadStatistics(); // 更新统计信息
            } else {
                this.showError(result.detail || '录音处理失败');
                document.getElementById('realtime-result').innerHTML = '';
            }
        } catch (error) {
            console.error('停止录音失败:', error);
            this.showError('录音处理失败');
            document.getElementById('realtime-result').innerHTML = '';
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
    }

    async processFile(file) {
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
        
        try {
            this.showUploadProgress(true);
            this.showLoadingInElement('upload-result', '正在分析音频文件...');
            
            const formData = new FormData();
            formData.append('file', file);
            
            const response = await fetch('/api/upload-audio', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            this.showUploadProgress(false);
            
            if (response.ok) {
                this.displayAnalysisResult(result, 'upload-result');
                this.showSuccess('文件分析完成');
                this.loadStatistics();
            } else {
                this.showError(result.detail || '文件分析失败');
                document.getElementById('upload-result').innerHTML = '';
            }
        } catch (error) {
            console.error('文件上传失败:', error);
            this.showError('文件上传失败');
            this.showUploadProgress(false);
            document.getElementById('upload-result').innerHTML = '';
        }
    }

    async analyzeText() {
        const textInput = document.getElementById('text-input');
        const text = textInput.value.trim();
        
        if (!text) {
            this.showError('请输入要分析的文本内容');
            return;
        }
        
        try {
            this.showLoadingInElement('text-result', '正在分析文本...');
            
            const response = await fetch('/api/analyze-text', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text: text })
            });
            
            const result = await response.json();
            
            if (response.ok) {
                this.displayAnalysisResult(result, 'text-result');
                this.showSuccess('文本分析完成');
                this.loadStatistics();
            } else {
                this.showError(result.detail || '文本分析失败');
                document.getElementById('text-result').innerHTML = '';
            }
        } catch (error) {
            console.error('文本分析失败:', error);
            this.showError('文本分析失败');
            document.getElementById('text-result').innerHTML = '';
        }
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
    }

    showDetailedResult(analysisId) {
        const result = this.analysisHistory.find(r => r.analysis_id === analysisId);
        if (!result) return;
        
        const modalBody = document.getElementById('modal-body-content');
        modalBody.innerHTML = this.generateDetailedResultHTML(result);
        
        const modal = new bootstrap.Modal(document.getElementById('result-modal'));
        modal.show();
    }

    generateDetailedResultHTML(result) {
        const risk = result.risk_analysis;
        
        let html = `
            <div class="detail-section">
                <h6><i class="fas fa-info-circle me-2"></i>基本信息</h6>
                <div class="row">
                    <div class="col-md-6">
                        <p><strong>分析ID:</strong> ${result.analysis_id}</p>
                        <p><strong>分析时间:</strong> ${new Date(result.timestamp).toLocaleString('zh-CN')}</p>
                        <p><strong>风险等级:</strong> <span class="risk-badge ${result.summary.risk_level}">${result.summary.risk_level}</span></p>
                    </div>
                    <div class="col-md-6">
                        <p><strong>风险分数:</strong> ${result.summary.risk_score}</p>
                        <p><strong>文本长度:</strong> ${result.summary.text_length || '未知'} 字符</p>
                        ${result.summary.audio_duration ? `<p><strong>音频时长:</strong> ${result.summary.audio_duration.toFixed(2)} 秒</p>` : ''}
                    </div>
                </div>
            </div>
        `;
        
        // 转录文本
        if (result.transcription) {
            html += `
                <div class="detail-section">
                    <h6><i class="fas fa-microphone me-2"></i>语音转录</h6>
                    <div class="p-3 bg-light rounded">
                        <p class="mb-0">${result.transcription.text || result.input_text || '无文本内容'}</p>
                    </div>
                </div>
            `;
        }
        
        // 关键词分析
        if (risk.keyword_analysis.found_keywords) {
            html += `
                <div class="detail-section">
                    <h6><i class="fas fa-key me-2"></i>关键词分析</h6>
                    <p><strong>分数:</strong> ${risk.keyword_analysis.score}</p>
            `;
            
            Object.entries(risk.keyword_analysis.found_keywords).forEach(([level, keywords]) => {
                html += `
                    <h7 class="text-muted">${level}关键词:</h7>
                    <div class="mb-3">
                `;
                keywords.forEach(kw => {
                    html += `<span class="keyword-highlight me-2 mb-1 d-inline-block">${kw.keyword} (${kw.count})</span>`;
                });
                html += `</div>`;
            });
            
            html += `</div>`;
        }
        
        // 上下文分析
        html += `
            <div class="detail-section">
                <h6><i class="fas fa-brain me-2"></i>语义分析</h6>
                <p><strong>分数:</strong> ${risk.context_analysis.score}</p>
                <p><strong>关键短语:</strong> ${risk.context_analysis.key_phrases.join(', ')}</p>
                <p><strong>语义密度:</strong> ${(risk.context_analysis.semantic_density * 100).toFixed(2)}%</p>
            </div>
        `;
        
        // 改进建议
        html += `
            <div class="recommendations">
                <h6><i class="fas fa-lightbulb me-2"></i>改进建议</h6>
                <ul>
                    ${risk.recommendations.map(rec => `<li>${rec}</li>`).join('')}
                </ul>
            </div>
        `;
        
        return html;
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
            
            html += `
                <tr>
                    <td>${time}</td>
                    <td><i class="fas fa-${result.transcription ? 'microphone' : 'keyboard'} me-1"></i>${type}</td>
                    <td><span class="risk-badge ${result.summary.risk_level}">${result.summary.risk_level}</span></td>
                    <td>${result.summary.risk_score}</td>
                    <td>${result.summary.text_length || 0}</td>
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
}

// 初始化应用
const app = new AudioMonitorApp();
