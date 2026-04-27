document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const langToggle = document.getElementById('lang-toggle');
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');
    const previewSection = document.getElementById('preview-section');
    const imagePreview = document.getElementById('image-preview');
    const previewFilename = document.getElementById('preview-filename');
    const previewFilesize = document.getElementById('preview-filesize');
    
    const btnClear = document.getElementById('btn-clear');
    const btnAnalyze = document.getElementById('btn-analyze');

    // State Views
    const stateEmpty = document.getElementById('state-empty');
    const stateLoading = document.getElementById('state-loading');
    const stateError = document.getElementById('state-error');
    const stateResult = document.getElementById('state-result');
    const errorMessage = document.getElementById('error-message');
    const elapsedTimeLabel = document.getElementById('elapsed-time');

    let currentFile = null;
    let lastReport = null;
    let lastHealth = null;
    let currentLang = localStorage.getItem('ai-image-judge-lang') || 'en';

    const I18N = {
        en: {
            pageTitle: 'AI Image Judge | Image Forensics Workbench',
            switchLanguage: '中文',
            targetFile: 'Target File',
            uploadTitle: 'Drop an image here, or <strong>click to upload</strong>',
            uploadDesc: 'Supports JPEG, PNG, WEBP, GIF. Max 25MB.',
            clear: 'Clear',
            analyze: 'Analyze',
            forensicReport: 'Forensic Report',
            emptyState: 'Upload an image on the left and click “Analyze”.<br>The forensic report will appear here.',
            loadingState: 'Running multi-layer forensic analysis...',
            errorDefault: 'Analysis failed. Please try again.',
            confidence: 'Confidence',
            modelProbability: 'Model Probability',
            modelDisclaimer: 'AI-detection models are statistical signals. Treat the result as supporting evidence, not final proof.',
            aiGenerated: 'AI-generated',
            realPhoto: 'Camera-originated',
            fileInfo: 'File Fingerprint & Basics',
            mimeType: 'MIME Type',
            resolution: 'Resolution',
            fileSize: 'File Size',
            modelStatus: 'Model Status',
            sha256: 'SHA256 Hash',
            evidence: 'Evidence',
            warnings: 'Warnings',
            metadataDetails: 'Metadata Details',
            rawReport: 'Raw Detection Report',
            c2paTool: 'C2PA Tool',
            exifTool: 'ExifTool',
            modelReady: 'Model Ready',
            modelOffline: 'Model Offline',
            model: 'Model',
            uploadImageAlert: 'Please upload an image file (JPEG, PNG, WEBP, or GIF).',
            fileTooLargeAlert: 'The image is too large. Please upload a file under 25MB.',
            unknownError: 'An unknown error occurred during analysis. Please try again.',
            requestFailed: 'Analysis request failed',
            elapsed: 'Elapsed',
            unknown: 'Unknown',
            verdictTitles: {
                confirmed_ai_generated: 'Confirmed AI-generated or AI-edited',
                likely_ai_generated: 'Likely AI-generated',
                possibly_ai_generated: 'Possibly AI-generated or re-exported',
                possibly_ai_edited: 'Possibly AI-edited',
                likely_camera_originated: 'Likely camera-originated',
                inconclusive: 'Inconclusive'
            },
            verdictFallback: 'Unknown result'
        },
        zh: {
            pageTitle: 'AI Image Judge | 图像取证工作台',
            switchLanguage: 'English',
            targetFile: '目标文件',
            uploadTitle: '拖拽图片至此处，或 <strong>点击上传</strong>',
            uploadDesc: '支持 JPEG, PNG, WEBP, GIF 格式，最大 25MB',
            clear: '清除',
            analyze: '开始检测',
            forensicReport: '取证报告',
            emptyState: '请在左侧上传图片并点击“开始检测”<br>取证结果将在此处显示',
            loadingState: '正在执行多维度取证分析...',
            errorDefault: '检测失败，请重试。',
            confidence: '综合置信度',
            modelProbability: '模型判定概率',
            modelDisclaimer: '提示：AI 检测模型基于统计学特征，结果仅供参考，不作为绝对判定依据。',
            aiGenerated: 'AI 生成',
            realPhoto: '真实拍摄',
            fileInfo: '文件指纹与基础信息',
            mimeType: 'MIME 类型',
            resolution: '分辨率',
            fileSize: '文件大小',
            modelStatus: '模型状态',
            sha256: 'SHA256 哈希值',
            evidence: '取证分析点',
            warnings: '异常警告',
            metadataDetails: '元数据详情',
            rawReport: '原始检测报告',
            c2paTool: 'C2PA 工具',
            exifTool: 'ExifTool',
            modelReady: '模型已就绪',
            modelOffline: '模型未配置',
            model: '模型',
            uploadImageAlert: '请上传图片文件 (JPEG, PNG, WEBP 或 GIF)',
            fileTooLargeAlert: '图片过大，请上传 25MB 以内的文件',
            unknownError: '检测过程发生未知错误，请重试。',
            requestFailed: '分析请求失败',
            elapsed: '耗时',
            unknown: '未知',
            verdictTitles: {
                confirmed_ai_generated: '基本可确认 AI 生成或 AI 编辑',
                likely_ai_generated: '很可能是 AI 生成',
                possibly_ai_generated: '可能是 AI 生成或平台重导出的图片',
                possibly_ai_edited: '可能经过 AI 编辑',
                likely_camera_originated: '更倾向相机/手机拍摄来源',
                inconclusive: '证据不足，无法判断'
            },
            verdictFallback: '未知结果'
        }
    };

    // Initialization
    applyTranslations();
    checkHealth();

    // Event Listeners - Upload
    dropzone.addEventListener('click', () => fileInput.click());
    
    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
    });

    dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('dragover');
    });

    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    btnClear.addEventListener('click', resetUI);
    
    btnAnalyze.addEventListener('click', () => {
        if (currentFile) {
            analyzeImage(currentFile);
        }
    });

    langToggle.addEventListener('click', () => {
        currentLang = currentLang === 'en' ? 'zh' : 'en';
        localStorage.setItem('ai-image-judge-lang', currentLang);
        applyTranslations();
        if (lastHealth) {
            renderHealth(lastHealth);
        }
        if (lastReport) {
            renderReport(lastReport);
        }
    });

    function t(key) {
        return I18N[currentLang][key] || I18N.en[key] || key;
    }

    function applyTranslations() {
        document.documentElement.lang = currentLang === 'zh' ? 'zh-CN' : 'en';
        document.title = t('pageTitle');
        langToggle.textContent = t('switchLanguage');
        langToggle.setAttribute('aria-label', currentLang === 'en' ? 'Switch to Chinese' : '切换到英文');
        document.querySelectorAll('[data-i18n]').forEach((element) => {
            element.textContent = t(element.dataset.i18n);
        });
        document.querySelectorAll('[data-i18n-html]').forEach((element) => {
            element.innerHTML = t(element.dataset.i18nHtml);
        });
        if (!stateError.classList.contains('hidden') && !lastReport) {
            errorMessage.textContent = t('errorDefault');
        }
    }

    // Core Logic
    async function checkHealth() {
        try {
            const res = await fetch('/api/health');
            const data = await res.json();
            lastHealth = data;
            renderHealth(data);
        } catch (err) {
            console.error("Health check failed:", err);
            updateStatusBadge('status-c2pa', false, t('c2paTool'));
            updateStatusBadge('status-exif', false, t('exifTool'));
            updateStatusBadge('status-model', false, t('model'));
        }
    }

    function renderHealth(data) {
        updateStatusBadge('status-c2pa', Boolean(data.tools?.c2patool), t('c2paTool'));
        updateStatusBadge('status-exif', Boolean(data.tools?.exiftool), t('exifTool'));
        
        const modelBadge = document.getElementById('status-model');
        modelBadge.classList.remove('online', 'offline');
        if (data.model?.exists) {
            modelBadge.classList.add('online');
            modelBadge.title = data.model.checkpoint || '';
            modelBadge.innerHTML = `<span class="dot"></span>${t('modelReady')}`;
        } else {
            modelBadge.classList.add('offline');
            modelBadge.title = data.model?.checkpoint || '';
            modelBadge.innerHTML = `<span class="dot"></span>${t('modelOffline')}`;
        }
    }

    function updateStatusBadge(id, isOk, label) {
        const el = document.getElementById(id);
        el.classList.remove('online', 'offline');
        el.classList.add(isOk ? 'online' : 'offline');
        if (label) {
            el.innerHTML = `<span class="dot"></span>${label}`;
        }
    }

    function handleFileSelect(file) {
        if (!file.type.startsWith('image/')) {
            alert(t('uploadImageAlert'));
            return;
        }
        if (file.size > 25 * 1024 * 1024) {
            alert(t('fileTooLargeAlert'));
            return;
        }
        currentFile = file;
        
        // Populate preview
        imagePreview.src = URL.createObjectURL(file);
        previewFilename.textContent = file.name;
        previewFilesize.textContent = formatBytes(file.size);
        
        // Toggle UI
        dropzone.classList.add('hidden');
        previewSection.classList.remove('hidden');
        
        // Reset right panel if it was showing result
        lastReport = null;
        showState(stateEmpty);
        elapsedTimeLabel.classList.add('hidden');
    }

    function resetUI() {
        currentFile = null;
        lastReport = null;
        fileInput.value = '';
        if (imagePreview.src) {
            URL.revokeObjectURL(imagePreview.src);
            imagePreview.src = '';
        }
        dropzone.classList.remove('hidden');
        previewSection.classList.add('hidden');
        showState(stateEmpty);
        elapsedTimeLabel.classList.add('hidden');
    }

    async function analyzeImage(file) {
        showState(stateLoading);
        elapsedTimeLabel.classList.add('hidden');
        
        const formData = new FormData();
        formData.append('image', file);

        try {
            const res = await fetch('/api/analyze', {
                method: 'POST',
                body: formData
            });

            const data = await res.json();
            if (!res.ok) {
                throw new Error(data.error || `${t('requestFailed')} (HTTP ${res.status})`);
            }
            renderReport(data);
        } catch (err) {
            console.error(err);
            errorMessage.textContent = localizeMessage(err.message) || t('unknownError');
            showState(stateError);
        }
    }

    function renderReport(data) {
        lastReport = data;
        showState(stateResult);
        
        // Elapsed time
        if (data.elapsed_ms) {
            elapsedTimeLabel.textContent = `${t('elapsed')}: ${(data.elapsed_ms / 1000).toFixed(2)}s`;
            elapsedTimeLabel.classList.remove('hidden');
        }

        // Verdict
        const verdictTitle = document.getElementById('verdict-title');
        const verdictLabel = document.getElementById('verdict-label');
        const verdictCard = document.querySelector('.verdict-card');
        
        verdictTitle.textContent = verdictTitleFor(data.verdict);
        verdictLabel.textContent = data.verdict.label || "UNKNOWN";
        document.getElementById('verdict-confidence').textContent = data.verdict.confidence ? `${(data.verdict.confidence * 100).toFixed(1)}%` : 'N/A';

        // Color coding based on label keywords
        const label = data.verdict.label || "";
        if (label === 'confirmed_ai_generated' || label === 'likely_ai_generated' || label === 'possibly_ai_generated' || label === 'possibly_ai_edited') {
            verdictCard.setAttribute('data-type', 'AI');
        } else if (label === 'likely_camera_originated') {
            verdictCard.setAttribute('data-type', 'REAL');
        } else {
            verdictCard.removeAttribute('data-type');
        }

        // Model Probabilities
        const probAi = Number(data.model?.ai_probability || 0);
        const probReal = Number(data.model?.real_probability || 0);
        document.getElementById('bar-ai').style.width = `${Math.max(0, Math.min(100, probAi * 100))}%`;
        document.getElementById('prob-ai-val').textContent = data.model?.status === 'available' ? `${(probAi * 100).toFixed(1)}%` : 'N/A';
        document.getElementById('bar-real').style.width = `${Math.max(0, Math.min(100, probReal * 100))}%`;
        document.getElementById('prob-real-val').textContent = data.model?.status === 'available' ? `${(probReal * 100).toFixed(1)}%` : 'N/A';

        // Basic Info
        document.getElementById('info-type').textContent = (data.image_type || t('unknown')).toUpperCase();
        document.getElementById('info-dim').textContent = data.dimensions ? `${data.dimensions.width} x ${data.dimensions.height}` : t('unknown');
        document.getElementById('info-size').textContent = formatBytes(data.bytes || 0);
        document.getElementById('info-model').textContent = data.model?.status || "unknown";
        document.getElementById('info-sha256').textContent = data.sha256 || "N/A";

        // Evidence
        renderList('list-evidence', data.verdict.evidence, 'evidence-section');
        // Warnings
        renderList('list-warnings', data.verdict.warnings, 'warning-section');

        // JSON Metadata
        document.getElementById('code-metadata').textContent = data.metadata ? JSON.stringify(data.metadata, null, 2) : "{}";
        document.getElementById('code-raw').textContent = JSON.stringify(data, null, 2);
    }

    function renderList(elementId, items, sectionId) {
        const ul = document.getElementById(elementId);
        const section = document.getElementById(sectionId);
        ul.innerHTML = '';
        
        if (items && items.length > 0) {
            section.classList.remove('hidden');
            items.forEach(item => {
                const li = document.createElement('li');
                li.textContent = localizeMessage(item);
                ul.appendChild(li);
            });
        } else {
            section.classList.add('hidden');
        }
    }

    function verdictTitleFor(verdict) {
        const label = verdict?.label || '';
        return I18N[currentLang].verdictTitles[label] || verdict?.title || t('verdictFallback');
    }

    function localizeMessage(message) {
        if (!message || currentLang === 'zh') {
            return message;
        }
        const text = String(message);
        const replacements = [
            [/未检测到可验证的 C2PA 来源凭证/g, 'No verifiable C2PA provenance credential was detected'],
            [/本机未安装 c2patool；无法验证 GPT\/DALL-E 等图片可能携带的 C2PA 签名/g, 'c2patool is not installed, so possible GPT/DALL-E C2PA signatures cannot be verified'],
            [/本机未安装 exiftool；只能读取一部分基础 EXIF\/XMP/g, 'exiftool is not installed, so only limited built-in EXIF/XMP parsing is available'],
            [/元数据或来源凭证中出现 AI 工具痕迹：/g, 'AI-tool traces found in metadata or provenance: '],
            [/检测到相机\/镜头\/拍摄时间等来源字段，倾向真实拍摄来源，但这些字段可被编辑/g, 'Camera, lens, or capture-time fields were found. This leans toward camera origin, but these fields can be edited'],
            [/训练模型输出 AI 分数：/g, 'Trained model AI score: '],
            [/训练模型判断来源更接近：/g, 'The trained model thinks the source is closer to: '],
            [/未找到训练好的 checkpoint；当前只使用元数据和像素启发式证据。/g, 'No trained checkpoint was found; using metadata and pixel heuristics only.'],
            [/未发现基础 EXIF；这常见于 AI 导出、截图、微信转发或社交平台重压缩/g, 'No basic EXIF was found. This is common in AI exports, screenshots, WeChat forwarding, and social-media recompression'],
            [/图片为常见生成模型方图尺寸 ([0-9]+)x([0-9]+)；仅作为弱信号/g, 'The image has a common square generation size $1x$2; this is only a weak signal'],
            [/宽高均为 64 的倍数；这在生成图中常见，但不能单独说明问题/g, 'Both width and height are multiples of 64. This is common in generated images but is not conclusive'],
            [/([A-Z]+) 文件缺少相机来源信息；这更像导出图而不是相机原图/g, '$1 lacks camera-origin metadata, which looks more like an exported image than a camera original'],
            [/像素统计显示高频细节和边缘密度偏低；这可能来自生成图、磨皮、压缩或缩放/g, 'Pixel statistics show low high-frequency detail and edge density; this may come from generation, smoothing, compression, or resizing'],
            [/局部残差分布较均匀；部分生成图会出现这种弱纹理特征/g, 'Local residual distribution is relatively uniform; some generated images show this weak texture pattern'],
            [/频域能量集中在中频且残差波动较低；这是弱可疑信号/g, 'Frequency energy is concentrated in mid frequencies with low residual variation; this is a weak suspicious signal'],
            [/检测到 8x8 网格压缩痕迹；更像经过 JPEG\/社交平台二次处理/g, '8x8 grid compression artifacts were detected, suggesting JPEG or social-platform reprocessing'],
            [/图片带透明通道；这更常见于设计\/导出素材，不常见于相机原图/g, 'The image has an alpha channel, which is more common in design/exported assets than camera originals'],
            [/该结果是证据链分析，不是司法级结论；元数据可以被删除、伪造或在转发时丢失。/g, 'This is evidence-chain analysis, not a forensic-grade conclusion. Metadata can be deleted, forged, or lost during sharing.'],
            [/模型只在 Defactify 数据集分布上训练；豆包、Flux、微信压缩、局部重绘等场景仍可能错判。/g, 'The model was trained on the Defactify distribution only. Doubao, Flux, WeChat compression, local inpainting, and similar cases may still be misclassified.'],
            [/模型分数只是概率证据；建议结合 C2PA、EXIF 和来源链一起判断。/g, 'The model score is probabilistic evidence only. Combine it with C2PA, EXIF, and source-chain evidence.'],
            [/这是弱启发式判断，不是确认结论；微信转发、截图、压缩和修图都可能产生类似信号。/g, 'This is a weak heuristic result, not confirmation. WeChat forwarding, screenshots, compression, and editing can produce similar signals.'],
            [/当前没有训练好的分类模型；弱信号不能作为 AI 生成结论。/g, 'No trained classifier is available; weak signals alone cannot prove AI generation.'],
            [/未接入训练好的图像分类器时，不能仅凭缺失元数据判断是否为 AI 图片。/g, 'Without a trained image classifier, missing metadata alone cannot determine whether an image is AI-generated.'],
            [/建议安装 c2patool 后重新检测，以验证签名和 manifest 内容。/g, 'Install c2patool and run the check again to verify signatures and manifest content.'],
            [/暂不支持该文件类型；请上传 JPEG、PNG、GIF 或 WebP 图片。/g, 'Unsupported file type. Please upload a JPEG, PNG, GIF, or WebP image.'],
            [/图片过大；当前限制为 25MB。/g, 'The image is too large. The current limit is 25MB.'],
            [/请使用字段名 image 上传图片文件。/g, 'Please upload the image using the form field named "image".'],
            [/上传文件为空。/g, 'The uploaded file is empty.'],
            [/没有收到上传内容。/g, 'No upload content was received.']
        ];
        return replacements.reduce((value, [pattern, replacement]) => value.replace(pattern, replacement), text);
    }

    function showState(element) {
        stateEmpty.classList.add('hidden');
        stateLoading.classList.add('hidden');
        stateError.classList.add('hidden');
        stateResult.classList.add('hidden');
        element.classList.remove('hidden');
    }

    function formatBytes(bytes, decimals = 2) {
        if (!+bytes) return '0 Bytes';
        const k = 1024;
        const dm = decimals < 0 ? 0 : decimals;
        const sizes =['Bytes', 'KB', 'MB', 'GB', 'TB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return `${parseFloat((bytes / Math.pow(k, i)).toFixed(dm))} ${sizes[i]}`;
    }
});
