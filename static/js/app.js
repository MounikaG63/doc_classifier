// Setup event listeners on page load
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    loadDbStats();
});

async function loadDbStats() {
    try {
        const response = await fetch('/db-stats/');
        const data = await response.json();
        
        // Update stats display if element exists
        const statsDiv = document.getElementById('db-stats');
        if (statsDiv) {
            let html = `
                <div style="display: flex; justify-content: space-between; align-items: center; cursor: pointer;" onclick="toggleDbStats()">
                    <p style="margin: 0;"><strong>Total Documents:</strong> ${data.total}</p>
                    <span id="db-stats-arrow" style="font-size: 1.2em; transition: transform 0.3s;">▼</span>
                </div>
            `;
            if (data.total > 0 && data.by_type) {
                html += '<div id="db-stats-list" style="display: none; margin-top: 10px; max-height: 300px; overflow-y: auto;">';
                html += '<p style="margin-bottom: 5px;"><strong>By Type:</strong></p>';
                html += '<ul style="margin: 5px 0; padding-left: 20px; font-size: 0.9em;">';
                for (const [type, count] of Object.entries(data.by_type)) {
                    html += `<li style="margin: 3px 0;">${type}: <strong>${count}</strong></li>`;
                }
                html += '</ul></div>';
            }
            statsDiv.innerHTML = html;
        }
    } catch (error) {
        console.error('Error loading DB stats:', error);
    }
}

function toggleDbStats() {
    const list = document.getElementById('db-stats-list');
    const arrow = document.getElementById('db-stats-arrow');
    
    if (list && arrow) {
        if (list.style.display === 'none') {
            list.style.display = 'block';
            arrow.style.transform = 'rotate(180deg)';
        } else {
            list.style.display = 'none';
            arrow.style.transform = 'rotate(0deg)';
        }
    }
}

function setupEventListeners() {
    const fileInput = document.getElementById('file-input');
    const uploadArea = document.getElementById('upload-area');
    const rebuildBtn = document.getElementById('rebuild-btn');
    const addFilesBtn = document.getElementById('add-files-btn');

    // File upload
    fileInput.addEventListener('change', handleFileSelect);
    
    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.style.background = '#e0e7ff';
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.style.background = '#f8f9ff';
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.style.background = '#f8f9ff';
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            fileInput.files = files;
            handleFileSelect();
        }
    });

    // Database actions
    rebuildBtn.addEventListener('click', rebuildDatabase);
    addFilesBtn.addEventListener('click', addNewFiles);
}



function handleFileSelect() {
    const fileInput = document.getElementById('file-input');
    const files = Array.from(fileInput.files);
    
    if (files.length === 0) return;
    
    // Show preview
    const preview = document.getElementById('file-preview');
    let previewHTML = `<p><strong>${files.length} file(s) selected:</strong></p><div style="display: flex; flex-wrap: wrap; gap: 10px;">`;
    
    files.forEach(file => {
        const icon = file.name.toLowerCase().endsWith('.zip') ? '📦' : '📄';
        previewHTML += `<div style="padding: 8px 12px; background: #f0f2ff; border-radius: 5px; font-size: 0.9em;">${icon} ${file.name}</div>`;
    });
    
    previewHTML += '</div>';
    preview.innerHTML = previewHTML;
    
    // Check if any file is a ZIP
    const zipFile = files.find(f => f.name.toLowerCase().endsWith('.zip'));
    
    if (zipFile) {
        // Process ZIP file
        classifyZipFile(zipFile);
    } else if (files.length === 1) {
        classifyDocument(files[0]);
    } else {
        classifyMultipleDocuments(files);
    }
}

async function classifyZipFile(file) {
    const resultsSection = document.getElementById('results-section');
    const loading = document.getElementById('loading');
    const resultsContent = document.getElementById('results-content');
    
    resultsSection.style.display = 'block';
    loading.style.display = 'block';
    resultsContent.style.display = 'none';
    
    loading.innerHTML = `
        <div class="spinner"></div>
        <p>Extracting and processing ZIP file...</p>
        <p style="font-size: 0.9em; color: #666;">📦 ${file.name}</p>
    `;
    
    const ocrEngine = document.querySelector('input[name="classify_ocr_engine"]:checked').value;
    
    const formData = new FormData();
    formData.append('file', file);
    formData.append('ocr_engine', ocrEngine);
    
    try {
        const response = await fetch('/classify-zip/', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        loading.style.display = 'none';
        resultsContent.style.display = 'block';
        
        if (data.error) {
            document.getElementById('results-container').innerHTML = `<p class="error">❌ ${data.error}</p>`;
            return;
        }
        
        // Convert ZIP results to the format expected by displayMultipleResults
        const allResults = data.results.map(r => ({
            filename: r.filename,
            result: r
        }));
        
        displayMultipleResults(allResults, ocrEngine);
        
    } catch (error) {
        loading.style.display = 'none';
        resultsContent.innerHTML = `<p class="error">Error: ${error.message}</p>`;
        resultsContent.style.display = 'block';
    }
}

async function classifyDocument(file) {
    const resultsSection = document.getElementById('results-section');
    const loading = document.getElementById('loading');
    const resultsContent = document.getElementById('results-content');
    
    resultsSection.style.display = 'block';
    loading.style.display = 'block';
    resultsContent.style.display = 'none';
    
    loading.innerHTML = `
        <div class="spinner"></div>
        <p>Processing document...</p>
    `;
    
    // Get selected OCR engine
    const ocrEngine = document.querySelector('input[name="classify_ocr_engine"]:checked').value;
    
    const formData = new FormData();
    formData.append('file', file);
    formData.append('ocr_engine', ocrEngine);
    
    try {
        const response = await fetch('/classify/', {
            method: 'POST',
            body: formData
        });
        
        const results = await response.json();
        
        loading.style.display = 'none';
        resultsContent.style.display = 'block';
        
        displayResults(results);
    } catch (error) {
        loading.style.display = 'none';
        resultsContent.innerHTML = `<p class="error">Error: ${error.message}</p>`;
        resultsContent.style.display = 'block';
    }
}

async function classifyMultipleDocuments(files) {
    const resultsSection = document.getElementById('results-section');
    const loading = document.getElementById('loading');
    const resultsContent = document.getElementById('results-content');
    
    console.log('Starting classification of', files.length, 'files');
    
    resultsSection.style.display = 'block';
    loading.style.display = 'block';
    resultsContent.style.display = 'none';
    
    const ocrEngineRadio = document.querySelector('input[name="classify_ocr_engine"]:checked');
    const ocrEngine = ocrEngineRadio ? ocrEngineRadio.value : 'tesseract';
    console.log('Selected OCR engine:', ocrEngine);
    const allResults = [];
    
    for (let i = 0; i < files.length; i++) {
        const file = files[i];
        
        loading.innerHTML = `
            <div class="spinner"></div>
            <p>Processing ${i + 1} of ${files.length} documents...</p>
            <p style="font-size: 0.9em; color: #666;">${file.name}</p>
            <div style="margin-top: 10px; background: #f0f2ff; padding: 10px; border-radius: 5px;">
                <div style="width: ${((i) / files.length) * 100}%; height: 4px; background: #667eea; border-radius: 2px;"></div>
            </div>
        `;
        
        const formData = new FormData();
        formData.append('file', file);
        formData.append('ocr_engine', ocrEngine);
        
        try {
            console.log(`[${i+1}/${files.length}] Classifying:`, file.name);
            const response = await fetch('/classify/', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            
            const result = await response.json();
            console.log(`[${i+1}/${files.length}] Result:`, result);
            
            // Ensure result has required fields
            if (!result.ocr_engine) {
                result.ocr_engine = ocrEngine;
            }
            
            console.log(`[${i+1}/${files.length}] Adding to results:`, {filename: file.name, result: result});
            allResults.push({
                filename: file.name,
                result: result
            });
            console.log(`[${i+1}/${files.length}] allResults length:`, allResults.length);
        } catch (error) {
            console.error(`[${i+1}/${files.length}] Error:`, error);
            allResults.push({
                filename: file.name,
                result: {
                    document_type: 'unknown',
                    reason: `Error: ${error.message}`,
                    ocr_engine: ocrEngine
                }
            });
        }
        
        // Small delay to prevent overwhelming the server
        if (i < files.length - 1) {
            await new Promise(resolve => setTimeout(resolve, 100));
        }
    }
    
    console.log('All results:', allResults);
    
    loading.style.display = 'none';
    resultsContent.style.display = 'block';
    
    displayMultipleResults(allResults, ocrEngine);
}

function displayMultipleResults(allResults, ocrEngine) {
    console.log('Displaying results for', allResults.length, 'documents');
    console.log('OCR Engine received:', ocrEngine);
    console.log('All results:', JSON.stringify(allResults, null, 2));
    
    const container = document.getElementById('results-container');
    
    if (!container) {
        console.error('Results container not found!');
        return;
    }
    
    // Fallback for ocrEngine
    if (!ocrEngine && allResults.length > 0 && allResults[0].result) {
        ocrEngine = allResults[0].result.ocr_engine || 'Unknown';
    }
    
    let html = `
        <div class="multi-results-header">
            <h3>📊 Batch Classification Results (${ocrEngine})</h3>
            <p>${allResults.length} documents processed</p>
        </div>
        <div class="results-table-container">
            <table class="results-table">
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Filename</th>
                        <th>Document Type</th>
                        <th>Similarity</th>
                        <th>Time</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
    `;
    
    allResults.forEach((item, index) => {
        const result = item.result || {};
        const status = result.document_type && result.document_type !== 'unknown' ? '✅' : '❌';
        const docType = result.document_type && result.document_type !== 'unknown' ? result.document_type : (result.reason || 'Failed');
        const similarity = result.similarity_score ? `${result.similarity_score}%` : '-';
        const time = result.processing_time ? `${result.processing_time}s` : '-';
        
        html += `
            <tr>
                <td>${index + 1}</td>
                <td style="max-width: 200px; overflow: hidden; text-overflow: ellipsis;" title="${item.filename}">${item.filename}</td>
                <td><strong>${docType}</strong></td>
                <td>${similarity}</td>
                <td>${time}</td>
                <td>${status}</td>
            </tr>
        `;
    });
    
    html += `
                </tbody>
            </table>
        </div>
        <div class="export-section">
            <button onclick="exportResults()" class="btn btn-primary">📥 Export as CSV</button>
        </div>
    `;
    
    container.innerHTML = html;
    console.log('Results displayed successfully');
    
    // Store results for export
    window.batchResults = allResults;
}

function exportResults() {
    if (!window.batchResults) return;
    
    let csv = 'Filename,Document Type,Similarity Score,Processing Time,Status\n';
    
    window.batchResults.forEach(item => {
        const result = item.result;
        const status = result.document_type !== 'unknown' ? 'Success' : 'Failed';
        const docType = result.document_type || 'unknown';
        const similarity = result.similarity_score || 0;
        const time = result.processing_time || 0;
        
        csv += `"${item.filename}","${docType}",${similarity},${time},${status}\n`;
    });
    
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `classification_results_${new Date().getTime()}.csv`;
    a.click();
    window.URL.revokeObjectURL(url);
}

// Duplicate function removed - using the one above

function displayResults(results) {
    const container = document.getElementById('results-container');
    
    let html = '<div class="result-card">';
    html += `<h3>📄 ${results.ocr_engine || 'OCR'} Results</h3>`;
    html += '<div class="classification-results">';
    html += formatOcrResult(results);
    html += '</div></div>';
    
    container.innerHTML = html;
}

function formatOcrResult(result) {
    if (result.document_type === 'unknown') {
        return `<p class="error">❌ Classification Failed</p>
                <p>Reason: ${result.reason || 'Unknown'}</p>`;
    }
    
    let html = `<div class="success-badge">✅ ${result.document_type}</div>`;
    
    // Check if Gemini was used (different display)
    if (result.gemini_used) {
        html += `
            <div class="gemini-badge" style="background: #e8f5e9; padding: 15px; border-radius: 8px; margin: 15px 0; border-left: 4px solid #4caf50;">
                <div style="display: flex; align-items: center; margin-bottom: 10px;">
                    <span style="font-size: 1.5em; margin-right: 10px;">🤖</span>
                    <strong style="font-size: 1.1em;">Classified by Gemini AI</strong>
                </div>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr>
                        <td style="padding: 8px 0; border-bottom: 1px solid #c8e6c9; font-weight: 600; width: 40%;">Document Type:</td>
                        <td style="padding: 8px 0; border-bottom: 1px solid #c8e6c9;">${result.document_type || '-'}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px 0; border-bottom: 1px solid #c8e6c9; font-weight: 600;">Category:</td>
                        <td style="padding: 8px 0; border-bottom: 1px solid #c8e6c9;">${result.category || '-'}</td>
                    </tr>
                    ${result.sub_category ? `
                    <tr>
                        <td style="padding: 8px 0; border-bottom: 1px solid #c8e6c9; font-weight: 600;">Sub-category:</td>
                        <td style="padding: 8px 0; border-bottom: 1px solid #c8e6c9;">${result.sub_category}</td>
                    </tr>
                    ` : ''}
                    <tr>
                        <td style="padding: 8px 0; border-bottom: 1px solid #c8e6c9; font-weight: 600;">Confidence:</td>
                        <td style="padding: 8px 0; border-bottom: 1px solid #c8e6c9;">
                            <span style="background: ${result.confidence === 'high' ? '#4caf50' : result.confidence === 'medium' ? '#ff9800' : '#f44336'}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.85em;">${result.confidence || 'unknown'}</span>
                        </td>
                    </tr>
                    <tr>
                        <td style="padding: 8px 0; font-weight: 600;">Processing Time:</td>
                        <td style="padding: 8px 0;">${result.processing_time || 0}s</td>
                    </tr>
                </table>
            </div>
        `;
        if (result.reason) {
            html += `<p style="color: #666; font-size: 0.9em; margin-top: 10px; background: #f5f5f5; padding: 10px; border-radius: 5px;">ℹ️ ${result.reason}</p>`;
        }
    } else {
        // Standard OCR result display with hybrid classification
        html += `
            <div class="metrics-row">
                <div class="metric-box">
                    <div class="metric-label">Confidence</div>
                    <div class="metric-value">${result.confidence || 0}%</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Method</div>
                    <div class="metric-value" style="font-size: 0.9em; text-transform: capitalize;">${result.classification_method || 'vector'}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Time</div>
                    <div class="metric-value">${result.processing_time || 0}s</div>
                </div>
            </div>
        `;
        
        // Show classification details
        if (result.details) {
            html += `<div style="background: #f0f4ff; padding: 12px; border-radius: 6px; margin: 15px 0; border-left: 4px solid #667eea;">
                        <strong>Classification Details:</strong> ${result.details}
                    </div>`;
        }
        
        // Top 3 Matches with Vector and Keyword Scores
        if (result.top_3_matches && result.top_3_matches.length > 0) {
            html += `<div style="margin-top: 20px;">
                        <strong style="display: block; margin-bottom: 12px; font-size: 1.05em;">📊 Top 3 Matches:</strong>
                        <div style="display: grid; gap: 12px;">`;
            
            result.top_3_matches.forEach((match, i) => {
                const isSelected = match.type === result.document_type;
                const methodBg = result.classification_method === 'keyword' ? '#e8f5e9' : '#e3f2fd';
                const methodColor = result.classification_method === 'keyword' ? '#2e7d32' : '#1565c0';
                
                html += `
                    <div style="background: white; border: 1px solid #e0e0e0; border-radius: 8px; padding: 12px; ${isSelected ? 'border: 2px solid #667eea; background: #f8f9ff;' : ''}">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                            <span style="font-weight: 600; font-size: 1.05em;">#${match.rank} ${match.type}</span>
                            ${isSelected ? '<span style="background: #667eea; color: white; padding: 4px 10px; border-radius: 20px; font-size: 0.8em; font-weight: 600;">SELECTED</span>' : ''}
                        </div>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                            <div style="background: #e3f2fd; padding: 10px; border-radius: 6px;">
                                <div style="font-size: 0.85em; color: #666; margin-bottom: 4px;">Vector Score</div>
                                <div style="font-size: 1.3em; font-weight: 700; color: #1565c0;">${match.vector_score}%</div>
                            </div>
                            <div style="background: #f3e5f5; padding: 10px; border-radius: 6px;">
                                <div style="font-size: 0.85em; color: #666; margin-bottom: 4px;">Keyword Score</div>
                                <div style="font-size: 1.3em; font-weight: 700; color: #6a1b9a;">${match.keyword_score}%</div>
                            </div>
                        </div>
                        <div style="background: #f5f5f5; padding: 8px; border-radius: 6px; margin-top: 8px; text-align: center;">
                            <div style="font-size: 0.85em; color: #666;">Combined Score</div>
                            <div style="font-size: 1.2em; font-weight: 700; color: #333;">${match.combined_score}%</div>
                        </div>
                    </div>
                `;
            });
            
            html += `</div></div>`;
        }
        
        if (result.ocr_text) {
            html += '<div class="ocr-section" style="margin-top: 20px;"><strong>OCR Text Preview:</strong>';
            html += `<div class="ocr-text">${result.ocr_text}</div></div>`;
        }
    }
    
    return html;
}



function displayWinner(results) {
    const winnerDiv = document.getElementById('winner-content');
    
    const easy = results.easyocr;
    const paddle = results.paddleocr;
    
    let easyScore = 0;
    let paddleScore = 0;
    let criteria = [];
    
    // Similarity score (50 points)
    const easySim = easy.similarity_score || 0;
    const paddleSim = paddle.similarity_score || 0;
    
    if (easySim > paddleSim) {
        easyScore += 50;
        criteria.push(`✅ EasyOCR: Higher similarity (${easySim}% vs ${paddleSim}%)`);
    } else if (paddleSim > easySim) {
        paddleScore += 50;
        criteria.push(`✅ PaddleOCR: Higher similarity (${paddleSim}% vs ${easySim}%)`);
    } else {
        criteria.push(`🤝 Similarity tied at ${easySim}%`);
    }
    
    // Document type (30 points)
    if (easy.document_type !== 'unknown' && paddle.document_type === 'unknown') {
        easyScore += 30;
        criteria.push(`✅ EasyOCR: Identified document type (${easy.document_type})`);
    } else if (paddle.document_type !== 'unknown' && easy.document_type === 'unknown') {
        paddleScore += 30;
        criteria.push(`✅ PaddleOCR: Identified document type (${paddle.document_type})`);
    } else if (easy.document_type === paddle.document_type && easy.document_type !== 'unknown') {
        criteria.push(`🤝 Both identified as: ${easy.document_type}`);
    }
    
    // Text length (10 points)
    const easyLen = easy.text_length || 0;
    const paddleLen = paddle.text_length || 0;
    
    if (easyLen > paddleLen * 1.1) {
        easyScore += 10;
        criteria.push(`✅ EasyOCR: Extracted more text (${easyLen} vs ${paddleLen} chars)`);
    } else if (paddleLen > easyLen * 1.1) {
        paddleScore += 10;
        criteria.push(`✅ PaddleOCR: Extracted more text (${paddleLen} vs ${easyLen} chars)`);
    }
    
    // Speed (10 points)
    const easyTime = easy.processing_time || 0;
    const paddleTime = paddle.processing_time || 0;
    
    if (easyTime < paddleTime * 0.9) {
        easyScore += 10;
        criteria.push(`⚡ EasyOCR: Faster processing (${easyTime}s vs ${paddleTime}s)`);
    } else if (paddleTime < easyTime * 0.9) {
        paddleScore += 10;
        criteria.push(`⚡ PaddleOCR: Faster processing (${paddleTime}s vs ${easyTime}s)`);
    }
    
    // Display criteria
    let html = '<div class="criteria-list">';
    criteria.forEach(c => {
        html += `<div class="criteria-item">${c}</div>`;
    });
    html += '</div>';
    
    // Display winner
    html += '<div style="margin-top: 30px; text-align: center;">';
    if (easyScore > paddleScore) {
        html += `<div class="winner-badge">🏆 Winner: EasyOCR (${easyScore} vs ${paddleScore})</div>`;
    } else if (paddleScore > easyScore) {
        html += `<div class="winner-badge">🏆 Winner: PaddleOCR (${paddleScore} vs ${easyScore})</div>`;
    } else {
        html += `<div class="winner-badge">🤝 Tie! (Both scored ${easyScore})</div>`;
    }
    html += '</div>';
    
    winnerDiv.innerHTML = html;
}

async function rebuildDatabase() {
    const btn = document.getElementById('rebuild-btn');
    const ocrEngine = document.querySelector('input[name="db_ocr_engine"]:checked').value;
    
    btn.disabled = true;
    btn.textContent = '⏳ Rebuilding...';
    
    try {
        const response = await fetch('/rebuild-db/', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ocr_engine: ocrEngine})
        });
        
        const data = await response.json();
        
        if (data.success) {
            alert(`✅ Database rebuilt with ${ocrEngine.toUpperCase()}!\n${data.added}/${data.total} documents indexed.`);
            loadDbStats();
        }
    } catch (error) {
        alert(`❌ Error: ${error.message}`);
    } finally {
        btn.disabled = false;
        btn.textContent = '🔄 Full Rebuild';
    }
}

async function addNewFiles() {
    const btn = document.getElementById('add-files-btn');
    const ocrEngine = document.querySelector('input[name="db_ocr_engine"]:checked').value;
    
    btn.disabled = true;
    btn.textContent = '⏳ Adding...';
    
    try {
        const response = await fetch('/add-files/', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ocr_engine: ocrEngine})
        });
        
        const data = await response.json();
        
        if (data.success) {
            if (data.added === 0) {
                alert('ℹ️ No new files found.');
            } else {
                alert(`✨ Added ${data.added} new documents with ${ocrEngine.toUpperCase()}.`);
                loadDbStats();
            }
        }
    } catch (error) {
        alert(`❌ Error: ${error.message}`);
    } finally {
        btn.disabled = false;
        btn.textContent = '➕ Add New Files';
    }
}
