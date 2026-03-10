// Multi-file display function
function displayMultipleResults(allResults) {
    let html = '<div class="multi-results">';
    
    // Summary
    html += '<div class="summary-section" style="background: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">';
    html += `<h3>📊 Processing Summary</h3>`;
    html += `<p><strong>Total Files:</strong> ${allResults.length}</p>`;
    
    const successful = allResults.filter(r => !r.error).length;
    const failed = allResults.length - successful;
    
    html += `<p><strong>Successful:</strong> ${successful}</p>`;
    if (failed > 0) {
        html += `<p style="color: #ef4444;"><strong>Failed:</strong> ${failed}</p>`;
    }
    html += '</div>';
    
    // Individual results
    allResults.forEach((item, index) => {
        html += `<div class="document-result" style="background: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; border-left: 5px solid #667eea; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">`;
        html += `<h3>📄 Document ${index + 1}: ${item.filename}</h3>`;
        
        if (item.error) {
            html += `<p class="error">❌ Error: ${item.error}</p>`;
        } else {
            const results = item.results;
            
            // Quick summary
            html += '<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0;">';
            
            // EasyOCR summary
            html += '<div style="padding: 15px; background: #f0f2ff; border-radius: 8px;">';
            html += '<h4>🟦 EasyOCR</h4>';
            if (results.easyocr.document_type !== 'unknown') {
                html += `<p><strong>Type:</strong> ${results.easyocr.document_type}</p>`;
                html += `<p><strong>Confidence:</strong> ${results.easyocr.similarity_score}%</p>`;
                html += `<p><strong>Time:</strong> ${results.easyocr.processing_time}s</p>`;
            } else {
                html += `<p class="error">❌ Failed: ${results.easyocr.reason || 'Unknown'}</p>`;
            }
            html += '</div>';
            
            // PaddleOCR summary
            html += '<div style="padding: 15px; background: #f0fff4; border-radius: 8px;">';
            html += '<h4>🟩 PaddleOCR</h4>';
            if (results.paddleocr.document_type !== 'unknown') {
                html += `<p><strong>Type:</strong> ${results.paddleocr.document_type}</p>`;
                html += `<p><strong>Confidence:</strong> ${results.paddleocr.similarity_score}%</p>`;
                html += `<p><strong>Time:</strong> ${results.paddleocr.processing_time}s</p>`;
            } else {
                html += `<p class="error">❌ Failed: ${results.paddleocr.reason || 'Unknown'}</p>`;
            }
            html += '</div>';
            
            html += '</div>';
            
            // Winner
            const easyScore = calculateScore(results.easyocr);
            const paddleScore = calculateScore(results.paddleocr);
            
            if (easyScore > paddleScore) {
                html += `<p style="text-align: center; font-size: 1.2em; color: #667eea; font-weight: 600;">🏆 Winner: EasyOCR (${easyScore} vs ${paddleScore})</p>`;
            } else if (paddleScore > easyScore) {
                html += `<p style="text-align: center; font-size: 1.2em; color: #10b981; font-weight: 600;">🏆 Winner: PaddleOCR (${paddleScore} vs ${easyScore})</p>`;
            } else {
                html += `<p style="text-align: center; font-size: 1.2em; font-weight: 600;">🤝 Tie (${easyScore} points each)</p>`;
            }
        }
        
        html += '</div>';
    });
    
    html += '</div>';
    
    document.getElementById('results-content').innerHTML = html;
}

function calculateScore(result) {
    let score = 0;
    
    if (result.document_type !== 'unknown') {
        score += 50; // Document type identified
        if (result.similarity_score) {
            score += Math.min(result.similarity_score / 2, 50); // Up to 50 points for similarity
        }
    }
    
    return Math.round(score);
}
