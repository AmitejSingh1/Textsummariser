const API_URL = 'http://localhost:5000/api';

// Wait for DOM to be fully loaded
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
});

function initializeApp() {
    const summarizeBtn = document.getElementById('summarizeBtn');
    if (!summarizeBtn) {
        console.error('Summarize button not found!');
        return;
    }

    summarizeBtn.addEventListener('click', handleSummarize);
    
    // Allow Enter key to trigger summarization (Ctrl+Enter)
    const textInput = document.getElementById('textInput');
    if (textInput) {
        textInput.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === 'Enter') {
                handleSummarize();
            }
        });
    }
}

async function handleSummarize() {
    const text = document.getElementById('textInput').value.trim();
    const numSentences = parseInt(document.getElementById('numSentences').value) || 3;
    const summarizeBtn = document.getElementById('summarizeBtn');

    if (!text) {
        showError('Please enter some text to summarize.');
        return;
    }

    // Disable button and show loading
    summarizeBtn.disabled = true;
    summarizeBtn.textContent = 'Processing...';
    
    // Show loading, hide results and error
    document.getElementById('loading').classList.remove('hidden');
    document.getElementById('results').classList.add('hidden');
    document.getElementById('error').classList.add('hidden');

    try {
        const response = await fetch(`${API_URL}/summarize`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: text,
                num_sentences: numSentences
            })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Failed to summarize text');
        }

        const data = await response.json();
        displayResults(data);

    } catch (error) {
        console.error('Summarization error:', error);
        let errorMsg = error.message;
        if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
            errorMsg = `Cannot connect to server. Make sure the Flask server is running at ${API_URL}`;
        }
        showError(`Error: ${errorMsg}`);
    } finally {
        const loadingEl = document.getElementById('loading');
        if (loadingEl) loadingEl.classList.add('hidden');
        
        const btn = document.getElementById('summarizeBtn');
        if (btn) {
            btn.disabled = false;
            btn.textContent = '✨ Summarize';
        }
    }
}

function displayResults(data) {
    const resultsContainer = document.getElementById('resultsContainer');
    const originalLength = document.getElementById('originalLength');
    
    originalLength.textContent = data.original_length;
    resultsContainer.innerHTML = '';

    const results = data.results;
    const methods = Object.keys(results);
    
    // Find the best score
    let bestScore = -1;
    let bestMethod = null;
    methods.forEach(method => {
        if (results[method].overall_score > bestScore) {
            bestScore = results[method].overall_score;
            bestMethod = method;
        }
    });

    methods.forEach(method => {
        const result = results[method];
        const isBest = method === bestMethod;
        
        const card = createResultCard(method, result, isBest);
        resultsContainer.appendChild(card);
    });

    document.getElementById('results').classList.remove('hidden');
}

function createResultCard(method, result, isBest) {
    const card = document.createElement('div');
    card.className = `result-card ${isBest ? 'best' : ''}`;

    const methodName = result.method || method.toUpperCase();
    const methodType = result.type || 'unknown';

    card.innerHTML = `
        <div class="card-header">
            <div>
                <span class="method-name">${methodName}</span>
                <span class="method-type ${methodType}">${methodType}</span>
                ${isBest ? '<span class="best-badge">🏆 Best</span>' : ''}
            </div>
            <div class="score-badge">${result.overall_score.toFixed(1)}</div>
        </div>
        
        <div class="summary-text">${escapeHtml(result.summary)}</div>
        
        <div class="metrics">
            <div class="metric">
                <div class="metric-label">ROUGE-1 F1</div>
                <div class="metric-value">${(result.rouge_scores.rouge1.f1 * 100).toFixed(1)}%</div>
            </div>
            <div class="metric">
                <div class="metric-label">ROUGE-2 F1</div>
                <div class="metric-value">${(result.rouge_scores.rouge2.f1 * 100).toFixed(1)}%</div>
            </div>
            <div class="metric">
                <div class="metric-label">ROUGE-L F1</div>
                <div class="metric-value">${(result.rouge_scores.rougeL.f1 * 100).toFixed(1)}%</div>
            </div>
        </div>
        
        <div class="compression-info">
            Compression Ratio: ${(result.compression_ratio * 100).toFixed(1)}% | 
            Summary Length: ${result.summary.split(' ').length} words
        </div>
    `;

    return card;
}

function showError(message) {
    const errorDiv = document.getElementById('error');
    errorDiv.textContent = message;
    errorDiv.classList.remove('hidden');
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}


