/**
 * PhishGuard Chrome Extension - Popup Script
 * =========================================
 *
 * Handles popup UI interactions and communicates with background service worker.
 */

// ============================================================================
// STATE
// ============================================================================

let currentAnalysis = null;
let isAnalyzing = false;

// ============================================================================
// CONFIGURATION
// ============================================================================

const CONFIG = {
    riskLevels: {
        safe: { color: '#4CAF50', icon: '✓', label: 'Safe', class: 'safe' },
        low: { color: '#8BC34A', icon: '✓', label: 'Low Risk', class: 'low' },
        medium: { color: '#FF9800', icon: '⚠', label: 'Medium', class: 'medium' },
        high: { color: '#F44336', icon: '⚠', label: 'High Risk', class: 'high' },
        critical: { color: '#B71C1C', icon: '🚨', label: 'CRITICAL!', class: 'critical' }
    }
};

// ============================================================================
// DOM ELEMENTS
// ============================================================================

const elements = {
    statusIndicator: document.getElementById('statusIndicator'),
    currentUrl: document.getElementById('currentUrl'),
    resultCard: document.getElementById('resultCard'),
    riskBadge: document.getElementById('riskBadge'),
    riskIcon: document.getElementById('riskIcon'),
    riskLabel: document.getElementById('riskLabel'),
    scoreCircle: document.getElementById('scoreCircle'),
    scoreValue: document.getElementById('scoreValue'),
    resultDetails: document.getElementById('resultDetails'),
    threatsSection: document.getElementById('threatsSection'),
    threatList: document.getElementById('threatList'),
    ctiSection: document.getElementById('ctiSection'),
    ctiList: document.getElementById('ctiList'),
    mlConfidence: document.getElementById('mlConfidence'),
    ctiCount: document.getElementById('ctiCount'),
    processingTime: document.getElementById('processingTime'),
    analyzeBtn: document.getElementById('analyzeBtn'),
    historyBtn: document.getElementById('historyBtn'),
    historyPanel: document.getElementById('historyPanel'),
    historyList: document.getElementById('historyList'),
    closeHistory: document.getElementById('closeHistory'),
    loadingOverlay: document.getElementById('loadingOverlay'),
    openDashboard: document.getElementById('openDashboard')
};

// ============================================================================
// INITIALIZATION
// ============================================================================

document.addEventListener('DOMContentLoaded', async () => {
    console.log('PhishGuard: Popup initializing');

    // Set up event listeners
    setupEventListeners();

    // Get current tab URL and load analysis
    await loadCurrentTabAnalysis();

    // Listen for analysis results from background
    chrome.runtime.onMessage.addListener(handleBackgroundMessage);
});

function setupEventListeners() {
    // Analyze button
    elements.analyzeBtn.addEventListener('click', async () => {
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
        if (tab && tab.url && tab.url.startsWith('http')) {
            await triggerAnalysis(tab.url);
        }
    });

    // History button
    elements.historyBtn.addEventListener('click', async () => {
        await showHistory();
    });

    // Close history button
    elements.closeHistory.addEventListener('click', () => {
        elements.historyPanel.style.display = 'none';
    });

    // Open dashboard link
    elements.openDashboard.addEventListener('click', (e) => {
        e.preventDefault();
        chrome.tabs.create({ url: 'http://localhost:5173' });
    });
}

// ============================================================================
// LOAD ANALYSIS
// ============================================================================

async function loadCurrentTabAnalysis() {
    try {
        // Get current tab
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

        if (!tab || !tab.url || !tab.url.startsWith('http')) {
            showNonHttpPage();
            return;
        }

        // Display URL
        elements.currentUrl.textContent = truncateUrl(tab.url);
        elements.currentUrl.title = tab.url;

        // Check status
        const status = await sendMessage({ type: 'GET_STATUS' });

        if (!status.apiAvailable) {
            updateStatusIndicator('error', 'API Offline');
            await loadCachedOrFallback(tab.url);
            return;
        }

        updateStatusIndicator('online', 'Connected');

        // Get existing analysis or trigger new one
        const existingAnalysis = await sendMessage({
            type: 'GET_ANALYSIS',
            url: tab.url
        });

        if (existingAnalysis && existingAnalysis.success) {
            displayAnalysis(existingAnalysis.data);
        } else {
            // Trigger new analysis
            await triggerAnalysis(tab.url);
        }

    } catch (error) {
        console.error('PhishGuard: Failed to load analysis', error);
        showError('Failed to connect to PhishGuard');
    }
}

async function loadCachedOrFallback(url) {
    showDefaultState();
}

// ============================================================================
// ANALYSIS
// ============================================================================

async function triggerAnalysis(url) {
    if (isAnalyzing) return;

    isAnalyzing = true;
    showLoading(true);
    updateStatusIndicator('analyzing', 'Analyzing...');

    try {
        const response = await sendMessage({
            type: 'ANALYZE_URL',
            url: url
        });

        if (response.success) {
            displayAnalysis(response.data);
            updateStatusIndicator('online', 'Connected');
        } else {
            throw new Error(response.error || 'Analysis failed');
        }

    } catch (error) {
        console.error('PhishGuard: Analysis failed', error);
        showError('Analysis failed: ' + error.message);
        updateStatusIndicator('error', 'Connection Error');
    } finally {
        isAnalyzing = false;
        showLoading(false);
    }
}

// ============================================================================
// DISPLAY
// ============================================================================

function displayAnalysis(result) {
    currentAnalysis = result;

    console.log('PhishGuard: Displaying analysis', result);

    // Determine risk level
    const riskLevel = getRiskLevel(result.risk_score);
    const riskConfig = CONFIG.riskLevels[riskLevel];

    // Update risk badge
    elements.riskIcon.textContent = riskConfig.icon;
    elements.riskLabel.textContent = riskConfig.label;
    elements.riskBadge.className = `risk-badge ${riskConfig.class}`;
    elements.resultCard.className = `result-card ${riskConfig.class}`;

    // Update score circle
    elements.scoreValue.textContent = Math.round(result.risk_score);
    elements.scoreCircle.style.borderColor = riskConfig.color;

    // Update ML info
    elements.mlConfidence.textContent = (result.ml_confidence * 100).toFixed(1) + '%';

    // Update CTI count
    const activeCtis = result.ctis ? result.ctis.filter(c => c.found).length : 0;
    elements.ctiCount.textContent = activeCtis > 0 ? `${activeCtis} sources` : 'None';

    // Update processing time
    if (result.processing_time_ms) {
        elements.processingTime.querySelector('span').textContent =
            Math.round(result.processing_time_ms);
    }

    // Display threats
    displayThreats(result.threats || []);

    // Display CTI results
    displayCTIResults(result.ctis || []);

    // Update classification badge
    if (result.is_malicious) {
        elements.resultDetails.innerHTML = `
            <div class="malicious-indicator">
                <span class="malicious-icon">🚨</span>
                <span>This URL is classified as MALICIOUS</span>
            </div>
        `;
    } else {
        elements.resultDetails.innerHTML = '';
    }
}

function displayThreats(threats) {
    if (!threats || threats.length === 0) {
        elements.threatsSection.style.display = 'none';
        return;
    }

    elements.threatsSection.style.display = 'block';
    elements.threatList.innerHTML = '';

    threats.forEach(threat => {
        const threatItem = document.createElement('div');
        threatItem.className = 'threat-item';
        threatItem.innerHTML = `
            <div class="threat-header">
                <span class="threat-category">${threat.category.replace('_', ' ')}</span>
                <span class="threat-confidence">${(threat.confidence * 100).toFixed(0)}%</span>
            </div>
            <div class="threat-description">${threat.description}</div>
            ${threat.indicators.length > 0 ? `
                <ul class="threat-indicators">
                    ${threat.indicators.slice(0, 3).map(i => `<li>${i}</li>`).join('')}
                </ul>
            ` : ''}
        `;
        elements.threatList.appendChild(threatItem);
    });
}

function displayCTIResults(ctis) {
    const activeCtis = ctis.filter(c => c.found);

    if (activeCtis.length === 0) {
        elements.ctiSection.style.display = 'none';
        return;
    }

    elements.ctiSection.style.display = 'block';
    elements.ctiList.innerHTML = '';

    activeCtis.forEach(cti => {
        const ctiItem = document.createElement('div');
        ctiItem.className = `cti-item ${cti.malicious ? 'malicious' : 'clean'}`;

        const statusIcon = cti.malicious ? '🔴' : '🟢';
        const sourceName = cti.source.charAt(0).toUpperCase() + cti.source.slice(1);

        ctiItem.innerHTML = `
            <div class="cti-header">
                <span class="cti-source">${statusIcon} ${sourceName}</span>
                ${cti.positives > 0 ? `<span class="cti-positives">${cti.positives}/${cti.total}</span>` : ''}
            </div>
            ${cti.metadata && cti.metadata.malicious_vendors ?
                `<div class="cti-vendors">Flagged by: ${cti.metadata.malicious_vendors.slice(0, 2).join(', ')}</div>` : ''}
        `;
        elements.ctiList.appendChild(ctiItem);
    });
}

function showNonHttpPage() {
    elements.currentUrl.textContent = 'Not a web page';
    elements.riskIcon.textContent = '❓';
    elements.riskLabel.textContent = 'N/A';
    elements.scoreValue.textContent = '--';
    elements.resultDetails.innerHTML = '<p>Only web pages can be analyzed.</p>';
}

function showDefaultState() {
    elements.riskIcon.textContent = '⏳';
    elements.riskLabel.textContent = 'Waiting...';
    elements.scoreValue.textContent = '--';
    elements.mlConfidence.textContent = '--';
    elements.ctiCount.textContent = '--';
}

function showError(message) {
    elements.riskIcon.textContent = '❌';
    elements.riskLabel.textContent = 'Error';
    elements.resultDetails.innerHTML = `<p class="error-message">${message}</p>`;
}

// ============================================================================
// HISTORY
// ============================================================================

async function showHistory() {
    elements.historyPanel.style.display = 'block';

    try {
        const response = await sendMessage({ type: 'GET_HISTORY' });

        if (response.success && response.data.length > 0) {
            displayHistory(response.data);
        } else {
            elements.historyList.innerHTML = '<p class="empty-history">No scan history yet.</p>';
        }
    } catch (error) {
        elements.historyList.innerHTML = `<p class="error-message">Failed to load history</p>`;
    }
}

function displayHistory(history) {
    elements.historyList.innerHTML = '';

    history.forEach(item => {
        const riskLevel = getRiskLevel(item.risk_score);
        const riskConfig = CONFIG.riskLevels[riskLevel];

        const historyItem = document.createElement('div');
        historyItem.className = `history-item ${riskConfig.class}`;

        historyItem.innerHTML = `
            <div class="history-icon">${riskConfig.icon}</div>
            <div class="history-content">
                <div class="history-url">${truncateUrl(item.url, 40)}</div>
                <div class="history-meta">
                    <span>${riskConfig.label}</span>
                    <span>${item.risk_score.toFixed(0)}/100</span>
                    <span>${formatTime(item.analyzed_at)}</span>
                </div>
            </div>
        `;

        historyItem.addEventListener('click', () => {
            chrome.tabs.create({ url: item.url });
        });

        elements.historyList.appendChild(historyItem);
    });
}

// ============================================================================
// UTILITIES
// ============================================================================

function getRiskLevel(score) {
    if (score >= 80) return 'critical';
    if (score >= 60) return 'high';
    if (score >= 40) return 'medium';
    if (score >= 20) return 'low';
    return 'safe';
}

function truncateUrl(url, maxLength = 50) {
    if (url.length <= maxLength) return url;
    return url.substring(0, maxLength) + '...';
}

function formatTime(timestamp) {
    if (!timestamp) return '';
    const date = new Date(timestamp);
    const now = new Date();
    const diff = now - date;

    if (diff < 60000) return 'Just now';
    if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
    if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
    return date.toLocaleDateString();
}

function updateStatusIndicator(status, text) {
    const indicator = elements.statusIndicator;
    indicator.className = `status-indicator ${status}`;
    indicator.querySelector('.status-text').textContent = text;
}

function showLoading(show) {
    elements.loadingOverlay.style.display = show ? 'flex' : 'none';
    elements.analyzeBtn.disabled = show;
}

// ============================================================================
// MESSAGE HANDLING
// ============================================================================

function sendMessage(message) {
    return new Promise((resolve, reject) => {
        chrome.runtime.sendMessage(message, response => {
            if (chrome.runtime.lastError) {
                reject(new Error(chrome.runtime.lastError.message));
            } else {
                resolve(response);
            }
        });
    });
}

function handleBackgroundMessage(message, sender, sendResponse) {
    if (message.type === 'ANALYSIS_RESULT' && currentAnalysis?.url === message.url) {
        // Update if we got a new result for current URL
        displayAnalysis(message.result);
    }
}

// Log initialization
console.log('PhishGuard: Popup script loaded');