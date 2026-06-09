/**
 * PhishGuard Chrome Extension - Background Service Worker
 * ========================================================
 *
 * This service worker intercepts URL navigations and page updates
 * to perform real-time phishing analysis using the PhishGuard API.
 *
 * Workflow:
 * 1. Monitors tab updates and URL changes
 * 2. Extracts current URL from active tab
 * 3. Sends URL to PhishGuard API for analysis
 * 4. Updates badge icon based on threat level
 * 5. Stores results for popup display
 *
 * Architecture:
 * - Uses Chrome Alarms for periodic health checks
 * - Maintains in-memory cache of recent analyses
 * - Communicates with popup via chrome.storage
 */

// ============================================================================
// CONFIGURATION
// ============================================================================

const CONFIG = {
    API_BASE_URL: 'http://localhost:8000/api/v1',
    // Fallback to online API if backend not locally available
    API_ENDPOINTS: {
        analyze: '/analyze',
        health: '/health'
    },

    // Cache settings
    CACHE_TTL_MS: 5 * 60 * 1000,  // 5 minutes
    MAX_CACHED_URLS: 100,

    // Alert settings
    NOTIFICATION_ENABLED: true,
    BADGE_COLORS: {
        safe: '#4CAF50',     // Green
        low: '#8BC34A',      // Light green
        medium: '#FF9800',   // Orange
        high: '#F44336',    // Red
        critical: '#B71C1C', // Dark red
        unknown: '#9E9E9E'   // Gray
    },

    // Risk thresholds
    RISK_THRESHOLDS: {
        SAFE: 20,
        LOW: 40,
        MEDIUM: 60,
        HIGH: 80
    }
};

// ============================================================================
// STATE MANAGEMENT
// ============================================================================

/**
 * Extension state storage
 */
class ExtensionState {
    constructor() {
        this.analyses = new Map();  // URL -> AnalysisResult
        this.currentTabUrl = null;
        this.lastHealthCheck = null;
        this.apiAvailable = true;
    }

    /**
     * Get analysis result for a URL
     */
    getAnalysis(url) {
        const cached = this.analyses.get(url);
        if (cached && Date.now() - cached.timestamp < CONFIG.CACHE_TTL_MS) {
            return cached.data;
        }
        return null;
    }

    /**
     * Store analysis result
     */
    setAnalysis(url, data) {
        // Cleanup old entries if cache is too large
        if (this.analyses.size >= CONFIG.MAX_CACHED_URLS) {
            const oldestKey = this.analyses.keys().next().value;
            this.analyses.delete(oldestKey);
        }

        this.analyses.set(url, {
            data,
            timestamp: Date.now()
        });

        // Persist to chrome.storage for popup access
        this.persistToStorage();
    }

    /**
     * Persist analyses to chrome.storage
     */
    async persistToStorage() {
        try {
            const data = Object.fromEntries(this.analyses);
            const serialized = {};

            for (const [url, entry] of Object.entries(data)) {
                serialized[url] = {
                    data: entry.data,
                    timestamp: entry.timestamp
                };
            }

            await chrome.storage.local.set({ analyses: serialized });
        } catch (error) {
            console.error('PhishGuard: Failed to persist to storage', error);
        }
    }

    /**
     * Load analyses from chrome.storage
     */
    async loadFromStorage() {
        try {
            const result = await chrome.storage.local.get('analyses');
            if (result.analyses) {
                for (const [url, entry] of Object.entries(result.analyses)) {
                    if (Date.now() - entry.timestamp < CONFIG.CACHE_TTL_MS) {
                        this.analyses.set(url, entry);
                    }
                }
            }
        } catch (error) {
            console.error('PhishGuard: Failed to load from storage', error);
        }
    }
}

const state = new ExtensionState();

// ============================================================================
// API CLIENT
// ============================================================================

/**
 * Call PhishGuard API to analyze a URL
 */
async function analyzeUrl(url) {
    // Check cache first
    const cachedResult = state.getAnalysis(url);
    if (cachedResult) {
        console.log('PhishGuard: Using cached result for', url);
        return cachedResult;
    }

    // Make API call
    try {
        console.log('PhishGuard: Analyzing URL', url);

        const response = await fetch(`${CONFIG.API_BASE_URL}${CONFIG.API_ENDPOINTS.analyze}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                url: url,
                include_raw_features: false,
                enable_cti: true
            })
        });

        if (!response.ok) {
            throw new Error(`API returned ${response.status}`);
        }

        const result = await response.json();

        // Cache the result
        state.setAnalysis(url, result);

        state.apiAvailable = true;
        return result;

    } catch (error) {
        console.error('PhishGuard: API call failed', error);
        state.apiAvailable = false;
        throw error;
    }
}

/**
 * Check API health
 */
async function checkApiHealth() {
    try {
        const response = await fetch(`${CONFIG.API_BASE_URL}${CONFIG.API_ENDPOINTS.health}`);
        if (response.ok) {
            state.apiAvailable = true;
            state.lastHealthCheck = Date.now();
            return true;
        }
    } catch (error) {
        state.apiAvailable = false;
    }
    return false;
}

// ============================================================================
// BADGE & NOTIFICATION MANAGEMENT
// ============================================================================

/**
 * Get risk level from threat score
 */
function getRiskLevel(threatScore) {
    if (threatScore >= CONFIG.RISK_THRESHOLDS.HIGH) return 'critical';
    if (threatScore >= CONFIG.RISK_THRESHOLDS.MEDIUM) return 'high';
    if (threatScore >= CONFIG.RISK_THRESHOLDS.LOW) return 'medium';
    if (threatScore >= CONFIG.RISK_THRESHOLDS.SAFE) return 'low';
    return 'safe';
}

/**
 * Get risk label for display
 */
function getRiskLabel(riskLevel, threatScore) {
    const labels = {
        safe: 'Safe',
        low: 'Low Risk',
        medium: 'Medium',
        high: 'High Risk',
        critical: 'CRITICAL!'
    };
    return labels[riskLevel] || 'Unknown';
}

/**
 * Update extension badge based on threat level
 */
function updateBadge(tabId, result) {
    const riskLevel = getRiskLevel(result.risk_score);
    const color = CONFIG.BADGE_COLORS[riskLevel];
    const label = getRiskLabel(riskLevel, result.risk_score);

    // Set badge text
    chrome.action.setBadgeText({
        text: label === 'Safe' ? '✓' : result.risk_score.toFixed(0),
        tabId: tabId
    });

    // Set badge color
    chrome.action.setBadgeBackgroundColor({
        color: color,
        tabId: tabId
    }).catch(() => {
        // Ignore errors if tab is not available
    });

    // Update badge title (tooltip)
    const title = result.is_malicious
        ? `PhishGuard: ${label} - Threats detected!`
        : `PhishGuard: ${label}`;

    chrome.action.setTitle({
        title: title,
        tabId: tabId
    }).catch(() => {});
}

/**
 * Show notification for high-risk URLs
 */
function showThreatNotification(tabId, result) {
    if (!CONFIG.NOTIFICATION_ENABLED) return;

    const riskLevel = getRiskLevel(result.risk_score);
    if (riskLevel !== 'high' && riskLevel !== 'critical') return;

    // Only notify for new threats
    const cached = state.getAnalysis(result.url);
    if (cached && cached.risk_score === result.risk_score) return;

    chrome.notifications.create({
        type: 'basic',
        iconUrl: 'icons/icon48.png',
        title: '⚠️ PhishGuard Alert',
        message: `Suspicious URL detected!\n\nThreat Score: ${result.risk_score}/100\nThreats: ${result.threats.length}`,
        priority: 2,
        requireInteraction: riskLevel === 'critical'
    });
}

// ============================================================================
// TAB MONITORING
// ============================================================================

/**
 * Handle tab URL changes
 */
async function handleTabUpdate(tabId, changeInfo) {
    if (changeInfo.status !== 'complete') return;

    try {
        const tab = await chrome.tabs.get(tabId);
        const url = tab.url;

        // Skip invalid URLs, chrome://, etc.
        if (!url || !url.startsWith('http') || url.startsWith('https://chrome')) {
            return;
        }

        state.currentTabUrl = url;

        // Analyze the URL
        try {
            const result = await analyzeUrl(url);
            updateBadge(tabId, result);

            if (result.is_malicious) {
                showThreatNotification(tabId, result);
            }

            // Send result to popup if open
            broadcastToPopup(url, result);

        } catch (error) {
            console.error('PhishGuard: Analysis failed', error);
            // Set neutral badge for errors
            chrome.action.setBadgeText({ text: '?', tabId: tabId });
            chrome.action.setBadgeBackgroundColor({ color: CONFIG.BADGE_COLORS.unknown, tabId: tabId });
        }

    } catch (error) {
        console.error('PhishGuard: Tab update failed', error);
    }
}

/**
 * Broadcast analysis result to popup
 */
function broadcastToPopup(url, result) {
    chrome.runtime.sendMessage({
        type: 'ANALYSIS_RESULT',
        url: url,
        result: result
    }).catch(() => {
        // Popup might not be open, that's fine
    });
}

// ============================================================================
// MESSAGE HANDLING
// ============================================================================

/**
 * Handle messages from popup and content scripts
 */
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    console.log('PhishGuard: Received message', message.type);

    switch (message.type) {
        case 'GET_ANALYSIS':
            handleGetAnalysis(message.url)
                .then(result => sendResponse({ success: true, data: result }))
                .catch(error => sendResponse({ success: false, error: error.message }));
            return true;  // Async response

        case 'ANALYZE_URL':
            handleAnalyzeUrl(message.url)
                .then(result => sendResponse({ success: true, data: result }))
                .catch(error => sendResponse({ success: false, error: error.message }));
            return true;

        case 'GET_STATUS':
            sendResponse({
                apiAvailable: state.apiAvailable,
                currentUrl: state.currentTabUrl,
                analysisCount: state.analyses.size
            });
            return false;

        case 'GET_HISTORY':
            getAnalysisHistory()
                .then(history => sendResponse({ success: true, data: history }))
                .catch(error => sendResponse({ success: false, error: error.message }));
            return true;

        default:
            console.log('PhishGuard: Unknown message type', message.type);
            return false;
    }
});

/**
 * Get analysis for a specific URL
 */
async function handleGetAnalysis(url) {
    const cached = state.getAnalysis(url);
    if (cached) return cached;

    return await analyzeUrl(url);
}

/**
 * Analyze a URL on demand
 */
async function handleAnalyzeUrl(url) {
    return await analyzeUrl(url);
}

/**
 * Get analysis history
 */
async function getAnalysisHistory() {
    const history = [];
    for (const [url, entry] of state.analyses) {
        history.push({
            url: url,
            ...entry.data,
            cached: Date.now() - entry.timestamp
        });
    }
    // Sort by most recent first
    history.sort((a, b) => new Date(b.analyzed_at) - new Date(a.analyzed_at));
    return history.slice(0, 50);
}

// ============================================================================
// SERVICE WORKER LIFECYCLE
// ============================================================================

/**
 * Extension installed/updated
 */
chrome.runtime.onInstalled.addListener(async (details) => {
    console.log('PhishGuard: Extension installed/updated', details.reason);

    // Initialize state from storage
    await state.loadFromStorage();

    // Check API health periodically
    checkApiHealth();
    setInterval(checkApiHealth, 5 * 60 * 1000);  // Every 5 minutes

    // Analyze current active tab if available
    try {
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
        if (tab && tab.url && tab.url.startsWith('http')) {
            state.currentTabUrl = tab.url;
            analyzeUrl(tab.url).then(result => {
                if (result) updateBadge(tab.id, result);
            }).catch(() => {});
        }
    } catch (error) {
        console.error('PhishGuard: Failed to analyze active tab on install', error);
    }
});

/**
 * Extension started (e.g., browser restart)
 */
chrome.runtime.onStartup.addListener(async () => {
    console.log('PhishGuard: Extension starting up');
    await state.loadFromStorage();

    // Check API health
    await checkApiHealth();

    // Analyze current active tab
    try {
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
        if (tab && tab.url && tab.url.startsWith('http')) {
            state.currentTabUrl = tab.url;
            analyzeUrl(tab.url).then(result => {
                if (result) updateBadge(tab.id, result);
            }).catch(() => {});
        }
    } catch (error) {
        console.error('PhishGuard: Failed to analyze active tab on startup', error);
    }
});

/**
 * Listen for tab updates
 */
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
    handleTabUpdate(tabId, changeInfo);
});

/**
 * Listen for tab switches
 */
chrome.tabs.onActivated.addListener(async (activeInfo) => {
    try {
        const tab = await chrome.tabs.get(activeInfo.tabId);
        if (tab && tab.url && tab.url.startsWith('http')) {
            state.currentTabUrl = tab.url;

            // Check cache first
            const cached = state.getAnalysis(tab.url);
            if (cached) {
                updateBadge(activeInfo.tabId, cached);
            } else if (state.apiAvailable) {
                analyzeUrl(tab.url).then(result => {
                    if (result) updateBadge(activeInfo.tabId, result);
                }).catch(() => {});
            }
        }
    } catch (error) {
        console.error('PhishGuard: Tab activation error', error);
    }
});

console.log('PhishGuard: Background service worker initialized');