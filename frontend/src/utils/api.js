/**
 * PhishGuard API Client
 *
 * Centralized API utilities for communicating with the PhishGuard backend.
 */
import axios from 'axios'

// API Configuration
// Use proxy in development (/api -> backend), fallback to direct URL
const API_BASE_URL = import.meta.env.VITE_API_URL || '/api/v1'

// For production without proxy, uncomment:
// const API_BASE_URL = 'http://localhost:8000/api/v1'

// Create axios instance
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor
api.interceptors.request.use(
  (config) => {
    console.log(`[API] ${config.method?.toUpperCase()} ${config.url}`)
    return config
  },
  (error) => {
    console.error('[API] Request error:', error)
    return Promise.reject(error)
  }
)

// Response interceptor
api.interceptors.response.use(
  (response) => {
    return response
  },
  (error) => {
    console.error('[API] Response error:', error.response?.data || error.message)
    return Promise.reject(error)
  }
)

// ============================================================================
// API Endpoints
// ============================================================================

/**
 * Analyze a URL for phishing
 */
export async function analyzeUrl(url, options = {}) {
  const response = await api.post('/analyze', {
    url,
    include_raw_features: options.includeFeatures || false,
    enable_cti: options.enableCti !== false,
  })
  return response.data
}

/**
 * Batch analyze multiple URLs
 */
export async function batchAnalyze(urls, options = {}) {
  const response = await api.post('/analyze/batch', {
    urls,
    enable_cti: options.enableCti !== false,
  })
  return response.data
}

/**
 * Get analysis history
 */
export async function getHistory(limit = 50, riskLevel = null) {
  const params = new URLSearchParams()
  params.append('limit', limit)
  if (riskLevel) params.append('risk_level', riskLevel)

  const response = await api.get(`/history?${params}`)
  return response.data
}

/**
 * Get indicators of compromise
 */
export async function getIoCs(limit = 100, minRiskScore = 50) {
  const params = new URLSearchParams()
  params.append('limit', limit)
  params.append('min_risk_score', minRiskScore)

  const response = await api.get(`/iocs?${params}`)
  return response.data
}

/**
 * Export IoCs as JSON or CSV
 */
export async function exportIoCs(format = 'json', minRiskScore = 50) {
  const response = await api.get('/iocs/export', {
    params: { format, min_risk_score: minRiskScore },
    responseType: format === 'json' ? 'json' : 'text',
  })

  return response.data
}

/**
 * Get API health status
 */
export async function getHealth() {
  const response = await api.get('/health')
  return response.data
}

/**
 * Get ML model information
 */
export async function getModelInfo() {
  const response = await api.get('/model/info')
  return response.data
}

/**
 * Get dashboard statistics
 */
export async function getStats() {
  const response = await api.get('/stats')
  return response.data
}

/**
 * Clear analysis history
 */
export async function clearHistory() {
  const response = await api.delete('/history')
  return response.data
}

export default api