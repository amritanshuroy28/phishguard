/**
 * PhishGuard Utility Functions
 */

/**
 * Get risk level from score
 */
export function getRiskLevel(score) {
  if (score >= 80) return { level: 'critical', label: 'CRITICAL', color: 'text-white', bg: 'bg-critical' }
  if (score >= 60) return { level: 'high', label: 'High Risk', color: 'text-white', bg: 'bg-high' }
  if (score >= 40) return { level: 'medium', label: 'Medium', color: 'text-white', bg: 'bg-medium' }
  if (score >= 20) return { level: 'low', label: 'Low Risk', color: 'text-white', bg: 'bg-low' }
  return { level: 'safe', label: 'Safe', color: 'text-white', bg: 'bg-safe' }
}

/**
 * Get badge colors for risk levels
 */
export const RISK_COLORS = {
  safe: { bg: 'bg-safe', text: 'text-safe', light: 'bg-safe-light' },
  low: { bg: 'bg-low', text: 'text-low', light: 'bg-low-light' },
  medium: { bg: 'bg-medium', text: 'text-medium', light: 'bg-medium-light' },
  high: { bg: 'bg-high', text: 'text-high', light: 'bg-high-light' },
  critical: { bg: 'bg-critical', text: 'text-critical', light: 'bg-critical-light' },
}

/**
 * Get chart colors for risk levels
 */
export const CHART_COLORS = {
  safe: '#4CAF50',
  low: '#8BC34A',
  medium: '#FF9800',
  high: '#F44336',
  critical: '#B71C1C',
}

/**
 * Format timestamp to readable string
 */
export function formatTimestamp(timestamp) {
  if (!timestamp) return '-'
  const date = new Date(timestamp)
  return date.toLocaleString()
}

/**
 * Format relative time (e.g., "2 hours ago")
 */
export function formatRelativeTime(timestamp) {
  if (!timestamp) return '-'
  const date = new Date(timestamp)
  const now = new Date()
  const diff = now - date

  const seconds = Math.floor(diff / 1000)
  const minutes = Math.floor(seconds / 60)
  const hours = Math.floor(minutes / 60)
  const days = Math.floor(hours / 24)

  if (seconds < 60) return 'Just now'
  if (minutes < 60) return `${minutes}m ago`
  if (hours < 24) return `${hours}h ago`
  if (days < 7) return `${days}d ago`
  return formatTimestamp(timestamp)
}

/**
 * Truncate URL for display
 */
export function truncateUrl(url, maxLength = 60) {
  if (!url) return '-'
  if (url.length <= maxLength) return url

  try {
    const parsed = new URL(url)
    const truncated = `${parsed.protocol}//${parsed.host}`
    if (truncated.length >= maxLength - 3) {
      return truncated.substring(0, maxLength - 3) + '...'
    }
    return truncated + '/...'
  } catch {
    return url.substring(0, maxLength - 3) + '...'
  }
}

/**
 * Format number with thousands separator
 */
export function formatNumber(num) {
  if (num === undefined || num === null) return '-'
  return new Intl.NumberFormat().format(num)
}

/**
 * Calculate percentage
 */
export function calculatePercentage(value, total) {
  if (!total) return 0
  return Math.round((value / total) * 100)
}

/**
 * Download data as file
 */
export function downloadFile(data, filename, type = 'application/json') {
  const blob = new Blob([typeof data === 'string' ? data : JSON.stringify(data, null, 2)], { type })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
  URL.revokeObjectURL(url)
}

/**
 * Copy to clipboard
 */
export async function copyToClipboard(text) {
  try {
    await navigator.clipboard.writeText(text)
    return true
  } catch {
    return false
  }
}