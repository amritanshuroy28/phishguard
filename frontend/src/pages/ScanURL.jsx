import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { Search, CheckCircle, XCircle, Loader2, Copy, ExternalLink } from 'lucide-react'
import { analyzeUrl } from '../utils/api'
import { RiskBadge, Spinner, AlertBanner } from '../components/SharedComponents'
import { formatTimestamp, copyToClipboard, getRiskLevel } from '../utils/helpers'

export default function ScanURL() {
  const [url, setUrl] = useState('')
  const [showResult, setShowResult] = useState(false)

  const analyzeMutation = useMutation({
    mutationFn: (url) => analyzeUrl(url, { includeFeatures: true, enableCti: true }),
    onSuccess: () => setShowResult(true),
  })

  const handleSubmit = (e) => {
    e.preventDefault()
    if (url.trim()) {
      analyzeMutation.mutate(url.trim())
    }
  }

  const result = analyzeMutation.data

  return (
    <div className="max-w-3xl mx-auto space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Scan URL</h1>
        <p className="text-gray-500">Analyze a single URL for phishing indicators</p>
      </div>

      {/* URL Input Form */}
      <form onSubmit={handleSubmit} className="bg-white rounded-lg shadow p-6">
        <div className="flex gap-3">
          <div className="flex-1">
            <input
              type="url"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="Enter URL to analyze (e.g., https://example.com)"
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 outline-none"
              required
            />
          </div>
          <button
            type="submit"
            disabled={analyzeMutation.isPending}
            className="px-6 py-3 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:bg-gray-300 disabled:cursor-not-allowed flex items-center gap-2 transition-colors"
          >
            {analyzeMutation.isPending ? (
              <>
                <Loader2 className="h-5 w-5 animate-spin" />
                Analyzing...
              </>
            ) : (
              <>
                <Search className="h-5 w-5" />
                Scan
              </>
            )}
          </button>
        </div>

        {analyzeMutation.isError && (
          <AlertBanner
            type="error"
            message={`Analysis failed: ${analyzeMutation.error.message}`}
          />
        )}
      </form>

      {/* Results */}
      {analyzeMutation.isPending && (
        <div className="bg-white rounded-lg shadow p-12 text-center">
          <Spinner size="lg" />
          <p className="mt-4 text-gray-600">Analyzing URL...</p>
        </div>
      )}

      {showResult && result && (
        <div className="bg-white rounded-lg shadow overflow-hidden">
          {/* Result Header */}
          <div className={`p-6 ${result.is_malicious ? 'bg-critical-light' : 'bg-safe-light'}`}>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                {result.is_malicious ? (
                  <XCircle className="h-12 w-12 text-critical" />
                ) : (
                  <CheckCircle className="h-12 w-12 text-safe" />
                )}
                <div>
                  <h2 className="text-xl font-bold">
                    {result.is_malicious ? 'Malicious URL Detected!' : 'URL Appears Safe'}
                  </h2>
                  <p className="text-gray-600">{truncateUrl(result.url)}</p>
                </div>
              </div>
              <div className="text-right">
                <RiskBadge score={result.risk_score} size="lg" />
                <p className="text-sm text-gray-500 mt-1">
                  Score: {result.risk_score.toFixed(1)}/100
                </p>
              </div>
            </div>
          </div>

          {/* Details */}
          <div className="p-6 space-y-6">
            {/* ML Analysis */}
            <div>
              <h3 className="text-sm font-semibold text-gray-700 mb-3">ML Analysis</h3>
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div className="bg-gray-50 rounded p-3">
                  <span className="text-gray-500">Prediction:</span>
                  <span className={`ml-2 font-medium ${result.ml_prediction ? 'text-critical' : 'text-safe'}`}>
                    {result.ml_prediction ? 'Phishing' : 'Legitimate'}
                  </span>
                </div>
                <div className="bg-gray-50 rounded p-3">
                  <span className="text-gray-500">Confidence:</span>
                  <span className="ml-2 font-medium">{(result.ml_confidence * 100).toFixed(1)}%</span>
                </div>
                <div className="bg-gray-50 rounded p-3">
                  <span className="text-gray-500">Model:</span>
                  <span className="ml-2 font-mono text-xs">{result.ml_model_version}</span>
                </div>
                <div className="bg-gray-50 rounded p-3">
                  <span className="text-gray-500">Processing Time:</span>
                  <span className="ml-2 font-medium">{result.processing_time_ms.toFixed(0)}ms</span>
                </div>
              </div>
            </div>

            {/* CTI Results */}
            {result.ctis && result.ctis.filter(c => c.found).length > 0 && (
              <div>
                <h3 className="text-sm font-semibold text-gray-700 mb-3">Threat Intelligence</h3>
                <div className="space-y-2">
                  {result.ctis.filter(c => c.found).map((cti, idx) => (
                    <div key={idx} className={`p-3 rounded ${cti.malicious ? 'bg-critical-light' : 'bg-gray-50'}`}>
                      <div className="flex justify-between items-center">
                        <span className="font-medium">{cti.source}</span>
                        <span className={`text-sm ${cti.malicious ? 'text-critical' : 'text-safe'}`}>
                          {cti.malicious ? 'Malicious' : 'Clean'}
                        </span>
                      </div>
                      {cti.positives > 0 && (
                        <p className="text-sm text-gray-600 mt-1">
                          Detections: {cti.positives}/{cti.total}
                        </p>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Threats */}
            {result.threats && result.threats.length > 0 && (
              <div>
                <h3 className="text-sm font-semibold text-gray-700 mb-3">Detected Threats</h3>
                <div className="space-y-2">
                  {result.threats.map((threat, idx) => (
                    <div key={idx} className="bg-high-light border-l-4 border-high p-3 rounded">
                      <div className="flex justify-between">
                        <span className="font-medium capitalize">{threat.category.replace('_', ' ')}</span>
                        <span className="text-sm bg-high text-white px-2 py-0.5 rounded">
                          {Math.round(threat.confidence * 100)}%
                        </span>
                      </div>
                      <p className="text-sm text-gray-600 mt-1">{threat.description}</p>
                      {threat.indicators && threat.indicators.length > 0 && (
                        <ul className="text-xs text-gray-500 mt-2 list-disc list-inside">
                          {threat.indicators.slice(0, 3).map((ind, i) => (
                            <li key={i}>{ind}</li>
                          ))}
                        </ul>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Actions */}
            <div className="flex gap-3 pt-4 border-t">
              <button
                onClick={() => copyToClipboard(result.url)}
                className="px-4 py-2 bg-gray-100 text-gray-700 rounded hover:bg-gray-200 flex items-center gap-2 text-sm"
              >
                <Copy className="h-4 w-4" />
                Copy URL
              </button>
              <button
                onClick={() => navigator.clipboard.writeText(JSON.stringify(result, null, 2))}
                className="px-4 py-2 bg-gray-100 text-gray-700 rounded hover:bg-gray-200 flex items-center gap-2 text-sm"
              >
                Copy JSON
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function truncateUrl(url, maxLen = 80) {
  if (!url || url.length <= maxLen) return url || '-'
  return url.substring(0, maxLen) + '...'
}