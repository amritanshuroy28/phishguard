import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Download, AlertTriangle, Copy, Check } from 'lucide-react'
import { getIoCs, exportIoCs } from '../utils/api'
import { formatTimestamp, downloadFile, copyToClipboard } from '../utils/helpers'
import { RiskBadge, Spinner, EmptyState } from '../components/SharedComponents'

export default function Threats() {
  const [copiedId, setCopiedId] = useState(null)

  const { data, isLoading } = useQuery({
    queryKey: ['iocs'],
    queryFn: () => getIoCs(100, 0),
  })

  const handleExport = async (format) => {
    const exportData = await exportIoCs(format, 0)
    const filename = `phishguard-threats-${new Date().toISOString().split('T')[0]}.${format}`
    downloadFile(exportData, filename, format === 'csv' ? 'text/csv' : 'application/json')
  }

  const handleCopy = async (value, id) => {
    await copyToClipboard(value)
    setCopiedId(id)
    setTimeout(() => setCopiedId(null), 2000)
  }

  const threats = data || []

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Indicators of Compromise</h1>
          <p className="text-gray-500">{threats.length} threats detected</p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => handleExport('csv')}
            className="px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 flex items-center gap-2"
          >
            <Download className="h-4 w-4" />
            Export CSV
          </button>
          <button
            onClick={() => handleExport('json')}
            className="px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 flex items-center gap-2"
          >
            <Download className="h-4 w-4" />
            Export JSON
          </button>
        </div>
      </div>

      {/* Info Banner */}
      <div className="bg-primary-50 border border-primary-200 rounded-lg p-4">
        <div className="flex items-start gap-3">
          <AlertTriangle className="h-5 w-5 text-primary-600 mt-0.5" />
          <div>
            <h3 className="font-semibold text-primary-900">About IoCs</h3>
            <p className="text-sm text-primary-700 mt-1">
              Indicators of Compromise are URLs flagged as malicious by either ML analysis or threat intelligence sources.
              Export these for integration with your SIEM, firewall, or other security tools.
            </p>
          </div>
        </div>
      </div>

      {/* Threats Table */}
      <div className="bg-white rounded-lg shadow overflow-hidden">
        {isLoading ? (
          <div className="p-12"><Spinner /></div>
        ) : threats.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Type</th>
                  <th>Value</th>
                  <th>Threat Type</th>
                  <th>Risk Level</th>
                  <th>Confidence</th>
                  <th>Source</th>
                  <th>Last Seen</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {threats.map((ioc, idx) => (
                  <tr key={idx} className="hover:bg-gray-50">
                    <td>
                      <span className="px-2 py-1 bg-gray-100 rounded text-xs uppercase">
                        {ioc.type}
                      </span>
                    </td>
                    <td className="max-w-xs truncate font-mono text-sm" title={ioc.value}>
                      {ioc.value.length > 50 ? ioc.value.substring(0, 50) + '...' : ioc.value}
                    </td>
                    <td className="capitalize">{ioc.threat_type.replace('_', ' ')}</td>
                    <td>
                      <RiskBadge score={ioc.risk_level === 'critical' ? 90 : ioc.risk_level === 'high' ? 70 : 50} size="sm" />
                    </td>
                    <td>{(ioc.confidence * 100).toFixed(0)}%</td>
                    <td className="text-gray-500">{ioc.source}</td>
                    <td className="text-gray-500">{formatTimestamp(ioc.last_seen)}</td>
                    <td>
                      <button
                        onClick={() => handleCopy(ioc.value, idx)}
                        className="p-1 hover:bg-gray-100 rounded"
                        title="Copy value"
                      >
                        {copiedId === idx ? (
                          <Check className="h-4 w-4 text-green-600" />
                        ) : (
                          <Copy className="h-4 w-4 text-gray-400" />
                        )}
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <EmptyState
            icon={AlertTriangle}
            title="No threats detected"
            description="URLs flagged as malicious will appear here"
          />
        )}
      </div>

      {/* Threat Type Summary */}
      {threats.length > 0 && (
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-lg font-semibold mb-4">Threat Summary</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {getThreatSummary(threats).map((item) => (
              <div key={item.type} className="bg-gray-50 rounded-lg p-4">
                <div className="text-2xl font-bold text-gray-900">{item.count}</div>
                <div className="text-sm text-gray-500 capitalize">{item.type.replace('_', ' ')}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

function getThreatSummary(threats) {
  const summary = {}
  threats.forEach(t => {
    summary[t.threat_type] = (summary[t.threat_type] || 0) + 1
  })
  return Object.entries(summary).map(([type, count]) => ({ type, count }))
}