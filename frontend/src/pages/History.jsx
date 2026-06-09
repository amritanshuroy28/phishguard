import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Download, Filter, Trash2 } from 'lucide-react'
import { getHistory, clearHistory, exportIoCs } from '../utils/api'
import { formatRelativeTime, downloadFile } from '../utils/helpers'
import { RiskBadge, Spinner, EmptyState } from '../components/SharedComponents'

const RISK_LEVELS = ['all', 'safe', 'low', 'medium', 'high', 'critical']

export default function History() {
  const [filter, setFilter] = useState('all')
  const [page, setPage] = useState(1)
  const pageSize = 20

  const { data, isLoading, refetch } = useQuery({
    queryKey: ['history', page],
    queryFn: () => getHistory(100),
  })

  const handleExport = async (format) => {
    const data = await exportIoCs(format === 'json' ? 'json' : 'csv')
    const filename = `phishguard-iocs-${new Date().toISOString().split('T')[0]}.${format}`
    downloadFile(data, filename, format === 'csv' ? 'text/csv' : 'application/json')
  }

  const handleClear = async () => {
    if (window.confirm('Clear all scan history? This cannot be undone.')) {
      await clearHistory()
      refetch()
    }
  }

  // Filter data
  const filteredData = data?.filter(item => {
    if (filter === 'all') return true
    const level = getLevel(item.risk_score)
    return level === filter
  }) || []

  const totalPages = Math.ceil(filteredData.length / pageSize)
  const paginatedData = filteredData.slice((page - 1) * pageSize, page * pageSize)

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Scan History</h1>
          <p className="text-gray-500">{data?.length || 0} total scans</p>
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
          <button
            onClick={handleClear}
            className="px-4 py-2 text-red-600 hover:bg-red-50 rounded-lg flex items-center gap-2"
          >
            <Trash2 className="h-4 w-4" />
            Clear All
          </button>
        </div>
      </div>

      {/* Filters */}
      <div className="flex gap-2 flex-wrap">
        {RISK_LEVELS.map((level) => (
          <button
            key={level}
            onClick={() => { setFilter(level); setPage(1); }}
            className={`px-3 py-1.5 rounded-full text-sm font-medium capitalize transition-colors ${
              filter === level
                ? 'bg-primary-600 text-white'
                : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
            }`}
          >
            {level === 'all' ? 'All' : level}
          </button>
        ))}
      </div>

      {/* Table */}
      <div className="bg-white rounded-lg shadow overflow-hidden">
        {isLoading ? (
          <div className="p-12"><Spinner /></div>
        ) : paginatedData.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="data-table">
              <thead>
                <tr>
                  <th>URL</th>
                  <th>Risk Level</th>
                  <th>Score</th>
                  <th>ML Confidence</th>
                  <th>Analysis Time</th>
                  <th>Processed</th>
                </tr>
              </thead>
              <tbody>
                {paginatedData.map((item) => (
                  <tr key={item.id}>
                    <td className="max-w-md truncate" title={item.url}>
                      {item.url}
                    </td>
                    <td>
                      <RiskBadge score={item.risk_score} size="sm" />
                    </td>
                    <td className="font-mono">{item.risk_score.toFixed(1)}</td>
                    <td>{(item.ml_confidence * 100).toFixed(1)}%</td>
                    <td>{item.processing_time_ms.toFixed(0)}ms</td>
                    <td className="text-gray-500">{formatRelativeTime(item.analyzed_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <EmptyState
            title="No scan history"
            description="Start scanning URLs to see them here"
          />
        )}
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex justify-center gap-2">
          <button
            onClick={() => setPage(Math.max(1, page - 1))}
            disabled={page === 1}
            className="px-3 py-1 bg-gray-100 rounded hover:bg-gray-200 disabled:opacity-50"
          >
            Previous
          </button>
          <span className="px-4 py-1">
            Page {page} of {totalPages}
          </span>
          <button
            onClick={() => setPage(Math.min(totalPages, page + 1))}
            disabled={page === totalPages}
            className="px-3 py-1 bg-gray-100 rounded hover:bg-gray-200 disabled:opacity-50"
          >
            Next
          </button>
        </div>
      )}
    </div>
  )
}

function getLevel(score) {
  if (score >= 80) return 'critical'
  if (score >= 60) return 'high'
  if (score >= 40) return 'medium'
  if (score >= 20) return 'low'
  return 'safe'
}