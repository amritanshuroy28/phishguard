import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Shield, AlertTriangle, CheckCircle, Clock, Trash2, ExternalLink } from 'lucide-react'
import {
  PieChart, Pie, Cell, ResponsiveContainer,
  BarChart, Bar, XAxis, YAxis, Tooltip,
} from 'recharts'
import { getStats, getHistory, clearHistory } from '../utils/api'
import { formatNumber, formatRelativeTime, CHART_COLORS, RISK_COLORS } from '../utils/helpers'
import { StatCard, RiskBadge, Spinner, AlertBanner } from '../components/SharedComponents'
import { Link } from 'react-router-dom'

export default function Dashboard() {
  const queryClient = useQueryClient()

  const { data: stats, isLoading: statsLoading, error: statsError, refetch: refetchStats } = useQuery({
    queryKey: ['stats'],
    queryFn: getStats,
    refetchInterval: 30000,
  })

  const { data: history, isLoading: historyLoading } = useQuery({
    queryKey: ['history', 10],
    queryFn: () => getHistory(10),
    refetchInterval: 15000,
  })

  const clearMutation = useMutation({
    mutationFn: clearHistory,
    onSuccess: () => {
      queryClient.invalidateQueries(['stats', 'history'])
    },
  })

  const handleClearHistory = () => {
    if (window.confirm('Are you sure you want to clear all scan history?')) {
      clearMutation.mutate()
    }
  }

  // Prepare chart data
  const riskDistribution = stats?.risk_distribution ? [
    { name: 'Safe', value: stats.risk_distribution.safe || 0, fill: CHART_COLORS.safe },
    { name: 'Low', value: stats.risk_distribution.low || 0, fill: CHART_COLORS.low },
    { name: 'Medium', value: stats.risk_distribution.medium || 0, fill: CHART_COLORS.medium },
    { name: 'High', value: stats.risk_distribution.high || 0, fill: CHART_COLORS.high },
    { name: 'Critical', value: stats.risk_distribution.critical || 0, fill: CHART_COLORS.critical },
  ].filter(d => d.value > 0) : []

  if (statsError) {
    return (
      <div className="max-w-7xl mx-auto">
        <AlertBanner
          type="error"
          message="Unable to connect to PhishGuard API. Please ensure the backend is running on port 8000."
        />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Security Dashboard</h1>
          <p className="text-gray-500">Real-time phishing URL analysis overview</p>
        </div>
        <Link
          to="/scan"
          className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors"
        >
          Scan New URL
        </Link>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="Total Scans"
          value={formatNumber(stats?.total_scans || 0)}
          icon={Shield}
          color="primary"
        />
        <StatCard
          title="Threats Detected"
          value={formatNumber(stats?.threats_detected || 0)}
          subtitle={`${stats?.threat_rate || 0}% threat rate`}
          icon={AlertTriangle}
          color="danger"
        />
        <StatCard
          title="Avg Risk Score"
          value={(stats?.average_risk_score || 0).toFixed(1)}
          subtitle="Out of 100"
          icon={Clock}
          color="warning"
        />
        <StatCard
          title="ML Status"
          value={stats?.ml_model_loaded ? 'Active' : 'Loading...'}
          icon={CheckCircle}
          color="success"
        />
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Risk Distribution Pie Chart */}
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Risk Distribution</h2>
          {statsLoading ? (
            <Spinner size="md" />
          ) : riskDistribution.length > 0 ? (
            <ResponsiveContainer width="100%" height={250}>
              <PieChart>
                <Pie
                  data={riskDistribution}
                  cx="50%"
                  cy="50%"
                  outerRadius={80}
                  dataKey="value"
                  label={({ name, percent }) => `${name} (${(percent * 100).toFixed(0)}%)`}
                  labelLine={false}
                >
                  {riskDistribution.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.fill} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-64 flex items-center justify-center text-gray-500">
              No scan data yet
            </div>
          )}
        </div>

        {/* Risk Bar Chart */}
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Detection Breakdown</h2>
          {statsLoading ? (
            <Spinner size="md" />
          ) : riskDistribution.length > 0 ? (
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={riskDistribution} layout="vertical">
                <XAxis type="number" />
                <YAxis dataKey="name" type="category" width={80} />
                <Tooltip />
                <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                  {riskDistribution.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.fill} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-64 flex items-center justify-center text-gray-500">
              No scan data yet
            </div>
          )}
        </div>
      </div>

      {/* Recent Scans Table */}
      <div className="bg-white rounded-lg shadow">
        <div className="flex justify-between items-center px-6 py-4 border-b border-gray-200">
          <h2 className="text-lg font-semibold text-gray-900">Recent Scans</h2>
          <div className="flex gap-2">
            <button
              onClick={handleClearHistory}
              className="p-2 text-gray-500 hover:text-red-600 hover:bg-red-50 rounded"
              title="Clear History"
            >
              <Trash2 className="h-4 w-4" />
            </button>
            <Link to="/history" className="text-primary-600 hover:text-primary-700 text-sm">
              View All →
            </Link>
          </div>
        </div>

        {historyLoading ? (
          <div className="p-8"><Spinner /></div>
        ) : history && history.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="data-table">
              <thead>
                <tr>
                  <th>URL</th>
                  <th>Risk Level</th>
                  <th>Score</th>
                  <th>ML Confidence</th>
                  <th>Time</th>
                </tr>
              </thead>
              <tbody>
                {history.slice(0, 10).map((scan) => (
                  <tr key={scan.id} className="cursor-pointer hover:bg-gray-50">
                    <td className="max-w-xs truncate" title={scan.url}>
                      {scan.url.length > 50 ? scan.url.substring(0, 50) + '...' : scan.url}
                    </td>
                    <td>
                      <RiskBadge score={scan.risk_score} size="sm" />
                    </td>
                    <td className="font-mono">{scan.risk_score.toFixed(1)}</td>
                    <td>{(scan.ml_confidence * 100).toFixed(1)}%</td>
                    <td className="text-gray-500">{formatRelativeTime(scan.analyzed_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="p-8 text-center text-gray-500">
            No scans yet. <Link to="/scan" className="text-primary-600">Start scanning</Link>
          </div>
        )}
      </div>

      {/* Legend */}
      <div className="bg-white rounded-lg shadow p-4">
        <h3 className="text-sm font-semibold text-gray-700 mb-3">Risk Level Guide</h3>
        <div className="flex flex-wrap gap-4">
          {Object.entries(RISK_COLORS).map(([level, colors]) => (
            <div key={level} className="flex items-center gap-2">
              <span className={`w-3 h-3 rounded-full ${colors.bg}`}></span>
              <span className="text-sm capitalize">{level}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}