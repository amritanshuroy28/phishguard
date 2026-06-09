import { getRiskLevel } from '../utils/helpers'

/**
 * Risk Badge Component
 */
export function RiskBadge({ score, size = 'md' }) {
  const { level, label, bg } = getRiskLevel(score)

  const sizeClasses = {
    sm: 'px-2 py-0.5 text-xs',
    md: 'px-2.5 py-1 text-sm',
    lg: 'px-3 py-1.5 text-base',
  }

  return (
    <span
      className={`inline-flex items-center font-semibold rounded-full ${bg} ${sizeClasses[size]} ${level === 'critical' ? 'animate-pulse' : ''}`}
    >
      {label}
    </span>
  )
}

/**
 * Score Circle Component
 */
export function ScoreCircle({ score, size = 80 }) {
  const { level, color } = getRiskLevel(score)
  const circumference = 2 * Math.PI * (size / 2 - 4)
  const progress = (score / 100) * circumference

  const colorMap = {
    safe: '#4CAF50',
    low: '#8BC34A',
    medium: '#FF9800',
    high: '#F44336',
    critical: '#B71C1C',
  }

  return (
    <div
      className="relative inline-flex items-center justify-center"
      style={{ width: size, height: size }}
    >
      <svg className="transform -rotate-90" width={size} height={size}>
        {/* Background circle */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={size / 2 - 4}
          fill="none"
          stroke="#e5e7eb"
          strokeWidth="6"
        />
        {/* Progress circle */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={size / 2 - 4}
          fill="none"
          stroke={colorMap[level]}
          strokeWidth="6"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={circumference - progress}
          className="transition-all duration-500"
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-2xl font-bold" style={{ color: colorMap[level] }}>
          {Math.round(score)}
        </span>
      </div>
    </div>
  )
}

/**
 * Stat Card Component
 */
export function StatCard({ title, value, subtitle, icon: Icon, trend, color = 'primary' }) {
  const colorClasses = {
    primary: 'bg-primary-50 text-primary-600',
    success: 'bg-green-50 text-green-600',
    danger: 'bg-red-50 text-red-600',
    warning: 'bg-yellow-50 text-yellow-600',
    info: 'bg-blue-50 text-blue-600',
  }

  return (
    <div className="bg-white rounded-lg shadow p-4 card-hover">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium text-gray-500">{title}</p>
          <p className="text-2xl font-bold text-gray-900 mt-1">{value}</p>
          {subtitle && <p className="text-xs text-gray-400 mt-1">{subtitle}</p>}
          {trend && (
            <p className={`text-xs mt-1 ${trend > 0 ? 'text-green-600' : 'text-red-600'}`}>
              {trend > 0 ? '↑' : '↓'} {Math.abs(trend)}%
            </p>
          )}
        </div>
        {Icon && (
          <div className={`p-3 rounded-lg ${colorClasses[color]}`}>
            <Icon className="h-6 w-6" />
          </div>
        )}
      </div>
    </div>
  )
}

/**
 * Loading Spinner Component
 */
export function Spinner({ size = 'md' }) {
  const sizeClasses = {
    sm: 'w-4 h-4',
    md: 'w-8 h-8',
    lg: 'w-12 h-12',
  }

  return (
    <div className="flex items-center justify-center p-4">
      <div className={`${sizeClasses[size]} spinner`} />
    </div>
  )
}

/**
 * Empty State Component
 */
export function EmptyState({ icon: Icon, title, description, action }) {
  return (
    <div className="text-center py-12">
      {Icon && <Icon className="mx-auto h-12 w-12 text-gray-400" />}
      <h3 className="mt-2 text-sm font-semibold text-gray-900">{title}</h3>
      {description && <p className="mt-1 text-sm text-gray-500">{description}</p>}
      {action && <div className="mt-6">{action}</div>}
    </div>
  )
}

/**
 * Alert Banner Component
 */
export function AlertBanner({ type = 'info', message, onDismiss }) {
  const typeClasses = {
    info: 'bg-blue-50 text-blue-800 border-blue-200',
    success: 'bg-green-50 text-green-800 border-green-200',
    warning: 'bg-yellow-50 text-yellow-800 border-yellow-200',
    error: 'bg-red-50 text-red-800 border-red-200',
  }

  return (
    <div className={`p-4 rounded-lg border ${typeClasses[type]}`}>
      <div className="flex items-center justify-between">
        <p className="text-sm">{message}</p>
        {onDismiss && (
          <button onClick={onDismiss} className="text-sm hover:opacity-75">
            Dismiss
          </button>
        )}
      </div>
    </div>
  )
}