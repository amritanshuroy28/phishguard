/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Risk level colors
        safe: '#4CAF50',
        'safe-light': '#E8F5E9',
        low: '#8BC34A',
        'low-light': '#F1F8E9',
        medium: '#FF9800',
        'medium-light': '#FFF3E0',
        high: '#F44336',
        'high-light': '#FFEBEE',
        critical: '#B71C1C',
        'critical-light': '#FFCDD2',

        // Brand colors
        primary: {
          50: '#E3F2FD',
          100: '#BBDEFB',
          200: '#90CAF9',
          300: '#64B5F6',
          400: '#42A5F5',
          500: '#2196F3',
          600: '#1E88E5',
          700: '#1976D2',
          800: '#1565C0',
          900: '#0D47A1',
        }
      },
      fontFamily: {
        sans: [
          '-apple-system',
          'BlinkMacSystemFont',
          '"Segoe UI"',
          'Roboto',
          '"Helvetica Neue"',
          'Arial',
          'sans-serif'
        ],
        mono: [
          '"SF Mono"',
          'Monaco',
          '"Cascadia Code"',
          '"Roboto Mono"',
          'Consolas',
          'monospace'
        ]
      }
    },
  },
  plugins: [],
}