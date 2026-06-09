import { Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import ScanURL from './pages/ScanURL'
import History from './pages/History'
import Threats from './pages/Threats'

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/scan" element={<ScanURL />} />
        <Route path="/history" element={<History />} />
        <Route path="/threats" element={<Threats />} />
      </Routes>
    </Layout>
  )
}