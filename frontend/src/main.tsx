import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'

// No StrictMode: its dev-only double-mount would open two chat WebSockets and
// hence two vibe-acp sessions per page load.
createRoot(document.getElementById('root')!).render(<App />)
