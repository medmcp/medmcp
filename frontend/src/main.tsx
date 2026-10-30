import { createRoot } from 'react-dom/client'
import '@fontsource/inter/400.css'
import '@fontsource/inter/500.css'
import '@fontsource/inter/600.css'
import '@fontsource/inter/700.css'
import '@fontsource/inter/800.css'
import '@fontsource/jetbrains-mono/400.css'
import '@fontsource/jetbrains-mono/500.css'
// KaTeX math styling. Imported through Vite so its woff2/ttf fonts are bundled
// into dist/assets (offline operation — never a CDN), like @fontsource above.
import 'katex/dist/katex.min.css'
import './index.css'
import App from './App.tsx'
import { canUseApi } from './api.ts'
import { ConfirmProvider } from './components/ConfirmDialog.tsx'
import LoginScreen from './components/LoginScreen.tsx'
import UnreachableScreen from './components/UnreachableScreen.tsx'

/** Show the workspace, the login screen, or why neither is available.
 *
 * Only a 401 is a password problem. A 403 is the origin guard or the agent
 * network and no password answers it, so sending those to the login screen left
 * the reader typing a correct password into a box that answered "Wrong password."
 */
async function boot() {
  const access = await canUseApi()
  // No StrictMode: its dev-only double-mount would open two chat WebSockets and
  // hence two vibe-acp sessions per page load.
  createRoot(document.getElementById('root')!).render(
    access.allowed ? (
      <ConfirmProvider>
        <App />
      </ConfirmProvider>
    ) : access.status === 401 ? (
      <LoginScreen />
    ) : (
      <UnreachableScreen status={access.status} detail={access.detail} />
    ),
  )
}

void boot()
