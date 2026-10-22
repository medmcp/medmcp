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
import { isPaired, pair } from './api.ts'
import { ConfirmProvider } from './components/ConfirmDialog.tsx'
import PairingScreen from './components/PairingScreen.tsx'

/** Pair from a `?token=` link, then show the workspace or the pairing screen. */
async function boot() {
  const url = new URL(window.location.href)
  const token = url.searchParams.get('token')
  if (token) {
    // Out of the address bar and history either way; a wrong one just leaves
    // the browser unpaired.
    url.searchParams.delete('token')
    window.history.replaceState(null, '', url)
    await pair(token).catch(() => undefined)
  }
  const paired = await isPaired()
  // No StrictMode: its dev-only double-mount would open two chat WebSockets and
  // hence two vibe-acp sessions per page load.
  createRoot(document.getElementById('root')!).render(
    paired ? (
      <ConfirmProvider>
        <App />
      </ConfirmProvider>
    ) : (
      <PairingScreen />
    ),
  )
}

void boot()
