import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import license from 'rollup-plugin-license'

// Dev server proxies API + WebSocket traffic to the FastAPI workspace server
// (medmcp-workspace, port 8100). In production the same server serves dist/.
export default defineConfig({
  plugins: [
    react(),
    // Emit dist/LICENSES.txt with the copyright + full license text of every
    // dependency that ends up in the minified bundle — this is what carries the
    // attribution notices into the distributed (minified) frontend.
    license({
      thirdParty: {
        output: { file: 'dist/LICENSES.txt' },
      },
    }),
  ],
  server: {
    proxy: {
      '/api': 'http://127.0.0.1:8100',
      '/ws': { target: 'ws://127.0.0.1:8100', ws: true },
    },
  },
})
