import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  // Use 127.0.0.1 explicitly — Node 18+ resolves 'localhost' to IPv6 ::1
  // first, but uvicorn binds IPv4 0.0.0.0 by default → ECONNREFUSED.
  const backend = env.VITE_API_URL || 'http://127.0.0.1:8001'

  return {
    plugins: [react()],
    server: {
      proxy: {
        '/api': { target: backend, changeOrigin: true },
        '/images': { target: backend, changeOrigin: true },
      },
    },
  }
})
