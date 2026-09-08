import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Builds straight into server/static so FastAPI can serve the dashboard at "/"
// without any change to the existing FL server logic.
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: '../server/static',
    emptyOutDir: true,
  },
  server: {
    proxy: {
      '/api': 'http://localhost:8000',
      '/plots': 'http://localhost:8000',
    },
  },
})
