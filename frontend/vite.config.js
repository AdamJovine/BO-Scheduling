// vite.config.js
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  server: {
    host: 'localhost',
    watch: { usePolling: true, interval: 100 },
    proxy: {
      '/api': {
        target: 'http://localhost:5000',
        changeOrigin: true,
      },
      '/images': {
        target: 'http://localhost:5000',
        changeOrigin: true,
      },
      '/download': {
        target: 'http://localhost:5000',
        changeOrigin: true,
      },
    },
    // Optional: also allow it in dev mode
    allowedHosts: ['byx2rxinsc.us-east-2.awsapprunner.com']
  },
  // This is picked up by `vite preview`
  preview: {
    allowedHosts: ['byx2rxinsc.us-east-2.awsapprunner.com']
  },
  plugins: [react()],
})
