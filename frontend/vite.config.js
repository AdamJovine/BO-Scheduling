import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const API_PORT = 8000;
const API_HOST = process.env.NODE_ENV === 'production'
  ? 'cornellschedulingteam.orie.cornell.edu'
  : '127.0.0.1';
const API_TARGET = `http://${API_HOST}:${API_PORT}`;

export default defineConfig({
  server: {
    host: '0.0.0.0',
    port: 3000,
    watch: { usePolling: true, interval: 100 },
    proxy: {
      '/api': {
        target: API_TARGET,
        changeOrigin: true,
        secure: false,
      },
      '/images': {
        target: API_TARGET,
        changeOrigin: true,
        secure: false,
      },
      '/download': {
        target: API_TARGET,
        changeOrigin: true,
        secure: false,
      },
    },
  },
  plugins: [react()],
  define: {
    global: 'globalThis',
  },
  build: {
    rollupOptions: {
      external: [],
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom'],
          slider: ['react-slider']
        }
      }
    },
    commonjsOptions: {
      include: [/node_modules/],
      transformMixedEsModules: true
    },
    target: 'es2022' // Ensure crypto.randomUUID is available
  },
  optimizeDeps: {
    include: ['react-slider', '@babel/runtime/helpers/inheritsLoose']
  }
})