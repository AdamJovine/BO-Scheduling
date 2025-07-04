import React from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'
import './crypto-polyfill.js';
const rootEl = document.getElementById('root')
if (!rootEl) throw new Error('Could not find #root element')

console.log('🔥 main.jsx bootstrapping')
const root = createRoot(rootEl)
root.render(<App />)
