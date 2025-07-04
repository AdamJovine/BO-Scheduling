// src/api.js

import axios from 'axios'

// Use relative URLs in production, specific URL only in development
const BASE = import.meta.env.PROD
  ? '' // Empty string for production - uses relative URLs through nginx proxy
  : (import.meta.env.VITE_API_URL || 'http://localhost:8000')

export const API = axios.create({
  baseURL: BASE,
  timeout: 10000, // 10 second timeout
})

// Add request interceptor for debugging
API.interceptors.request.use(request => {
  console.log('🚀 API Request:', request.method?.toUpperCase(), request.url)
  return request
})

// Add response interceptor for debugging
API.interceptors.response.use(
  response => {
    console.log('✅ API Response:', response.config.url, response.status)
    return response
  },
  error => {
    console.error('❌ API Error:', error.config?.url, error.message)
    return Promise.reject(error)
  }
)

// if your "api" routes are always under "/api", export that too:
export const API_BASE = `${BASE}/api`

export function fetchSchedules(datePrefix) {
  return API.get(`/api/schedules/${datePrefix}`)
    .then(r => r.data)
}

export function downloadSchedule(basename) {
  // Use relative URL for production, absolute for development
  const downloadUrl = import.meta.env.PROD
    ? `/api/download/schedules/${basename}`
    : `${BASE}/api/download/schedules/${basename}`

  window.open(downloadUrl, '_blank')
}

// Additional API functions with better error handling
export async function checkTables() {
  try {
    const response = await API.get('/api/debug/check-tables')
    return response.data
  } catch (error) {
    console.error('Failed to check tables:', error)
    throw error
  }
}

export async function getConfigs() {
  try {
    const response = await API.get('/api/slider-configs')
    return response.data
  } catch (error) {
    console.error('Failed to get configs:', error)
    throw error
  }
}