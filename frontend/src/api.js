// src/api.js

import axios from 'axios'

// use VITE_API_URL || fallback
const BASE = import.meta.env.VITE_API_URL || 'http://localhost:5173'

export const API = axios.create({
  baseURL: BASE
})

// if your "api" routes are always under "/api", export that too:
export const API_BASE = `${BASE}/api`

export function fetchSchedules(datePrefix) {
  return API.get(`/api/schedules/${datePrefix}`)
    .then(r => r.data)
}

export function downloadSchedule(basename) {
  window.open(`${BASE}/api/download/schedules/${basename}`, '_blank')
}
