// src/api.js

import axios from 'axios'


// In prod this is something like "https://your-eb-url"
// In dev it'll be undefined, so we fall back to empty string → relative URLs
const BASE = import.meta.env.VITE_API_URL || ''

export const API = axios.create({
  baseURL: BASE,
})

// (optional) if you ever want the “raw” base for window.open, etc.
export const API_BASE = BASE

// if your "api" routes are always under "/api", export that too:
// src/config.js
export const API_URL = import.meta.env.VITE_API_URL;

export function fetchSchedules(datePrefix) {
  return API.get(`/api/schedules/${datePrefix}`)
    .then(r => r.data)
}

export function downloadSchedule(basename) {
  window.open(`${BASE}/api/download/schedules/${basename}`, '_blank')
}
