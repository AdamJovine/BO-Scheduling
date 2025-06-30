import React, { useState, useEffect, useMemo, useRef } from 'react'
import '../App.css'
import ReactSlider from 'react-slider'

import { API_BASE } from '../api'

const sliderConfigAPI = {
  getConfigs: async () => {
    try {
      const response = await fetch(`${API_BASE}/slider-configs`)
      if (!response.ok) {
        throw new Error('Failed to fetch configurations')
      }
      const result = await response.json()
      return result
    } catch (error) {
      console.error('Error getting configurations:', error)
      throw error
    }
  },

  saveConfig: async (name, thresholds) => {
    try {
      const response = await fetch(`${API_BASE}/slider-configs`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, thresholds })
      })

      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.message || 'Failed to save configuration')
      }

      const result = await response.json()
      return result
    } catch (error) {
      console.error('Error saving configuration:', error)
      throw error
    }
  }
}

const sliderRecordingAPI = {
  initializeTables: async () => {
    try {
      const response = await fetch(`${API_BASE}/init-tables`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      })

      const responseText = await response.text()

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${responseText}`)
      }

      const result = JSON.parse(responseText)
      return result
    } catch (error) {
      console.error('Error initializing tables:', error)
      throw error
    }
  },

  checkTables: async () => {
    try {
      const response = await fetch(`${API_BASE}/debug/check-tables`)

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`)
      }

      const result = await response.json()
      return result
    } catch (error) {
      console.error('Error checking tables:', error)
      throw error
    }
  },

  recordInteraction: async (sessionId, sliderKey, value, minValue, maxValue) => {
    const payload = {
      session_id: sessionId,
      slider_key: sliderKey,
      value: value,
      min_value: minValue,
      max_value: maxValue
    }

    try {
      const response = await fetch(`${API_BASE}/slider-recordings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      })

      const responseText = await response.text()

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${responseText}`)
      }

      const result = JSON.parse(responseText)
      return result
    } catch (error) {
      console.error('Error recording interaction:', error)
      throw error
    }
  },

  recordBatch: async (sessionId, recordings) => {
    const payload = {
      session_id: sessionId,
      recordings: recordings
    }

    try {
      const response = await fetch(`${API_BASE}/slider-recordings/batch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      })

      const responseText = await response.text()

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${responseText}`)
      }

      const result = JSON.parse(responseText)
      return result
    } catch (error) {
      console.error('Error recording batch:', error)
      throw error
    }
  },

  getRecordings: async (sessionId = null, sliderKey = null, limit = null) => {
    const params = new URLSearchParams()
    if (sessionId) params.append('session_id', sessionId)
    if (sliderKey) params.append('slider_key', sliderKey)
    if (limit) params.append('limit', limit)

    const url = `${API_BASE}/slider-recordings?${params}`

    try {
      const response = await fetch(url)

      if (!response.ok) {
        throw new Error('Failed to fetch recordings')
      }

      const result = await response.json()
      return result
    } catch (error) {
      console.error('Error fetching recordings:', error)
      throw error
    }
  }
}

export default function MetricSliders({ data, onChange }) {
  // Store last non-empty keys for sliders
  const lastMetricKeys = useRef([])
  const lastParamKeys = useRef([])
  // Store initial thresholds to allow reset
  const initialThrRef = useRef({})

  // Generate a session ID for this component instance
  const sessionId = useRef(crypto.randomUUID())

  // Local state
  const [thr, setThr] = useState({})
  const [savedConfigs, setSavedConfigs] = useState([])
  const [configName, setConfigName] = useState('')
  const [loading, setLoading] = useState(false)
  const [message, setMessage] = useState('')
  const [recordingEnabled, setRecordingEnabled] = useState(true)
  const [tablesInitialized, setTablesInitialized] = useState(false)

  // Compute keys from data when available
  const metricKeys = useMemo(() => {
    const keys = data?.[0]?.metrics ? Object.keys(data[0].metrics) : []
    if (keys.length) {
      lastMetricKeys.current = keys
    }
    return lastMetricKeys.current
  }, [data])

  const paramKeys = useMemo(() => {
    const keys = data?.[0]?.params ? Object.keys(data[0].params) : []
    if (keys.length) {
      lastParamKeys.current = keys
    }
    return lastParamKeys.current
  }, [data])

  // Compute thresholds from given data
  const computeThresholds = (sourceData) => {
    const init = {}

    metricKeys.forEach(k => {
      const vals = sourceData.map(s => Number(s.metrics?.[k] ?? 0))
      init[k] = Math.max(...vals, 0)
    })

    paramKeys.forEach(k => {
      const vals = sourceData.map(s => Number(s.params?.[k] ?? 0))
      init[k] = Math.max(...vals, 0)
    })

    return init
  }

  // Get min/max values for a specific key
  const getSliderBounds = (key) => {
    if (!data || data.length === 0) {
      return { min: 0, max: initialThrRef.current[key] ?? 0 }
    }

    const values = data.map(s => Number(s.metrics?.[key] ?? s.params?.[key] ?? 0))
    return {
      min: Math.min(...values),
      max: Math.max(...values)
    }
  }

  // Initialize tables if needed
  const initializeTablesIfNeeded = async () => {
    try {
      const tableStatus = await sliderRecordingAPI.checkTables()

      if (!tableStatus.slider_recordings_exists) {
        setMessage('Initializing database tables...')
        await sliderRecordingAPI.initializeTables()
        setMessage('Database initialized successfully!')
        setTimeout(() => setMessage(''), 3000)
      }
      setTablesInitialized(true)
    } catch (error) {
      console.error('Failed to initialize database tables:', error)
      setMessage('Database initialization failed. Please try again.')
    }
  }

  // Record slider interaction
  const recordSliderChange = async (key, value) => {
    if (!recordingEnabled || !tablesInitialized) return

    try {
      const bounds = getSliderBounds(key)
      await sliderRecordingAPI.recordInteraction(
        sessionId.current,
        key,
        value,
        bounds.min,
        bounds.max
      )
    } catch (error) {
      console.error(`Failed to record slider change for ${key}:`, error)
    }
  }

  // Record all current slider positions
  const recordAllSliders = async () => {
    if (!recordingEnabled || !tablesInitialized) return

    try {
      const recordings = []
      const allKeys = [...metricKeys, ...paramKeys]

      allKeys.forEach(key => {
        const bounds = getSliderBounds(key)
        const value = thr[key] ?? bounds.max
        recordings.push({
          slider_key: key,
          value: value,
          min_value: bounds.min,
          max_value: bounds.max
        })
      })

      if (recordings.length > 0) {
        await sliderRecordingAPI.recordBatch(sessionId.current, recordings)
      }
    } catch (error) {
      console.error('Failed to record slider batch:', error)
    }
  }

  // API functions
  const loadSavedConfigs = async () => {
    try {
      const result = await sliderConfigAPI.getConfigs()
      const configs = result.configs || []
      setSavedConfigs(configs)
    } catch (err) {
      console.error('Failed to load configurations:', err)
    }
  }

  const handleSaveConfig = async () => {
    if (!configName.trim()) {
      setMessage('Please enter a configuration name')
      setTimeout(() => setMessage(''), 3000)
      return
    }

    try {
      setLoading(true)
      await sliderConfigAPI.saveConfig(configName.trim(), thr)
      setMessage('Configuration saved!')
      setConfigName('')
      await loadSavedConfigs()
      setTimeout(() => setMessage(''), 3000)
    } catch (err) {
      console.error('Error saving configuration:', err)
      setMessage(`Error: ${err.message}`)
      setTimeout(() => setMessage(''), 3000)
    } finally {
      setLoading(false)
    }
  }

  const handleLoadConfig = (config) => {
    setThr(config.thresholds)
    onChange(config.thresholds)
    setMessage(`Loaded: ${config.name}`)
    setTimeout(() => setMessage(''), 3000)

    // Record the loaded configuration
    if (recordingEnabled && tablesInitialized) {
      setTimeout(() => recordAllSliders(), 100)
    }
  }

  // Reset handler uses stored initial thresholds
  const handleReset = () => {
    const init = initialThrRef.current
    setThr(init)
    onChange(init)

    // Record the reset
    if (recordingEnabled && tablesInitialized) {
      setTimeout(() => recordAllSliders(), 100)
    }
  }

  // Initialize on component mount
  useEffect(() => {
    const initialize = async () => {
      await initializeTablesIfNeeded()
      loadSavedConfigs()
    }
    initialize()
  }, [])

  // Initialize thresholds when data becomes non-empty
  useEffect(() => {
    if (!data || data.length === 0) return

    const init = computeThresholds(data)
    initialThrRef.current = init
    setThr(init)
    onChange(init)

    // Record initial state
    if (recordingEnabled && tablesInitialized) {
      setTimeout(() => recordAllSliders(), 100)
    }
  }, [data, onChange, recordingEnabled, tablesInitialized])

  const renderSlider = key => {
    const bounds = getSliderBounds(key)

    if (bounds.min === bounds.max) {
      return null
    }

    const isDisabled = data.length === 0
    const currentValue = thr[key] ?? bounds.max

    return (
      <div key={key} className={isDisabled ? 'slider-disabled' : ''}>
        <label>{key}</label>
        <div>≤ {currentValue}</div>
        <div className="slider-container">
          <ReactSlider
            className="metric-slider"
            trackClassName="metric-slider-track"
            thumbClassName="metric-slider-thumb"
            min={bounds.min}
            max={bounds.max}
            value={currentValue}
            onChange={v => {
              setThr(prev => ({ ...prev, [key]: v }))
            }}
            onAfterChange={v => {
              const newThr = { ...thr, [key]: v }
              onChange(newThr)
              recordSliderChange(key, v)
            }}
            disabled={isDisabled}
          />
        </div>
      </div>
    )
  }

  if (!metricKeys.length && !paramKeys.length) {
    return <div>No metrics available</div>
  }

  return (
    <div>
      <div style={{ marginBottom: '1rem', padding: '0.5rem', backgroundColor: '#f5f5f5', borderRadius: '4px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', flexWrap: 'wrap' }}>
          <span>Session ID: <code>{sessionId.current.slice(0, 8)}...</code></span>
          <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <input
              type="checkbox"
              checked={recordingEnabled}
              onChange={e => setRecordingEnabled(e.target.checked)}
            />
            Record interactions
          </label>
          <button
            onClick={recordAllSliders}
            disabled={!recordingEnabled || !tablesInitialized}
            style={{ fontSize: '0.8rem', opacity: (recordingEnabled && tablesInitialized) ? 1 : 0.5 }}
          >
            Record Current State
          </button>
          <button
            onClick={initializeTablesIfNeeded}
            style={{ fontSize: '0.8rem' }}
          >
            Initialize DB Tables
          </button>
          {tablesInitialized && (
            <span style={{ color: 'green', fontSize: '0.8rem' }}>✓ DB Ready</span>
          )}
        </div>
      </div>

      <h3>Metric thresholds</h3>
      {metricKeys.map(renderSlider)}

      <h3>Parameter thresholds</h3>
      {paramKeys.map(renderSlider)}

      <div style={{ marginTop: '1rem', padding: '1rem', border: '1px solid #ddd', borderRadius: '4px' }}>
        <h4>Save & Load Configurations</h4>

        {/* Message */}
        {message && (
          <div style={{
            marginBottom: '0.5rem',
            padding: '0.25rem',
            backgroundColor: message.startsWith('Error') ? '#fee' : '#efe',
            color: message.startsWith('Error') ? 'red' : 'green',
            borderRadius: '3px'
          }}>
            {message}
          </div>
        )}

        {/* Save New Configuration */}
        <div style={{ marginBottom: '1rem' }}>
          <input
            type="text"
            placeholder="Configuration name"
            value={configName}
            onChange={e => setConfigName(e.target.value)}
            style={{ marginRight: '0.5rem', padding: '0.25rem', width: '150px' }}
          />
          <button
            onClick={handleSaveConfig}
            disabled={loading || !configName.trim()}
            style={{ opacity: (!configName.trim() || loading) ? 0.6 : 1 }}
          >
            {loading ? 'Saving...' : 'Save Current'}
          </button>
        </div>

        {/* Saved Configurations List */}
        {savedConfigs.length > 0 && (
          <div>
            <h5>Saved Configurations ({savedConfigs.length})</h5>
            <div style={{ maxHeight: '150px', overflowY: 'auto', border: '1px solid #eee', borderRadius: '3px' }}>
              {savedConfigs.map((config) => (
                <div key={config.id} style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  padding: '0.5rem',
                  borderBottom: '1px solid #eee'
                }}>
                  <span>{config.name}</span>
                  <button
                    onClick={() => handleLoadConfig(config)}
                    style={{ fontSize: '0.8rem' }}
                  >
                    Load
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}

        {savedConfigs.length === 0 && (
          <div style={{ color: '#666', fontStyle: 'italic' }}>
            No saved configurations yet
          </div>
        )}
      </div>

      <button onClick={handleReset} style={{ marginTop: '1rem' }}>
        Reset thresholds
      </button>
    </div>
  )
}