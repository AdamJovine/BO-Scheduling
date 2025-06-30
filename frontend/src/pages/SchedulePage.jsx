import React, { useState, useEffect, useCallback, useMemo } from 'react'
import { fetchSchedules, downloadSchedule } from '../api'
import SlotExclusion from '../components/SlotExclusion'
import MetricSliders from '../components/MetricSliders'
import ScheduleCard from '../components/ScheduleCard'
import { usePinnedSchedules } from '../hooks/usePinnedButton'

export default function SchedulePage() {
  const scheduleId = '20250620'
  const [all, setAll] = useState([])
  const [filtered, setFiltered] = useState([])
  const [excl, setExcl] = useState([])
  const [thresh, setThresh] = useState({})

  // Pin functionality
  const userId = 'user123' // Replace with actual user ID from your auth system

  const {
    pinnedSchedules,
    loading: pinLoading,
    error: pinError,
    isPinned,
    togglePin
  } = usePinnedSchedules(userId)

  const [pinLoadingStates, setPinLoadingStates] = useState({})

  // Memoized callbacks to prevent unnecessary re-renders
  const handleDownload = useCallback((basename) => {
    downloadSchedule(basename)
  }, [])

  const handlePin = useCallback(async (schedule) => {
    const scheduleId = schedule.id || schedule.name || schedule.basename

    // Set loading state for this specific schedule
    setPinLoadingStates(prev => ({ ...prev, [scheduleId]: true }))

    try {
      await togglePin(schedule)
    } finally {
      setPinLoadingStates(prev => ({ ...prev, [scheduleId]: false }))
    }
  }, [togglePin])

  // Initial data fetch
  useEffect(() => {
    fetchSchedules(scheduleId)
      .then(data => {
        setAll(data)
        setFiltered(data)
      })
      .catch(console.error)
  }, [scheduleId])

  // Memoized filtering logic
  const filteredSchedules = useMemo(() => {
    if (!all.length) return []

    const result = all.filter((schedule) => {
      // First check: Remove schedules with fewer than 18 active slots
      const activeSlots = Object.values(schedule.columns).reduce((sum, value) => sum + (value === 1 ? 1 : 0), 0)

      if (activeSlots < 18) {
        return false
      }

      // Check slot exclusions - handle both string and number keys
      if (excl.length > 0) {
        const excludedSlotsInSchedule = excl.filter(excludedSlot => {
          // Try both string and number versions of the slot
          const columnKeyString = excludedSlot.toString()
          const columnKeyNumber = Number(excludedSlot)

          const hasSlotString = schedule.columns[columnKeyString] === 1
          const hasSlotNumber = schedule.columns[columnKeyNumber] === 1
          const hasSlot = hasSlotString || hasSlotNumber

          return hasSlot
        })

        if (excludedSlotsInSchedule.length > 0) {
          return false
        }
      }

      // Check metric thresholds
      for (const [key, threshold] of Object.entries(thresh)) {
        const value = schedule.metrics?.[key] ?? schedule.params?.[key]
        if (value !== undefined && Number(value) > threshold) {
          return false
        }
      }

      return true
    })

    return result
  }, [all, thresh, excl])

  // Update filtered state when memoized result changes
  useEffect(() => {
    setFiltered(filteredSchedules)
  }, [filteredSchedules])

  // Memoized schedule cards to prevent unnecessary re-renders
  const scheduleCards = useMemo(() => {
    return filtered.map(schedule => {
      const scheduleId = schedule.id || schedule.name || schedule.basename
      return (
        <ScheduleCard
          key={schedule.basename}
          schedule={schedule}
          basename={schedule.basename}
          onDownload={handleDownload}
          onPin={handlePin}
          isPinned={isPinned(schedule)}
          pinLoading={pinLoadingStates[scheduleId] || false}
        />
      )
    })
  }, [filtered, handleDownload, handlePin, isPinned, pinLoadingStates])

  // Memoized values to prevent recalculation
  const numSlots = useMemo(() => {
    return all[0] ? Object.keys(all[0].columns).length : 24
  }, [all])

  const pinnedSchedulesInfo = useMemo(() => {
    if (pinnedSchedules.length === 0) return null

    return (
      <div style={{
        marginTop: '1rem',
        padding: '8px 12px',
        background: '#e3f2fd',
        borderRadius: 4,
        fontSize: '0.9em',
        color: '#1976d2'
      }}>
        📌 {pinnedSchedules.length} schedule{pinnedSchedules.length !== 1 ? 's' : ''} pinned
      </div>
    )
  }, [pinnedSchedules.length])

  return (
    <div style={{ display: 'flex', padding: 16 }}>
      <div style={{ width: '25%', marginRight: 16 }}>
        <h2>FILTER SCHEDULES</h2>

        <SlotExclusion numSlots={numSlots} onChange={setExcl} />

        <MetricSliders data={all} onChange={setThresh} />

        <div style={{ marginTop: '1rem', fontSize: '0.9em', color: '#666' }}>
          Showing {filtered.length} of {all.length} schedules
        </div>

        {pinnedSchedulesInfo}
      </div>

      <div style={{ flex: 1 }}>
        {filtered.length === 0
          ? <div>No schedules match the selected criteria.</div>
          : scheduleCards
        }
      </div>
    </div>
  )
}