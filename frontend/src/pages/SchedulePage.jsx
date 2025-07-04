import React, { useState, useEffect, useCallback, useMemo } from 'react'
import { fetchSchedules, downloadSchedule } from '../api'
import SlotExclusion from '../components/SlotExclusion'
import MetricSliders from '../components/MetricSliders'
import ScheduleCard from '../components/ScheduleCard'
import { usePinnedSchedules } from '../hooks/usePinnedButton'

export default function SchedulePage() {
  console.log('SchedulePage render start')
  const scheduleId = '20250624'
  console.log('SCHED SCHED  called', scheduleId)
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
  const handleDownload = useCallback((schedule) => {
    console.log('handleDownload called', schedule.basename)
    downloadSchedule(schedule.basename)
  }, [])

  const handlePin = useCallback(async (schedule) => {
    const scheduleId = schedule.id || schedule.name || schedule.basename
    console.log('handlePin called', scheduleId)

    // Set loading state for this specific schedule
    setPinLoadingStates(prev => ({ ...prev, [scheduleId]: true }))

    try {
      await togglePin(schedule)
      console.log('togglePin finished', scheduleId)
    } catch (error) {
      console.error('Error toggling pin:', error)
    } finally {
      setPinLoadingStates(prev => ({ ...prev, [scheduleId]: false }))
    }
  }, [togglePin])

  // Initial data fetch
  useEffect(() => {
    console.log('useEffect: fetchSchedules')
    fetchSchedules(scheduleId)
      .then(data => {
        console.log('fetchSchedules resolved', data)
        setAll(data)
        setFiltered(data)
      })
      .catch(err => {
        console.error('fetchSchedules error', err)
      })
  }, [scheduleId])

  // Memoized filtering logic
  const filteredSchedules = useMemo(() => {
    console.log('filteredSchedules useMemo called', { all, thresh, excl })
    if (!all.length) return []

    const result = all.filter((schedule) => {
      // First check: Remove schedules with fewer than 18 active slots
      const activeSlots = Object.values(schedule.columns).reduce((sum, value) => sum + (value === 1 ? 1 : 0), 0)

      if (activeSlots < 18) {
        console.log('filtered out by activeSlots', schedule)
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
          console.log('filtered out by slot exclusion', schedule)
          return false
        }
      }

      // Check metric thresholds
      for (const [key, threshold] of Object.entries(thresh)) {
        const value = schedule.metrics?.[key] ?? schedule.params?.[key]
        if (value !== undefined && Number(value) > threshold) {
          console.log('filtered out by threshold', { schedule, key, value, threshold })
          return false
        }
      }

      return true
    })

    console.log('filteredSchedules result', result)
    return result
  }, [all, thresh, excl])

  // Update filtered state when memoized result changes
  useEffect(() => {
    console.log('useEffect: setFiltered', filteredSchedules)
    setFiltered(filteredSchedules)
  }, [filteredSchedules])

  // Memoized schedule cards to prevent unnecessary re-renders
  const scheduleCards = useMemo(() => {
    console.log('scheduleCards useMemo called', filtered)
    return filtered.map(schedule => {
      const scheduleId = schedule.id || schedule.name || schedule.basename
      return (
        <ScheduleCard
          key={schedule.basename}
          schedule={schedule}
          onDownload={() => handleDownload(schedule)}
          onPin={handlePin}
          isPinned={isPinned(schedule)}
          pinLoading={pinLoadingStates[scheduleId] || false}
        />
      )
    })
  }, [filtered, handleDownload, handlePin, isPinned, pinLoadingStates])

  // Memoized values to prevent recalculation
  const numSlots = useMemo(() => {
    const n = all[0] ? Object.keys(all[0].columns).length : 24
    console.log('numSlots useMemo', n)
    return n
  }, [all])

  const pinnedSchedulesInfo = useMemo(() => {
    if (!pinnedSchedules || pinnedSchedules.length === 0) return null

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
  }, [pinnedSchedules?.length])

  // Display pin error if there is one
  const pinErrorDisplay = useMemo(() => {
    if (!pinError) return null

    return (
      <div style={{
        marginTop: '1rem',
        padding: '8px 12px',
        background: '#ffebee',
        borderRadius: 4,
        fontSize: '0.9em',
        color: '#c62828'
      }}>
        ⚠️ Pin error: {pinError}
      </div>
    )
  }, [pinError])

  console.log('SchedulePage render end')

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
        {pinErrorDisplay}
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