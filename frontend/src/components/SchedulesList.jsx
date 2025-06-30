
// ==========================================
// components/SchedulesList.jsx
import React, { useState } from 'react'
import ScheduleCard from '../components/ScheduleCard'
import { usePinnedSchedules } from '../hooks/usePinnedButton'

export default function SchedulesList({ schedules }) {
  // You'd get this from your auth system
  const userId = 'user123' // Replace with actual user ID

  const {
    pinnedSchedules,
    loading: pinLoading,
    error: pinError,
    isPinned,
    togglePin
  } = usePinnedSchedules(userId)

  const [pinLoadingStates, setPinLoadingStates] = useState({})

  const handlePin = async (schedule) => {
    const scheduleId = schedule.id || schedule.name || schedule.basename

    // Set loading state for this specific schedule
    setPinLoadingStates(prev => ({ ...prev, [scheduleId]: true }))

    try {
      await togglePin(schedule)
    } finally {
      setPinLoadingStates(prev => ({ ...prev, [scheduleId]: false }))
    }
  }

  const handleDownload = (schedule) => {
    // Your existing download logic
    console.log('Downloading:', schedule)
  }

  if (pinError) {
    console.error('Pin error:', pinError)
  }

  return (
    <div>
      <h2>All Schedules</h2>
      {schedules.map((schedule, index) => {
        const scheduleId = schedule.id || schedule.name || schedule.basename
        return (
          <ScheduleCard
            key={scheduleId || index}
            schedule={schedule}
            onDownload={() => handleDownload(schedule)}
            onPin={handlePin}
            isPinned={isPinned(schedule)}
            pinLoading={pinLoadingStates[scheduleId] || false}
          />
        )
      })}

      {pinnedSchedules.length > 0 && (
        <div style={{ marginTop: 32 }}>
          <h2>Pinned Schedules ({pinnedSchedules.length})</h2>
          {pinnedSchedules.map((pinnedSchedule) => (
            <div key={pinnedSchedule.schedule_id} style={{
              background: '#f8f9fa',
              padding: 12,
              marginBottom: 8,
              borderRadius: 4,
              border: '1px solid #007bff'
            }}>
              <strong>{pinnedSchedule.schedule_name}</strong>
              <div style={{ fontSize: '0.9em', color: '#666' }}>
                Pinned on: {new Date(pinnedSchedule.created_at).toLocaleDateString()}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}