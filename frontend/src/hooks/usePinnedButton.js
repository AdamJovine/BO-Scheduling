import { useState, useEffect, useCallback } from 'react'
import { API_BASE } from '../api'

export function usePinnedSchedules(userId) {
  const [pinnedSchedules, setPinnedSchedules] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  // Fetch pinned schedules from the API
  const fetchPinnedSchedules = useCallback(async () => {
    if (!userId) return

    try {
      setLoading(true)
      setError(null)

      const response = await fetch(`${API_BASE}/pinned-schedules/${userId}`)
      const data = await response.json()

      if (data.success) {
        setPinnedSchedules(data.pinned_schedules || [])
      } else {
        setError(data.error || 'Failed to fetch pinned schedules')
      }
    } catch (err) {
      console.error('Error fetching pinned schedules:', err)
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [userId, API_BASE])

  // Pin a schedule
  const pinSchedule = useCallback(async (schedule) => {
    if (!userId || !schedule) return

    try {
      setLoading(true)
      setError(null)

      // Use basename, id, or name as the schedule_id
      const scheduleId = schedule.basename || schedule.id || schedule.name

      const response = await fetch(`${API_BASE}/pinned-schedules/${userId}/${scheduleId}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          schedule_name: schedule.display_name || schedule.name || scheduleId,
          schedule_data: {
            basename: schedule.basename,
            metrics: schedule.metrics,
            params: schedule.params,
            columns: schedule.columns
          }
        })
      })

      const data = await response.json()

      if (data.success) {
        // Add to local state immediately for instant UI feedback
        const newPin = {
          id: Date.now(), // Temporary ID
          user_id: userId,
          sched_id: scheduleId,
          name: schedule.display_name || schedule.name || scheduleId,
          data: {
            basename: schedule.basename,
            metrics: schedule.metrics,
            params: schedule.params,
            columns: schedule.columns
          },
          created: new Date().toISOString()
        }
        setPinnedSchedules(prev => [newPin, ...prev])
      } else {
        setError(data.error || 'Failed to pin schedule')
      }
    } catch (err) {
      console.error('Error pinning schedule:', err)
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [userId, API_BASE])

  // Unpin a schedule
  const unpinSchedule = useCallback(async (schedule) => {
    if (!userId || !schedule) return

    try {
      setLoading(true)
      setError(null)

      const scheduleId = schedule.basename || schedule.id || schedule.name

      const response = await fetch(`${API_BASE}/pinned-schedules/${userId}/${scheduleId}`, {
        method: 'DELETE'
      })

      const data = await response.json()

      if (data.success) {
        // Remove from local state immediately for instant UI feedback
        setPinnedSchedules(prev =>
          prev.filter(pin => pin.sched_id !== scheduleId)
        )
      } else {
        setError(data.error || 'Failed to unpin schedule')
      }
    } catch (err) {
      console.error('Error unpinning schedule:', err)
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [userId, API_BASE])

  // Check if a schedule is pinned
  const isPinned = useCallback((schedule) => {
    if (!schedule || !pinnedSchedules.length) return false

    const scheduleId = schedule.basename || schedule.id || schedule.name
    return pinnedSchedules.some(pin => pin.sched_id === scheduleId)
  }, [pinnedSchedules])

  // Toggle pin status
  const togglePin = useCallback(async (schedule) => {
    if (isPinned(schedule)) {
      await unpinSchedule(schedule)
    } else {
      await pinSchedule(schedule)
    }
  }, [isPinned, pinSchedule, unpinSchedule])

  // Initial fetch when userId changes
  useEffect(() => {
    fetchPinnedSchedules()
  }, [fetchPinnedSchedules])

  return {
    pinnedSchedules,
    loading,
    error,
    pinSchedule,
    unpinSchedule,
    isPinned,
    togglePin,
    refetch: fetchPinnedSchedules
  }
}