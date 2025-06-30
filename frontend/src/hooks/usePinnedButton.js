// hooks/usePinnedSchedules.js
import { useState, useEffect, useCallback } from 'react'
import { API_BASE } from '../api'   // ← pull in the same BASE


export function usePinnedSchedules(userId) {
  const [pinnedSchedules, setPinnedSchedules] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  // Fetch pinned schedules from server
  const fetchPinnedSchedules = useCallback(async () => {
    if (!userId) return

    const url = `${API_BASE}/pinned-schedules/${userId}`
    console.log('🔍 Fetching pinned schedules from:', url)

    try {
      setLoading(true)
      setError(null)

      const response = await fetch(url, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
        // Add timeout
        signal: AbortSignal.timeout(10000) // 10 second timeout
      })

      console.log('📡 Response status:', response.status)
      console.log('📡 Response ok:', response.ok)

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const responseText = await response.text()
      console.log('📥 Raw response:', responseText)

      // Try to parse as JSON
      let data
      try {
        data = JSON.parse(responseText)
        console.log('📥 Parsed JSON data:', data)
      } catch (parseError) {
        console.error('❌ Failed to parse JSON:', parseError)
        console.log('📥 Response was HTML, not JSON. Check if route exists.')
        throw new Error('API returned HTML instead of JSON - route may not exist')
      }

      if (data.success) {
        setPinnedSchedules(data.pinned_schedules || [])
        console.log('✅ Set pinned schedules:', data.pinned_schedules?.length || 0)
      } else {
        setError(data.error || 'Failed to fetch pinned schedules')
        console.error('❌ API returned error:', data.error)
      }
    } catch (err) {
      console.error('❌ Error fetching pinned schedules:', err)
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [userId])

  // Pin a schedule
  const pinSchedule = async (schedule) => {
    try {
      const scheduleId = schedule.id || schedule.name || schedule.basename
      const url = `${API_BASE}/pinned-schedules/${userId}/${scheduleId}`

      console.log('📌 Pinning schedule:', scheduleId, 'to:', url)

      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          schedule_name: schedule.display_name || schedule.name || schedule.basename,
          schedule_data: schedule
        }),
        signal: AbortSignal.timeout(10000)
      })

      console.log('📡 Pin response status:', response.status)

      if (!response.ok) {
        const errorText = await response.text()
        console.error('❌ HTTP Error Response Body:', errorText)
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const responseText = await response.text()
      console.log('📥 Pin raw response:', responseText)

      let data
      try {
        data = JSON.parse(responseText)
        console.log('📥 Pin parsed data:', data)
      } catch (parseError) {
        console.error('❌ Pin failed to parse JSON:', parseError)
        throw new Error('Pin API returned HTML instead of JSON')
      }

      if (data.success) {
        console.log('✅ Successfully pinned schedule')
        await fetchPinnedSchedules()
        return true
      } else {
        setError(data.error || 'Failed to pin schedule')
        return false
      }
    } catch (err) {
      console.error('❌ Error pinning schedule:', err)
      setError(err.message)
      return false
    }
  }

  // Unpin a schedule
  const unpinSchedule = async (schedule) => {
    try {
      const scheduleId = schedule.id || schedule.name || schedule.basename
      const url = `${API_BASE}/pinned-schedules/${userId}/${scheduleId}`

      console.log('📌 Unpinning schedule:', scheduleId, 'from:', url)

      const response = await fetch(url, {
        method: 'DELETE',
        signal: AbortSignal.timeout(10000)
      })

      console.log('📡 Unpin response status:', response.status)

      if (!response.ok) {
        const errorText = await response.text()
        console.error('❌ Unpin error response:', errorText)
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const responseText = await response.text()
      console.log('📥 Unpin raw response:', responseText)

      let data
      try {
        data = JSON.parse(responseText)
        console.log('📥 Unpin parsed data:', data)
      } catch (parseError) {
        console.error('❌ Unpin failed to parse JSON:', parseError)
        throw new Error('Unpin API returned HTML instead of JSON')
      }

      if (data.success) {
        console.log('✅ Successfully unpinned schedule')
        await fetchPinnedSchedules()
        return true
      } else {
        setError(data.error || 'Failed to unpin schedule')
        return false
      }
    } catch (err) {
      console.error('❌ Error unpinning schedule:', err)
      setError(err.message)
      return false
    }
  }

  // Check if a schedule is pinned
  const isPinned = (schedule) => {
    const scheduleId = schedule.id || schedule.name || schedule.basename
    return pinnedSchedules.some(p => p.sched_id === scheduleId)
  }

  // Toggle pin status
  const togglePin = async (schedule) => {
    console.log('🔄 Toggling pin for schedule:', schedule.basename || schedule.name)
    if (isPinned(schedule)) {
      return await unpinSchedule(schedule)
    } else {
      return await pinSchedule(schedule)
    }
  }

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