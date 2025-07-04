import { API_BASE } from '../api'
export default function ScheduleCard({
  schedule: s,
  onDownload,
  onPin,
  isPinned = false,
  pinLoading = false
}) {
  const title = s.display_name || s.name

  // Try different image base paths

  const imgBase = `${API_BASE}/images`

  // Debug logging
  //console.log('Schedule basename:', s.basename)
  //console.log('Schedule image URL:', `${imgBase}/${s.basename}.png`)
  //console.log('Distribution image URL:', `${imgBase}/${s.basename}_dist.png`)

  const handleImageLoad = (imageType) => {
    console.log(`✅ Successfully loaded ${imageType} image`)
  }

  const handleImageError = (e, imageType) => {
    console.error(`❌ Failed to load ${imageType} image:`, e.target.src)
    e.target.style.opacity = 0.4

    // Set a static fallback image to avoid infinite loading loops
    e.target.onerror = null // Prevent further error handling
    e.target.src = '/fallback-image.png' // Make sure this image exists in your public folder
  }

  return (
    <div style={{
      background: '#fff',
      padding: 16,
      marginBottom: 16,
      borderRadius: 8,
      boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
      border: isPinned ? '2px solid #007bff' : '1px solid #eee'
    }}>
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        <h3 style={{ margin: 0, flex: 1 }}>{title}</h3>
        <div style={{ display: 'flex', gap: 8 }}>
          <button
            onClick={() => onPin(s)}
            disabled={pinLoading}
            style={{
              background: isPinned ? '#007bff' : '#f8f9fa',
              color: isPinned ? 'white' : '#333',
              border: isPinned ? '1px solid #007bff' : '1px solid #ccc',
              padding: '6px 12px',
              borderRadius: 4,
              cursor: pinLoading ? 'not-allowed' : 'pointer',
              opacity: pinLoading ? 0.6 : 1,
              transition: 'all 0.2s ease'
            }}
          >
            {pinLoading ? '⏳' : (isPinned ? '📌 Pinned' : '📌 Pin')}
          </button>
          <button
            onClick={() => onDownload(s)}
            style={{
              background: '#28a745',
              color: 'white',
              border: '1px solid #28a745',
              padding: '6px 12px',
              borderRadius: 4,
              cursor: 'pointer',
              transition: 'all 0.2s ease'
            }}
          >
            Download CSV
          </button>
        </div>
      </div>

      <div style={{ display: 'flex', gap: 16, marginTop: 12 }}>
        <div style={{
          background: '#eee',
          padding: 12,
          borderRadius: 4,
          flex: '1 1 30%',
          overflowY: 'visible',
          maxHeight: 'none'
        }}>
          <div style={{
            fontSize: '14px',
            fontWeight: 'bold',
            marginBottom: 8,
            color: '#333',
            borderBottom: '1px solid #ccc',
            paddingBottom: 4
          }}>
            Metrics
          </div>
          {Object.entries(s.metrics).map(([k, v]) => (
            <div key={k} style={{ marginBottom: 4 }}>
              <strong>{k}</strong>: {v}
            </div>
          ))}

          <div style={{
            fontSize: '14px',
            fontWeight: 'bold',
            marginTop: 16,
            marginBottom: 8,
            color: '#333',
            borderBottom: '1px solid #ccc',
            paddingBottom: 4
          }}>
            Parameters
          </div>
          {Object.entries(s.params || {}).map(([k, v]) => (
            <div key={k} style={{ marginBottom: 4 }}>
              <strong>{k}</strong>: {typeof v === 'number' ? v.toFixed(1) : v}
            </div>
          ))}
        </div>

        <img
          src={`${imgBase}/${s.basename}.png`}
          alt="Schedule plot"
          style={{ flex: '0 0 35%', maxWidth: '100%' }}
          onError={(e) => handleImageError(e, 'schedule')}
          onLoad={() => handleImageLoad('schedule')}
        />

        <img
          src={`${imgBase}/${s.basename}_dist.png`}
          alt="# done plot"
          style={{ flex: '0 0 35%', maxWidth: '100%' }}
          onError={(e) => handleImageError(e, 'distribution')}
          onLoad={() => handleImageLoad('distribution')}
        />
      </div>
    </div>
  )
}