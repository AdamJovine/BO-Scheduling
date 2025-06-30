import React, { useState, useEffect } from 'react'

export default function SlotExclusion({ numSlots, onChange }) {
  const [excluded, setExcluded] = useState([])
  const toggle = i => {
    setExcluded(prev =>
      prev.includes(i) ? prev.filter(x => x !== i) : [...prev, i]
    )
  }

  useEffect(() => { onChange(excluded) }, [excluded])
  console.log('excluded ', excluded)

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6,1fr)', gap: 8 }}>
      {Array.from({ length: numSlots }, (_, i) => i + 1).map(i => (
        <label key={i} style={{ textAlign: 'center' }}>
          <input
            type="checkbox"
            checked={excluded.includes(i)}
            onChange={() => toggle(i)}
          />
          <div>{i}</div>
        </label>
      ))}
    </div>
  )
}
