import { useState, useEffect } from 'react'

// Pixel cat frames (2-frame walk cycle drawn with box-shadows).
// Shared component — same bar used on the first-generate flow, so every
// waiting state in the app looks and behaves identically.
const CAT_FRAMES = [
  `0 0 #000,
   4px 0 #000, 8px 0 #000, 12px 0 #000,
   0 4px #000, 4px 4px #fff, 8px 4px #000, 12px 4px #000, 16px 4px #000,
   0 8px #000, 4px 8px #000, 8px 8px #000, 12px 8px #000,
   4px 12px #000, 12px 12px #000,
   0 16px #000, 4px 16px #000, 8px 16px #000, 16px 16px #000, 20px 16px #000,
   0 20px #000, 8px 20px #000, 20px 20px #000`,
  `0 0 #000,
   4px 0 #000, 8px 0 #000, 12px 0 #000,
   0 4px #000, 4px 4px #000, 8px 4px #fff, 12px 4px #000, 16px 4px #000,
   0 8px #000, 4px 8px #000, 8px 8px #000, 12px 8px #000,
   4px 12px #000, 12px 12px #000,
   0 16px #000, 4px 16px #000, 16px 16px #000, 20px 16px #000,
   4px 20px #000, 12px 20px #000`,
]

export default function PixelCatProgress({ progress }) {
  const [frame, setFrame] = useState(0)
  useEffect(() => {
    const id = setInterval(() => setFrame(f => (f + 1) % 2), 300)
    return () => clearInterval(id)
  }, [])
  return (
    <div style={{ position: 'relative', height: '40px' }}>
      <style>{`
        @keyframes catbounce {
          0%, 100% { transform: translateY(0px); }
          50% { transform: translateY(-2px); }
        }
      `}</style>
      <div style={{
        position: 'absolute', bottom: '0', left: '0', right: '0',
        height: '2px', background: 'var(--border)',
      }} />
      <div style={{
        position: 'absolute', bottom: '0', left: '0',
        width: `${progress}%`, height: '2px', background: 'var(--accent)',
        transition: 'width 0.6s ease',
      }} />
      <div style={{
        position: 'absolute', bottom: '4px',
        left: `calc(${progress}% - 12px)`,
        transition: 'left 0.6s ease',
        animation: 'catbounce 0.6s ease infinite',
      }}>
        <div style={{
          width: '4px', height: '4px',
          boxShadow: CAT_FRAMES[frame],
          imageRendering: 'pixelated',
        }} />
      </div>
    </div>
  )
}
