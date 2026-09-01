import { useRef, useState, useEffect, useCallback } from 'react'

const IMG_BASE_SIZE = 96

/**
 * ExploreView — semantic Google Maps for latent space.
 *
 * All images pre-computed. Zoom level reveals deeper tree levels:
 *   zoom < 1.5  → depth 0 only (roots, large)
 *   zoom < 3    → depth 0-1
 *   zoom < 5    → depth 0-2
 *   zoom < 8    → depth 0-3
 *   zoom >= 8   → all depths
 *
 * Parents shrink as you zoom in past their reveal threshold.
 */
export default function ExploreView({ manifest, onBack }) {
  const canvasRef = useRef(null)
  const [canvasSize, setCanvasSize] = useState({ w: 0, h: 0 })
  const [zoom, setZoom] = useState(1)
  const [pan, setPan] = useState({ x: 0, y: 0 })
  const [panning, setPanning] = useState(false)
  const panStart = useRef(null)
  const [hoveredId, setHoveredId] = useState(null)

  const images = manifest?.images || []
  const maxDepth = manifest?.max_depth || 4

  // Zoom thresholds per depth level
  const zoomThresholds = []
  for (let d = 0; d <= maxDepth; d++) {
    zoomThresholds.push(1 + d * 1.5)  // depth 0: 1.0, depth 1: 2.5, depth 2: 4.0, etc.
  }

  // Which depth levels are visible at current zoom
  const visibleMaxDepth = zoomThresholds.findIndex(t => zoom < t)
  const maxVisibleDepth = visibleMaxDepth === -1 ? maxDepth : Math.max(0, visibleMaxDepth - 1)

  // Canvas resize
  useEffect(() => {
    const el = canvasRef.current
    if (!el) return
    const ro = new ResizeObserver(entries => {
      const { width, height } = entries[0].contentRect
      setCanvasSize({ w: width, h: height })
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  // Zoom handler
  useEffect(() => {
    const el = canvasRef.current
    if (!el) return
    const handler = e => {
      e.preventDefault()
      const delta = e.deltaY > 0 ? 0.9 : 1.1
      setZoom(z => Math.max(0.3, Math.min(z * delta, 20)))
    }
    el.addEventListener('wheel', handler, { passive: false })
    return () => el.removeEventListener('wheel', handler)
  }, [])

  // Pan handlers
  function onMouseDown(e) {
    if (e.button !== 0) return
    const rect = canvasRef.current.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top
    setPanning(true)
    panStart.current = { x: x - pan.x, y: y - pan.y }
  }

  function onMouseMove(e) {
    if (!panning || !panStart.current) return
    const rect = canvasRef.current.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top
    setPan({ x: x - panStart.current.x, y: y - panStart.current.y })
  }

  const onMouseUp = useCallback(() => {
    setPanning(false)
    panStart.current = null
  }, [])

  useEffect(() => {
    window.addEventListener('mouseup', onMouseUp)
    return () => window.removeEventListener('mouseup', onMouseUp)
  }, [onMouseUp])

  // Image size based on depth and zoom
  function getImageSize(depth) {
    // Root images are largest, deeper images smaller
    const depthScale = 1.0 - 0.5 * (depth / Math.max(maxDepth, 1))
    return IMG_BASE_SIZE * depthScale
  }

  // Opacity: fade in as zoom crosses the reveal threshold
  function getOpacity(depth) {
    if (depth === 0) return 1
    const threshold = zoomThresholds[depth]
    const prevThreshold = zoomThresholds[depth - 1] || 1
    if (zoom >= threshold) return 1
    if (zoom < prevThreshold) return 0
    // Fade in between prev and current threshold
    return Math.min(1, (zoom - prevThreshold) / (threshold - prevThreshold))
  }

  // Filter visible images
  const visibleImages = images.filter(img => {
    const opacity = getOpacity(img.depth)
    return opacity > 0.01
  })

  return (
    <div style={{ height: '100vh', display: 'flex', flexDirection: 'column', fontFamily: 'system-ui' }}>
      {/* Top bar */}
      <div style={{
        height: '56px', display: 'flex', alignItems: 'center',
        justifyContent: 'space-between', padding: '0 20px',
        borderBottom: '1px solid #d0d0d0', flexShrink: 0,
      }}>
        <button onClick={onBack} style={{
          border: '1px solid #1a1a1a', padding: '6px 14px',
          background: 'none', cursor: 'pointer',
        }}>
          &larr; BACK
        </button>
        <div style={{
          textTransform: 'uppercase', color: '#888',
          letterSpacing: '0.08em', fontSize: '11px',
        }}>
          EXPLORE &bull; {manifest?.prompt || ''} &bull; {visibleImages.length}/{images.length} visible &bull; zoom {Math.round(zoom * 100)}%
        </div>
        <div style={{ fontSize: '10px', color: '#aaa' }}>
          Depth: {maxVisibleDepth}/{maxDepth}
        </div>
      </div>

      {/* Canvas */}
      <main
        ref={canvasRef}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        style={{
          flex: 1, position: 'relative', overflow: 'hidden',
          cursor: panning ? 'grabbing' : 'grab',
          userSelect: 'none', background: '#fafafa',
        }}
      >
        <div style={{
          position: 'absolute', inset: 0,
          transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
          transformOrigin: '0 0',
        }}>
          {canvasSize.w > 0 && visibleImages.map(img => {
            const cx = img.x * canvasSize.w
            const cy = img.y * canvasSize.h
            const imgSize = getImageSize(img.depth)
            const opacity = getOpacity(img.depth)
            const isHovered = hoveredId === img.id

            return (
              <div
                key={img.id}
                onMouseEnter={() => setHoveredId(img.id)}
                onMouseLeave={() => setHoveredId(null)}
                style={{
                  position: 'absolute',
                  left: cx - imgSize / 2,
                  top: cy - imgSize / 2,
                  width: imgSize,
                  height: imgSize,
                  opacity,
                  transition: 'opacity 0.3s ease',
                  boxShadow: isHovered
                    ? '0 0 0 3px #333, 0 4px 12px rgba(0,0,0,0.3)'
                    : img.depth === 0
                      ? '0 2px 8px rgba(0,0,0,0.2)'
                      : '0 1px 4px rgba(0,0,0,0.1)',
                  borderRadius: '2px',
                  overflow: 'hidden',
                  zIndex: img.depth === 0 ? 10 : 5 - img.depth,
                  pointerEvents: 'auto',
                }}
              >
                <img
                  src={img.url}
                  alt={`img-${img.id}`}
                  style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }}
                  draggable={false}
                />
              </div>
            )
          })}
        </div>

        {/* Zoom info overlay */}
        <div style={{
          position: 'absolute', bottom: '16px', right: '16px',
          background: 'rgba(255,255,255,0.9)', padding: '8px 14px',
          fontSize: '10px', letterSpacing: '0.08em',
          textTransform: 'uppercase', color: '#666',
          border: '1px solid #d0d0d0',
        }}>
          Scroll to zoom &bull; Drag to pan &bull; Zoom in to reveal children
        </div>
      </main>
    </div>
  )
}
