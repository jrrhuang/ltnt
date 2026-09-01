// CREATE MAP — radial zoom-LOD explorer. Self-contained.
import { useEffect, useRef, useState, useCallback, useMemo } from 'react'
import { startCreateMap, pollCreateMap } from './api'
import ForceMapView from './ForceMapView'
import ForceMap3DView from './ForceMap3DView'

const BASE_SIZE = 96        // depth-0 image size in world pixels
const ROOT_REVEAL = 0.8     // zoom at which roots are fully visible
const DEPTH_ZOOM_STEP = 2   // each deeper level needs this much more zoom

/**
 * Props:
 *   prompt      — string
 *   params      — optional overrides for CreateMapRequest
 *   onBack      — () => void
 *   onPick      — (image) => void; called when user clicks a node
 */
export default function CreateMapView({ prompt, params = {}, initialManifest = null, onManifestReady, onBack, onPick }) {
  // If we already have a manifest from a previous mount (lifted into App), go
  // straight to 'done' — skip the precompute entirely.
  const [status, setStatus] = useState(initialManifest ? 'done' : 'pending')
  const [progress, setProgress] = useState(initialManifest ? 100 : 0)
  const [stage, setStage] = useState(initialManifest ? 'Done' : 'Starting…')
  const [error, setError] = useState(null)
  const [manifest, setManifest] = useState(initialManifest)
  // Initial view honors the ?view= deep-link so a shared URL lands where it
  // says. Both map views work now (3D fogs in — see ForceMap3DView), so '3d'
  // is a first-class destination. Accepted: '3d' -> 3D fly-through; 'force' or
  // 'nebula' -> 2D NEBULA. Anything else (or absent) defaults to NEBULA.
  const [viewMode, setViewMode] = useState(() => {
    try {
      const v = (new URLSearchParams(window.location.search).get('view') || '').toLowerCase()
      if (v === '3d') return '3d'
      if (v === 'nebula' || v === 'force') return 'force'
      return 'force'
    } catch { return 'force' }
  })
  useEffect(() => {
    try { const u = new URL(window.location.href); u.searchParams.set('view', viewMode); window.history.replaceState({}, '', u) } catch {}
  }, [viewMode])
  // Guard against React StrictMode (dev) double-firing the mount effect, which
  // otherwise fires two POSTs to /api/create_map/generate and loads FLUX twice
  // on the GPU concurrently → OOM.
  const startedRef = useRef(false)

  // Kick off precompute (skip if we already have a cached manifest).
  // `startedRef` guards StrictMode double-mount; do NOT add a `cancelled` flag
  // — StrictMode fires the cleanup between mounts, silencing poll callbacks
  // and leaving the UI stuck at 0%.
  useEffect(() => {
    if (initialManifest) return
    if (startedRef.current) return
    startedRef.current = true
    ;(async () => {
      try {
        setStatus('running')
        const jobId = await startCreateMap(prompt, params)
        await pollCreateMap(jobId, (data) => {
          setProgress(data.progress ?? 0)
          setStage(data.stage ?? '')
          if (data.status === 'done' && data.result) {
            setManifest(data.result)
            setStatus('done')
            onManifestReady?.(data.result)
          }
        })
      } catch (e) {
        setError(String(e.message || e))
        setStatus('error')
      }
    })()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  if (status !== 'done') {
    return (
      <ProgressScreen
        status={status}
        progress={progress}
        stage={stage}
        error={error}
        prompt={prompt}
        onBack={onBack}
      />
    )
  }

  const tbBtn = { fontFamily: 'inherit', fontSize: 11, letterSpacing: '0.08em', textTransform: 'uppercase', color: '#cdd6f4', background: 'rgba(10,12,22,0.7)', border: '1px solid rgba(255,255,255,0.15)', borderRadius: 6, padding: '6px 12px', cursor: 'pointer', backdropFilter: 'blur(6px)' }
  return (
    <div style={{ position: 'absolute', inset: 0, background: '#05060c' }}>
      {(viewMode === '3d' || viewMode === 'cinematic')
        ? <ForceMap3DView manifest={manifest} cinematic={viewMode === 'cinematic'} onPick={onPick} />
        : <ForceMapView manifest={manifest} onPick={onPick} />}
      <div style={{ position: 'absolute', top: 14, left: 14, right: 14, display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 12, zIndex: 30, pointerEvents: 'none' }}>
        <button onClick={onBack} style={{ ...tbBtn, pointerEvents: 'auto', flexShrink: 0 }}>&larr; BACK</button>
        {/* Mode switcher. TOUR + RADIAL are HIDDEN for now (critic cmfix A2):
            they currently alias other views with no distinct working behavior,
            so only the views that actually work are offered. The array below is
            the single source of truth \u2014 re-add ['cinematic','\u25b7 TOUR'] and
            ['radial','\u2b21 RADIAL'] here to restore them once implemented. 3D is
            kept (Part B diagnoses its hang; button stays for the fix). */}
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', justifyContent: 'flex-end', pointerEvents: 'auto', maxWidth: '60%' }}>
          {[['force', '\u25c9 NEBULA'], ['3d', '\u25c8 3D']].map(([m, label]) => (
            <button key={m} onClick={() => setViewMode(m)}
              style={{ ...tbBtn, whiteSpace: 'nowrap', flexShrink: 0, opacity: viewMode === m ? 1 : 0.55, borderColor: viewMode === m ? 'rgba(150,200,255,0.7)' : 'rgba(255,255,255,0.15)' }}>{label}</button>
          ))}
        </div>
      </div>
    </div>
  )
}


function ProgressScreen({ status, progress, stage, error, prompt, onBack }) {
  return (
    <div style={{ height: '100vh', display: 'flex', flexDirection: 'column', fontFamily: 'system-ui' }}>
      <div style={{
        height: 56, display: 'flex', alignItems: 'center',
        justifyContent: 'space-between', padding: '0 20px',
        borderBottom: '1px solid #d0d0d0',
      }}>
        <button onClick={onBack} style={{ fontFamily: 'inherit', color: '#cdd6f4', border: '1px solid rgba(255,255,255,0.15)', borderRadius: 6, padding: '6px 14px', background: 'rgba(255,255,255,0.04)', cursor: 'pointer', fontSize: 11, letterSpacing: '0.08em' }}>
          &larr; BACK
        </button>
        <div style={{ textTransform: 'uppercase', color: '#9fb4e8', letterSpacing: '0.08em', fontSize: 11 }}>
          CREATE MAP &bull; {prompt}
        </div>
        <div style={{ width: 60 }} />
      </div>
      <main style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', flexDirection: 'column', gap: 20, background: '#fafafa' }}>
        <div style={{ fontSize: 13, color: '#555', letterSpacing: '0.08em', textTransform: 'uppercase' }}>
          {status === 'error' ? 'Error' : stage}
        </div>
        <div style={{ width: 400, height: 6, background: '#eee', borderRadius: 3, overflow: 'hidden' }}>
          <div style={{
            width: `${Math.max(2, Math.min(100, progress))}%`,
            height: '100%',
            background: status === 'error' ? '#c0392b' : '#1a1a1a',
            transition: 'width 0.3s ease',
          }} />
        </div>
        <div style={{ fontSize: 11, color: '#999', fontFamily: 'monospace' }}>
          {Math.round(progress)}%
        </div>
        {error && (
          <div style={{ fontSize: 11, color: '#c0392b', maxWidth: 500, textAlign: 'center' }}>
            {error}
          </div>
        )}
        <div style={{ fontSize: 10, color: '#aaa', marginTop: 40, maxWidth: 420, textAlign: 'center', lineHeight: 1.6 }}>
          Charting the prompt. This can take a minute or two — we're generating the whole tree up-front so you can pan and zoom through it afterwards.
        </div>
      </main>
    </div>
  )
}


function MapCanvas({ manifest, onBack, onPick }) {
  const canvasRef = useRef(null)
  const [canvasSize, setCanvasSize] = useState({ w: 0, h: 0 })
  const [zoom, setZoom] = useState(1)
  const [pan, setPan] = useState({ x: 0, y: 0 })
  // Mirrors of zoom/pan kept synchronously so the wheel-zoom anchor math always
  // reads the latest values, not React's possibly-stale render snapshot.
  const zoomRef = useRef(1)
  const panRef = useRef({ x: 0, y: 0 })
  const [panning, setPanning] = useState(false)
  const panStart = useRef(null)
  const draggedRef = useRef(false)
  const [hovered, setHovered] = useState(null)

  const images = manifest?.images || []
  const maxDepth = manifest?.max_depth || 0

  // Parent lookup for hover-ancestor highlight
  const byId = useMemo(() => {
    const m = new Map()
    for (const img of images) m.set(img.id, img)
    return m
  }, [images])

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

  // Wheel zoom around cursor — read+write refs so anchor math is never stale.
  useEffect(() => {
    const el = canvasRef.current
    if (!el) return
    const handler = e => {
      e.preventDefault()
      const rect = el.getBoundingClientRect()
      const sx = e.clientX - rect.left
      const sy = e.clientY - rect.top
      const z = zoomRef.current
      const p = panRef.current
      const delta = e.deltaY > 0 ? 1 / 1.025 : 1.025
      const nz = Math.max(0.3, Math.min(z * delta, 40))
      const np = {
        x: sx - (sx - p.x) * (nz / z),
        y: sy - (sy - p.y) * (nz / z),
      }
      zoomRef.current = nz
      panRef.current = np
      setZoom(nz)
      setPan(np)
    }
    el.addEventListener('wheel', handler, { passive: false })
    return () => el.removeEventListener('wheel', handler)
  }, [])

  function onMouseDown(e) {
    if (e.button !== 0) return
    setPanning(true)
    draggedRef.current = false
    panStart.current = { mx: e.clientX, my: e.clientY, px: panRef.current.x, py: panRef.current.y }
  }
  function onMouseMove(e) {
    if (!panning || !panStart.current) return
    const dx = e.clientX - panStart.current.mx
    const dy = e.clientY - panStart.current.my
    if (Math.abs(dx) + Math.abs(dy) > 6) draggedRef.current = true
    const np = { x: panStart.current.px + dx, y: panStart.current.py + dy }
    panRef.current = np
    setPan(np)
  }
  const onMouseUp = useCallback(() => {
    setPanning(false)
    panStart.current = null
  }, [])
  useEffect(() => {
    window.addEventListener('mouseup', onMouseUp)
    return () => window.removeEventListener('mouseup', onMouseUp)
  }, [onMouseUp])

  // LOD: reveal zoom per depth
  function revealZoom(d) {
    return ROOT_REVEAL * Math.pow(DEPTH_ZOOM_STEP, d)
  }
  function opacityFor(d) {
    // Expansive: EVERY depth visible immediately (many images, not sparse);
    // deeper generations more transparent but never hidden.
    return [1, 0.92, 0.85, 0.78, 0.72][Math.min(d, 4)]
  }

  // Per-depth world-pixel size; matches radius decay so deeper == smaller
  function sizeFor(d) {
    return Math.max(BASE_SIZE * Math.pow(0.8, d), 34)
  }

  // Ancestors of hovered node
  const ancestors = useMemo(() => {
    if (hovered == null) return new Set()
    const s = new Set()
    let cur = byId.get(hovered)
    while (cur && cur.parent != null) {
      s.add(cur.parent)
      cur = byId.get(cur.parent)
    }
    return s
  }, [hovered, byId])

  const hoveredChildren = useMemo(() => hovered == null ? new Set()
    : new Set(images.filter(i => i.parent === hovered).map(i => i.id)), [hovered, images])
  const visibleImages = images.filter(img => opacityFor(img.depth) > 0.01 || hoveredChildren.has(img.id))

  return (
    <div style={{ height: '100vh', display: 'flex', flexDirection: 'column', fontFamily: 'system-ui' }}>
      <div style={{
        height: 56, display: 'flex', alignItems: 'center',
        justifyContent: 'space-between', padding: '0 20px',
        borderBottom: '1px solid rgba(255,255,255,0.08)', background: '#070912', flexShrink: 0,
      }}>
        <button onClick={onBack} style={{ fontFamily: 'inherit', color: '#cdd6f4', border: '1px solid rgba(255,255,255,0.15)', borderRadius: 6, padding: '6px 14px', background: 'rgba(255,255,255,0.04)', cursor: 'pointer', fontSize: 11, letterSpacing: '0.08em' }}>
          &larr; BACK
        </button>
        <div style={{ textTransform: 'uppercase', color: '#9fb4e8', letterSpacing: '0.08em', fontSize: 11 }}>
          CREATE MAP &bull; {manifest?.prompt || ''} &bull; {visibleImages.length}/{images.length} &bull; zoom {Math.round(zoom * 100)}%
        </div>
        <button
          onClick={() => {
            zoomRef.current = 1
            panRef.current = { x: 0, y: 0 }
            setZoom(1)
            setPan({ x: 0, y: 0 })
          }}
          title='Back to the roots overview'
          style={{ fontFamily: 'inherit', color: '#bfe0ff', border: '1px solid rgba(150,200,255,0.5)', borderRadius: 6, padding: '6px 14px', background: 'rgba(120,175,255,0.12)', cursor: 'pointer', fontSize: 11, letterSpacing: '0.08em' }}
        >
          ⌂ OVERVIEW
        </button>
      </div>

      <main
        ref={canvasRef}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        style={{
          flex: 1, position: 'relative', overflow: 'hidden',
          cursor: panning ? 'grabbing' : 'grab',
          userSelect: 'none', background: '#04050a',
        }}
      >
        <div style={{
          position: 'absolute', inset: 0,
          transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
          transformOrigin: '0 0',
          // No `willChange: transform` — that promotes the subtree to a GPU
          // layer, the browser rasterizes once at layout size, and zoom-up
          // becomes a bitmap upscale (blur). ClusterView omits it for the
          // same reason. The CPU cost of repaint-on-zoom is negligible here.
        }}>
          {canvasSize.w > 0 && (
            <svg width={canvasSize.w} height={canvasSize.h}
              style={{ position: 'absolute', inset: 0, overflow: 'visible', pointerEvents: 'none' }}>
              {images.map(img => {
                if (img.parent == null) return null
                const pp = byId.get(img.parent)
                if (!pp) return null
                return <line key={'e' + img.id}
                  x1={pp.x * canvasSize.w} y1={pp.y * canvasSize.h}
                  x2={img.x * canvasSize.w} y2={img.y * canvasSize.h}
                  stroke='rgba(130,175,255,0.13)' strokeWidth={0.7} />
              })}
            </svg>
          )}
          {canvasSize.w > 0 && visibleImages.map(img => {
            const cx = img.x * canvasSize.w
            const cy = img.y * canvasSize.h
            const size = sizeFor(img.depth)
            const opacity = opacityFor(img.depth)
            const isHovered = hovered === img.id
            const isAncestor = ancestors.has(img.id)
            const isChild = hoveredChildren.has(img.id)
            return (
              <div
                key={img.id}
                onMouseEnter={() => setHovered(img.id)}
                onMouseLeave={() => setHovered(null)}
                onClick={(e) => {
                  e.stopPropagation()
                  if (draggedRef.current) return
                  onPick?.(img)
                }}
                style={{
                  position: 'absolute',
                  left: cx - size / 2,
                  top: cy - size / 2,
                  width: size,
                  height: size,
                  opacity: isHovered ? 1 : isChild ? Math.max(0.92, opacity) : (hovered != null ? opacity * 0.14 : opacity),
                  transition: 'opacity 0.25s ease, box-shadow 0.2s ease',
                  boxShadow: isHovered
                    ? '0 0 0 2px rgba(190,225,255,0.95), 0 0 28px rgba(130,185,255,0.85)'
                    : isChild
                      ? '0 0 0 1.5px rgba(170,210,255,0.8), 0 0 18px rgba(120,175,255,0.6)'
                      : img.depth === 0
                        ? '0 0 20px rgba(120,180,255,0.45)'
                        : '0 0 10px rgba(95,150,235,0.28)',
                  borderRadius: 6,
                  overflow: 'hidden',
                  zIndex: isHovered ? 100 : (20 - img.depth),
                  cursor: 'pointer',
                }}
              >
                <img
                  src={img.url}
                  alt={`n${img.id}`}
                  style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }}
                  draggable={false}
                />
              </div>
            )
          })}
        </div>

        <div style={{
          position: 'absolute', bottom: 16, right: 16,
          background: 'rgba(255,255,255,0.92)', padding: '8px 14px',
          fontSize: 10, letterSpacing: '0.08em',
          textTransform: 'uppercase', color: '#666',
          border: '1px solid #d0d0d0',
        }}>
          Scroll to zoom &bull; Drag to pan &bull; Click to pick
        </div>
      </main>
    </div>
  )
}
