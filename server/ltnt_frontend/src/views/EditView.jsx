import { useState, useRef, useEffect, useMemo, useCallback } from 'react'
import { editRegion, getEffects, compositeLayerStack } from '../api'
import CatLoader from '../components/CatLoader'
import { playRandomMeow } from '../utils/meow'
import Tour, { useTour, TourToggle } from '../Tour'

const EDIT_TOUR_STEPS = [
  {
    title: 'Polish your image',
    body: "Almost there! Here you can fine-tune one image at a time — change part of it, try a new style, sharpen it up, or remove the background. Every change is undoable.",
    placement: 'center',
  },
  {
    target: '[data-tour="image-grid"]',
    title: 'Your picks',
    body: "These are the images you chose. Click any of them to switch between them — each one keeps its own history.",
    placement: 'left',
  },
  {
    target: '[data-tour="operations"]',
    title: 'What would you like to do?',
    body: "Change a specific part of the image, try a few small variations, sharpen it up, or remove the background. Pick one, then use the canvas.",
    placement: 'right',
  },
  {
    target: '[data-tour="effects"]',
    title: 'Quick styles',
    body: "One-click style changes. Hover any effect to preview it, then click to apply.",
    placement: 'right',
  },
  {
    target: '[data-tour="export-btn"]',
    title: 'Save it',
    body: "When you're happy with the result, click here to download the image to your computer.",
    placement: 'bottom',
  },
]

// ── Constants ─────────────────────────────────────────────────────────────────

const OPERATIONS = [
  { id: 'edit',      label: 'EDIT'         },
  { id: 'vary',      label: 'VARY ×3'     },
  { id: 'upscale',   label: 'UPSCALE ×2'  },
  { id: 'remove_bg', label: 'REMOVE BG'   },
  { id: 'effect',    label: 'ADD EFFECT'  },
]

const MIN_DRAG_PX = 10
let _lid = 0
const genId = () => `l${++_lid}`

// ── Small helpers ─────────────────────────────────────────────────────────────

function SectionLabel({ children }) {
  return (
    <div style={{
      fontSize: '10px', letterSpacing: '0.08em', textTransform: 'uppercase',
      color: '#888', padding: '8px 12px 4px',
    }}>
      {children}
    </div>
  )
}

function EyeBtn({ visible, onClick }) {
  return (
    <button
      onClick={e => { e.stopPropagation(); onClick() }}
      title={visible ? 'Hide layer' : 'Show layer'}
      style={{
        border: 'none', background: 'none', padding: '0 3px',
        cursor: 'pointer', color: visible ? '#1a1a1a' : '#ccc',
        fontSize: '11px', lineHeight: 1, flexShrink: 0,
      }}
    >
      {visible ? '●' : '○'}
    </button>
  )
}

// ── Main component ────────────────────────────────────────────────────────────
//
// History model
// ─────────────
// histState[imageIdx] = { entries: Array<{src: string, layers: Layer[]}>, ptr: number }
//
// `src` in each entry is the pre-composited flat data URL stored at push time.
// `ptr` is a non-truncating pointer — old entries are never deleted.
//
// Layer shapes
// ────────────
// { id, type:'original', src, visible }
// { id, type:'effect',   effectName, editedB64, intensity, visible }
// { id, type:'region',   prompt, editedB64, visible,
//     selectionType:'rect'|'lasso',
//     region:{x,y,w,h}                         ← rect only
//     normalizedRegions:[{points:[{x,y},...]}]  ← lasso only
//     normalizedBbox:{x,y,w,h}                  ← lasso only }

export default function EditView({ images = [], onBack, tourMode = false, onTourModeChange = () => {} }) {
  const [imageIdx, setImageIdx] = useState(0)

  // ── Stable refs (never stale inside callbacks/timers) ─────────────────────
  const imageIdxRef     = useRef(0)
  const currentLayersRef = useRef([])
  useEffect(() => { imageIdxRef.current = imageIdx }, [imageIdx])

  // ── History ───────────────────────────────────────────────────────────────
  const [histState, setHistState] = useState(() =>
    Object.fromEntries(images.map((img, i) => {
      const src = img.src || img.url
      return [
        i,
        { entries: [{ src, layers: [{ id: 'orig', type: 'original', src, visible: true }] }], ptr: 0 },
      ]
    }))
  )

  const imgHist     = histState[imageIdx] ?? { entries: [], ptr: 0 }
  const activeEntry = imgHist.entries[imgHist.ptr] ?? imgHist.entries[imgHist.entries.length - 1]
  const currentLayers = useMemo(() => activeEntry?.layers ?? [], [activeEntry])
  useEffect(() => { currentLayersRef.current = currentLayers }, [currentLayers])

  // Push to history. Returns the composited src so callers can use it
  // immediately without waiting for a re-render.
  async function pushEntry(newLayers, imgIdx) {
    const src = await compositeLayerStack(newLayers)
    if (!src) return null
    const entry = { src, layers: newLayers }
    setHistState(s => {
      const cur = s[imgIdx] ?? { entries: [], ptr: -1 }
      const next = [...cur.entries, entry]
      return { ...s, [imgIdx]: { entries: next, ptr: next.length - 1 } }
    })
    return src
  }

  function jumpToEntry(idx) {
    setHistState(s => ({ ...s, [imageIdx]: { ...s[imageIdx], ptr: idx } }))
  }

  function undo() {
    setHistState(s => {
      const cur = s[imageIdx]
      if (!cur || cur.ptr <= 0) return s
      return { ...s, [imageIdx]: { ...cur, ptr: cur.ptr - 1 } }
    })
  }

  function redo() {
    setHistState(s => {
      const cur = s[imageIdx]
      if (!cur || cur.ptr >= cur.entries.length - 1) return s
      return { ...s, [imageIdx]: { ...cur, ptr: cur.ptr + 1 } }
    })
  }

  // ── Pending effect ────────────────────────────────────────────────────────
  // An uncommitted effect that the user is adjusting via the intensity slider.
  // Preview is rendered purely in JSX (base img + overlay img at CSS opacity).
  // No canvas composition needed until commit — that was the double-effect bug.
  const [pendingLayer,     setPendingLayer]     = useState(null)
  const [pendingIntensity, setPendingIntensity] = useState(80)
  const pendingLayerRef = useRef(null)
  const pendingIntRef   = useRef(80)
  const autoCommitTimer = useRef(null)
  useEffect(() => { pendingLayerRef.current = pendingLayer   }, [pendingLayer])
  useEffect(() => { pendingIntRef.current   = pendingIntensity }, [pendingIntensity])

  // ── Committed-layer intensity adjustment (from /FILTER LAYERS sliders) ────
  const [adjustLayerId,   setAdjustLayerId]   = useState(null)
  const [adjustIntensity, setAdjustIntensity] = useState(null)
  const adjustLayerIdRef   = useRef(null)
  const adjustIntensityRef = useRef(null)
  const adjustCommitTimer  = useRef(null)
  useEffect(() => { adjustLayerIdRef.current   = adjustLayerId   }, [adjustLayerId])
  useEffect(() => { adjustIntensityRef.current = adjustIntensity }, [adjustIntensity])

  // ── Display src ───────────────────────────────────────────────────────────
  // Normally = activeEntry.src (the pre-composited stored result).
  // During committed-layer intensity adjustment: recomposes live via canvas.
  // During pending effect: stays as activeEntry.src — the JSX overlay handles preview.
  const [displaySrc, setDisplaySrc] = useState(() =>
    Object.fromEntries(images.map((img, i) => [i, img.src || img.url]))
  )

  useEffect(() => {
    let cancelled = false

    async function compute() {
      const base = activeEntry?.src ?? null

      // Pending effect: JSX overlay handles preview, no canvas needed here.
      if (pendingLayer) {
        if (!cancelled) setDisplaySrc(d => ({ ...d, [imageIdx]: base }))
        return
      }

      // No layer adjustment active — use the stored flat result directly.
      if (adjustLayerId === null || adjustIntensity === null) {
        if (!cancelled) setDisplaySrc(d => ({ ...d, [imageIdx]: base }))
        return
      }

      // Committed-layer intensity adjustment: recompose with overridden intensity.
      const layers = currentLayers.map(l =>
        l.id === adjustLayerId ? { ...l, intensity: adjustIntensity } : l
      )
      const src = await compositeLayerStack(layers)
      if (!cancelled && src) setDisplaySrc(d => ({ ...d, [imageIdx]: src }))
    }

    compute()
    return () => { cancelled = true }
  }, [currentLayers, activeEntry?.src, pendingLayer, adjustLayerId, adjustIntensity, imageIdx])

  // The src actually rendered on the canvas.
  const currentSrc = displaySrc[imageIdx] ?? images[imageIdx]?.url

  // ── Intensity: pending effect ─────────────────────────────────────────────
  function handlePendingIntensity(value) {
    setPendingIntensity(value)
    pendingIntRef.current = value
    clearTimeout(autoCommitTimer.current)
    autoCommitTimer.current = setTimeout(async () => {
      const pl  = pendingLayerRef.current
      const idx = imageIdxRef.current
      const layers = currentLayersRef.current
      if (!pl) return
      const finalLayer = { ...pl, intensity: pendingIntRef.current, visible: true }
      await pushEntry([...layers, finalLayer], idx)
      setPendingLayer(null)
      setPendingIntensity(80)
    }, 800)
  }

  // ── Intensity: committed effect layer (right panel slider) ────────────────
  function handleLayerIntensity(layerId, value) {
    setAdjustLayerId(layerId)
    setAdjustIntensity(value)
    clearTimeout(adjustCommitTimer.current)
    adjustCommitTimer.current = setTimeout(() => {
      const idx    = imageIdxRef.current
      const layers = currentLayersRef.current
      const val    = adjustIntensityRef.current
      const newLayers = layers.map(l => l.id === layerId ? { ...l, intensity: val } : l)
      pushEntry(newLayers, idx)
      setAdjustLayerId(null)
      setAdjustIntensity(null)
    }, 800)
  }

  // ── Toggle layer visibility (undoable) ────────────────────────────────────
  function toggleLayerVisibility(layerId) {
    const newLayers = currentLayers.map(l =>
      l.id === layerId ? { ...l, visible: !l.visible } : l
    )
    pushEntry(newLayers, imageIdx)
  }

  // ── Keyboard shortcuts ────────────────────────────────────────────────────
  useEffect(() => {
    function onKey(e) {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return
      if ((e.ctrlKey || e.metaKey) && e.key === 'z') {
        e.preventDefault()
        e.shiftKey ? redo() : undo()
        return
      }
      if (e.key === 'r' || e.key === 'R') { setSelectionTool('rect');  setSelection(null) }
      if (e.key === 'l' || e.key === 'L') { setSelectionTool('lasso'); setSelection(null) }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [imageIdx]) // re-bind when imageIdx changes so undo/redo target correct slot

  // ── Operation mode ────────────────────────────────────────────────────────
  const [mode, setMode]             = useState('edit')
  const [effectName, setEffectName] = useState(null)
  const [effects, setEffects]       = useState([])
  const [effectsLoading, setEffectsLoading] = useState(false)
  const tour = useTour('ltnt.tour.edit', { autoOpen: tourMode })

  useEffect(() => {
    if (mode !== 'effect' || effects.length > 0) return
    setEffectsLoading(true)
    getEffects()
      .then(list => setEffects(Array.isArray(list) ? list : []))
      .catch(() => setEffects([]))
      .finally(() => setEffectsLoading(false))
  }, [mode])

  // ── Selection tools ───────────────────────────────────────────────────────
  // Rect:  { type:'rect',  x0, y0, x1, y1 }       — pixel coords in frame
  // Lasso: { type:'lasso', regions:[{points:[{x,y},...]},...] }
  //
  // Shift+draw (rect)  → expands the existing bounding box
  // Shift+draw (lasso) → appends a new polygon to regions[]

  const frameRef           = useRef(null)
  const [selectionTool, setSelectionTool] = useState('rect')
  const [drag, setDrag]           = useState(null)
  const [selection, setSelection] = useState(null)
  const selectionRef   = useRef(null)
  useEffect(() => { selectionRef.current = selection }, [selection])

  const lassoPtsRef     = useRef([])
  const isLassoDrawing  = useRef(false)
  const isAdditiveRef   = useRef(false)
  const additivePrevRef = useRef(null)
  const [lassoLive, setLassoLive] = useState(false) // true while finger is down

  function frameXY(e) {
    const r = frameRef.current.getBoundingClientRect()
    return {
      x: Math.max(0, Math.min(e.clientX - r.left, r.width)),
      y: Math.max(0, Math.min(e.clientY - r.top,  r.height)),
    }
  }

  function onMouseDown(e) {
    if (e.button !== 0 || isEditing || !currentSrc || mode !== 'edit' || pendingLayer) return
    e.preventDefault()
    setError(null)
    const pos = frameXY(e)

    if (selectionTool === 'lasso') {
      const additive = e.shiftKey && selectionRef.current?.type === 'lasso'
      isAdditiveRef.current  = additive
      isLassoDrawing.current = true
      lassoPtsRef.current    = [pos]
      setLassoLive(true)
      // Clear committed selection immediately for non-additive draw,
      // so the old polygon doesn't show underneath the new stroke.
      if (!additive) setSelection(null)
    } else {
      const additive = e.shiftKey && selectionRef.current?.type === 'rect'
      additivePrevRef.current = additive ? selectionRef.current : null
      if (!additive) setSelection(null)
      setDrag({ x0: pos.x, y0: pos.y, x1: pos.x, y1: pos.y })
    }
  }

  function onMouseMove(e) {
    if (selectionTool === 'lasso' && isLassoDrawing.current) {
      lassoPtsRef.current.push(frameXY(e))
      // Force re-render every other point to update the live polyline.
      if (lassoPtsRef.current.length % 2 === 0) setLassoLive(v => !v)
    } else if (drag) {
      const { x, y } = frameXY(e)
      setDrag(d => ({ ...d, x1: x, y1: y }))
    }
  }

  const onMouseUp = useCallback(() => {
    if (isLassoDrawing.current) {
      isLassoDrawing.current = false
      setLassoLive(false)
      if (lassoPtsRef.current.length > 4) {
        const newRegion = { points: [...lassoPtsRef.current] }
        const prev = selectionRef.current
        if (isAdditiveRef.current && prev?.type === 'lasso') {
          setSelection({ type: 'lasso', regions: [...prev.regions, newRegion] })
        } else {
          setSelection({ type: 'lasso', regions: [newRegion] })
        }
      }
      lassoPtsRef.current = []
      return
    }
    if (!drag) return
    if (Math.abs(drag.x1 - drag.x0) > MIN_DRAG_PX && Math.abs(drag.y1 - drag.y0) > MIN_DRAG_PX) {
      const prev = additivePrevRef.current
      if (prev) {
        setSelection({
          type: 'rect',
          x0: Math.min(prev.x0, prev.x1, drag.x0, drag.x1),
          y0: Math.min(prev.y0, prev.y1, drag.y0, drag.y1),
          x1: Math.max(prev.x0, prev.x1, drag.x0, drag.x1),
          y1: Math.max(prev.y0, prev.y1, drag.y0, drag.y1),
        })
      } else {
        setSelection({ type: 'rect', x0: drag.x0, y0: drag.y0, x1: drag.x1, y1: drag.y1 })
      }
    }
    setDrag(null)
  }, [drag])

  useEffect(() => {
    window.addEventListener('mouseup', onMouseUp)
    return () => window.removeEventListener('mouseup', onMouseUp)
  }, [onMouseUp])

  // Derived selection display values
  const activeRect = drag ?? (selection?.type === 'rect' ? selection : null)
  const selRect = activeRect ? {
    left:   Math.min(activeRect.x0, activeRect.x1),
    top:    Math.min(activeRect.y0, activeRect.y1),
    width:  Math.abs(activeRect.x1 - activeRect.x0),
    height: Math.abs(activeRect.y1 - activeRect.y0),
  } : null
  const selLabel = selection?.type === 'lasso'
    ? (selection.regions.length > 1 ? `LASSO ×${selection.regions.length}` : 'LASSO')
    : selRect ? `${Math.round(selRect.width)}×${Math.round(selRect.height)}PX` : null

  // ── Shared state ──────────────────────────────────────────────────────────
  const [prompt, setPrompt]       = useState('')
  const [isEditing, setIsEditing] = useState(false)
  const [credits, setCredits]     = useState(null)
  const [error, setError]         = useState(null)
  const [zoom, setZoom]           = useState(100)
  const [varyLoading, setVaryLoading] = useState(false)
  const [varyOptions, setVaryOptions] = useState(null)

  // ── Trigger effect ────────────────────────────────────────────────────────
  async function triggerEffect(name) {
    if (isEditing || varyLoading || !currentSrc) return
    const imgIdx = imageIdx   // capture before any awaits
    setEffectName(name)
    setError(null)
    clearTimeout(autoCommitTimer.current)

    // Commit any pending effect first, and get the resulting src to use as base.
    let baseSrc = activeEntry?.src ?? currentSrc
    const pl = pendingLayerRef.current
    if (pl) {
      const finalLayer = { ...pl, intensity: pendingIntRef.current, visible: true }
      const committed = await pushEntry([...currentLayersRef.current, finalLayer], imgIdx)
      setPendingLayer(null)
      setPendingIntensity(80)
      if (committed) baseSrc = committed
    }

    setIsEditing(true)
    try {
      const result = await editRegion(baseSrc, { operation: 'effect', effectName: name })
      if (result.credits_remaining != null) setCredits(result.credits_remaining)
      playRandomMeow()
      setPendingLayer({ id: genId(), type: 'effect', effectName: name, editedB64: result.edited_b64, visible: true })
      setPendingIntensity(80)
    } catch (e) {
      setError(e.message)
    } finally {
      setIsEditing(false)
    }
  }

  // ── Apply (edit / vary / upscale / remove_bg) ─────────────────────────────
  async function handleApply() {
    if (isEditing || varyLoading || !currentSrc || pendingLayer) return
    // Edit requires a prompt; selection is optional.
    if (mode === 'edit' && !prompt.trim()) return
    setError(null)
    const imgIdx  = imageIdx   // capture before any awaits
    const baseSrc = activeEntry?.src ?? currentSrc

    if (mode === 'vary') {
      setVaryLoading(true)
      setVaryOptions(null)
      try {
        const results = await Promise.all([
          editRegion(baseSrc, { operation: 'vary', prompt }),
          editRegion(baseSrc, { operation: 'vary', prompt }),
          editRegion(baseSrc, { operation: 'vary', prompt }),
        ])
        const last = results[results.length - 1]
        if (last.credits_remaining != null) setCredits(last.credits_remaining)
        setVaryOptions(results.map(r => `data:image/png;base64,${r.edited_b64}`))
        playRandomMeow()
      } catch (e) {
        setError(e.message)
      } finally {
        setVaryLoading(false)
      }
      return
    }

    setIsEditing(true)

    let region       = null
    let layerExtra   = {}

    if (mode === 'edit' && selection) {
      const { width: fw, height: fh } = frameRef.current.getBoundingClientRect()

      if (selection.type === 'lasso') {
        const allPts = selection.regions.flatMap(r => r.points)
        const xs = allPts.map(p => p.x), ys = allPts.map(p => p.y)
        const bx0 = Math.min(...xs), by0 = Math.min(...ys)
        const bx1 = Math.max(...xs), by1 = Math.max(...ys)
        region = { x: bx0/fw, y: by0/fh, w: (bx1-bx0)/fw, h: (by1-by0)/fh }
        layerExtra = {
          selectionType: 'lasso',
          normalizedRegions: selection.regions.map(r => ({
            points: r.points.map(p => ({ x: p.x/fw, y: p.y/fh })),
          })),
          normalizedBbox: region,
        }
      } else {
        region = {
          x: Math.min(selection.x0, selection.x1) / fw,
          y: Math.min(selection.y0, selection.y1) / fh,
          w: Math.abs(selection.x1 - selection.x0) / fw,
          h: Math.abs(selection.y1 - selection.y0) / fh,
        }
        layerExtra = { selectionType: 'rect', region }
      }
    }

    try {
      const result = await editRegion(baseSrc, {
        operation: mode, prompt: prompt.trim(), region, upscaleFactor: 2,
      })
      if (result.credits_remaining != null) setCredits(result.credits_remaining)
      playRandomMeow()

      if (mode === 'edit' && region) {
        const newLayer = {
          id: genId(), type: 'region', visible: true,
          prompt: prompt.trim(), editedB64: result.edited_b64,
          ...layerExtra,
        }
        await pushEntry([...currentLayersRef.current, newLayer], imgIdx)
        setSelection(null)
        setPrompt('')
      } else {
        // upscale / remove_bg: replace the original, keep other layers
        const newSrc      = `data:image/png;base64,${result.edited_b64}`
        const newOriginal = { id: genId(), type: 'original', src: newSrc, visible: true }
        const rest        = currentLayersRef.current.filter(l => l.type !== 'original')
        await pushEntry([newOriginal, ...rest], imgIdx)
      }
    } catch (e) {
      setError(e.message)
    } finally {
      setIsEditing(false)
    }
  }

  function acceptVary(src) {
    pushEntry([{ id: genId(), type: 'original', src, visible: true }], imageIdx)
    setVaryOptions(null)
    setPrompt('')
  }

  function handleExport() {
    if (!currentSrc) return
    const a = document.createElement('a')
    a.href     = currentSrc
    a.download = `ltnt_edit_${String(imageIdx).padStart(2, '0')}.png`
    a.click()
  }

  // ── Derived ───────────────────────────────────────────────────────────────
  // Edit only needs a prompt — the selection is an optional compositing
  // mask (without it, Reve's result is used for the full image).
  const canApply = !isEditing && !varyLoading && !!currentSrc && !pendingLayer && mode !== 'effect' && (
    mode === 'edit' ? !!prompt.trim() : true
  )
  const canUndo = imgHist.ptr > 0               && !pendingLayer && !varyLoading && !isEditing
  const canRedo = imgHist.ptr < imgHist.entries.length - 1 && !pendingLayer && !varyLoading && !isEditing
  const currentOp    = OPERATIONS.find(o => o.id === mode)
  const effectLayers = currentLayers.filter(l => l.type === 'effect')
  const regionLayers = currentLayers.filter(l => l.type === 'region')

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div style={{ height: '100vh', display: 'flex', background: '#e8e8e8' }}>
      <style>{GLOBAL_CSS}</style>

      {/* ── Left sidebar ─────────────────────────────────────────────────── */}
      <aside style={{
        width: '230px', flexShrink: 0,
        background: '#fff', borderRight: '1px solid #d0d0d0',
        display: 'flex', flexDirection: 'column', overflow: 'hidden',
      }}>
        <div style={{ padding: '14px 20px' }}>
          <div style={{ textTransform: 'uppercase' }}>[ × ] LTNT.APP</div>
        </div>

        {/* Image picker */}
        {images.length > 0 && (
          <div style={{ marginBottom: '14px' }}>
            <div style={{ fontSize: '10px', letterSpacing: '0.08em', textTransform: 'uppercase', color: '#888', padding: '0 14px', marginBottom: '6px' }}>
              /EDITING ({images.length})
            </div>
            <div data-tour="image-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '4px', padding: '0 14px' }}>
              {images.map((img, i) => {
                const slot   = histState[i]
                const topSrc = slot?.entries[slot.ptr]?.src ?? img.src ?? img.url
                return (
                  <div
                    key={img.id}
                    onClick={() => { setImageIdx(i); setSelection(null); setError(null); setVaryOptions(null); setPendingLayer(null) }}
                    style={{
                      aspectRatio: '1', overflow: 'hidden', cursor: 'pointer',
                      outline: i === imageIdx ? '2.5px solid #1a1a1a' : '2.5px solid transparent',
                      outlineOffset: '-2.5px',
                    }}
                  >
                    <img src={topSrc} alt="" style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }} />
                  </div>
                )
              })}
            </div>
          </div>
        )}

        {/* Operations */}
        <div data-tour="operations" style={{ marginBottom: '14px' }}>
          <div style={{ fontSize: '10px', letterSpacing: '0.08em', textTransform: 'uppercase', color: '#888', padding: '0 14px', marginBottom: '6px' }}>/OPERATIONS</div>
          {OPERATIONS.map(op => (
            <div
              key={op.id}
              onClick={() => { setMode(op.id); setSelection(null); setError(null); setVaryOptions(null) }}
              style={{
                padding: '5px 14px', display: 'flex', alignItems: 'center', gap: '8px',
                cursor: 'pointer', textTransform: 'uppercase', letterSpacing: '0.04em',
                background: mode === op.id ? '#f0f0f0' : 'transparent',
                borderLeft: mode === op.id ? '2px solid #1a1a1a' : '2px solid transparent',
              }}
            >
              <span style={{ color: mode === op.id ? '#1a1a1a' : '#bbb' }}>{mode === op.id ? '[×]' : '[ ]'}</span>
              <span>{op.label}</span>
            </div>
          ))}
        </div>

        {/* Effect list */}
        {mode === 'effect' && (
          <div data-tour="effects" style={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column', marginBottom: '14px' }}>
            <div style={{ fontSize: '10px', letterSpacing: '0.08em', textTransform: 'uppercase', color: '#888', padding: '0 14px', marginBottom: '6px' }}>
              /EFFECTS{effectsLoading ? ' ...' : ` (${effects.length})`}
            </div>
            <div style={{ overflowY: 'auto', flex: 1 }}>
              {effects.map(fx => (
                <div
                  key={fx.name}
                  onClick={() => triggerEffect(fx.name)}
                  style={{
                    padding: '4px 14px', cursor: isEditing ? 'wait' : 'pointer',
                    textTransform: 'uppercase', letterSpacing: '0.04em',
                    background: effectName === fx.name ? '#f0f0f0' : 'transparent',
                    borderLeft: effectName === fx.name ? '2px solid #1a1a1a' : '2px solid transparent',
                    display: 'flex', alignItems: 'center', gap: '8px',
                    opacity: isEditing ? 0.5 : 1,
                  }}
                >
                  <span style={{ color: effectName === fx.name ? '#1a1a1a' : '#bbb', flexShrink: 0 }}>
                    {effectName === fx.name ? '[×]' : '[ ]'}
                  </span>
                  <span>{fx.name}</span>
                </div>
              ))}
              {!effectsLoading && effects.length === 0 && (
                <div style={{ padding: '8px 14px', color: '#bbb', fontSize: '10px', textTransform: 'uppercase' }}>NONE AVAILABLE</div>
              )}
            </div>
          </div>
        )}

        {/* Credits */}
        <div style={{ marginTop: mode === 'effect' ? 0 : 'auto', marginBottom: '14px' }}>
          <div style={{ fontSize: '10px', letterSpacing: '0.08em', textTransform: 'uppercase', color: '#888', padding: '0 14px', marginBottom: '6px' }}>/CREDITS</div>
          <div style={{ margin: '0 14px', border: '1px solid #d0d0d0', background: '#f8f8f8', padding: '8px 10px' }}>
            <div style={{ fontSize: '10px', letterSpacing: '0.06em', textTransform: 'uppercase', color: '#888', marginBottom: '3px' }}>REMAINING</div>
            <div style={{ fontSize: '20px', letterSpacing: '-0.02em', color: credits === 0 ? '#cc3333' : '#1a1a1a', lineHeight: 1 }}>
              {credits == null ? '—' : credits.toLocaleString()}
            </div>
          </div>
        </div>

        <div style={{ marginTop: 'auto' }}>
          <div style={{ height: '1px', background: '#d0d0d0', margin: '0 14px 10px' }} />
          {['ABOUT', 'LOG'].map(l => (
            <div key={l} style={{ padding: '4px 14px', textTransform: 'uppercase', letterSpacing: '0.04em', cursor: 'pointer', marginBottom: '2px' }}>[ ] {l}</div>
          ))}
        </div>
      </aside>

      {/* ── Main column ──────────────────────────────────────────────────── */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', minWidth: 0 }}>

        {/* Top bar */}
        <div style={{
          height: '56px', display: 'flex', alignItems: 'center',
          justifyContent: 'space-between', padding: '0 16px',
          borderBottom: '1px solid #d0d0d0', flexShrink: 0, gap: '12px',
        }}>
          <div style={{ display: 'flex', gap: '8px', flexShrink: 0 }}>
            <TourToggle tourMode={tourMode} onTourModeChange={onTourModeChange} />
            <button onClick={onBack} style={{ border: '1px solid #1a1a1a', padding: '6px 12px', background: 'none' }}>← [ ] BACK</button>
          </div>

          <div style={{ flex: 1, textAlign: 'center', textTransform: 'uppercase', letterSpacing: '0.06em', color: '#888', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            /EDIT_VIEW • {currentOp?.label}
            {mode === 'edit' && selLabel && !drag && !isLassoDrawing.current ? ` • ${selection?.type === 'lasso' ? '⌖' : '▣'} ${selLabel}` : ''}
            {(isEditing || varyLoading) ? ' • APPLYING...' : ''}
            {pendingLayer ? ' • ADJUST INTENSITY' : ''}
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', flexShrink: 0 }}>
            <button onClick={undo} disabled={!canUndo} title="Undo (⌘Z)"
              style={{ border: '1px solid #d0d0d0', padding: '4px 8px', background: 'none', opacity: canUndo ? 1 : 0.3, cursor: canUndo ? 'pointer' : 'not-allowed' }}>↩</button>
            <button onClick={redo} disabled={!canRedo} title="Redo (⌘⇧Z)"
              style={{ border: '1px solid #d0d0d0', padding: '4px 8px', background: 'none', opacity: canRedo ? 1 : 0.3, cursor: canRedo ? 'pointer' : 'not-allowed' }}>↪</button>
            <button onClick={() => setZoom(z => Math.max(25, z - 25))}
              style={{ border: '1px solid #d0d0d0', padding: '4px 8px', background: 'none' }}>−</button>
            <span style={{ minWidth: '36px', textAlign: 'center', textTransform: 'uppercase' }}>{zoom}%</span>
            <button onClick={() => setZoom(z => Math.min(400, z + 25))}
              style={{ border: '1px solid #d0d0d0', padding: '4px 8px', background: 'none' }}>+</button>
          </div>

          <button data-tour="export-btn" onClick={handleExport} disabled={!currentSrc}
            style={{ border: '1px solid #1a1a1a', padding: '6px 14px', background: '#1a1a1a', color: 'white', flexShrink: 0, opacity: currentSrc ? 1 : 0.4 }}>
            [ ] EXPORT
          </button>
        </div>

        {/* Middle: canvas + right panel */}
        <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>

          {/* Canvas area */}
          <div style={{ flex: 1, position: 'relative', display: 'flex', alignItems: 'center', justifyContent: 'center', overflow: 'hidden' }}>
            {currentSrc ? (
              <div
                ref={frameRef}
                onMouseDown={onMouseDown}
                onMouseMove={onMouseMove}
                style={{
                  position: 'relative',
                  width:  `${Math.round(512 * zoom / 100)}px`,
                  height: `${Math.round(512 * zoom / 100)}px`,
                  maxWidth: 'calc(100% - 60px)', maxHeight: 'calc(100% - 40px)',
                  flexShrink: 0, overflow: 'hidden',
                  boxShadow: '0 4px 24px rgba(0,0,0,0.25)',
                  border: '1px solid rgba(0,0,0,0.1)',
                  cursor: isEditing || varyLoading ? 'wait' : mode === 'edit' && !pendingLayer ? 'crosshair' : 'default',
                  userSelect: 'none',
                }}
              >
                {/* Base image — always the committed flat result */}
                <img
                  src={currentSrc}
                  alt="" crossOrigin="anonymous" draggable={false}
                  style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block', pointerEvents: 'none' }}
                />

                {/* Pending edit preview: the edited crop is stretched into
                    the user's selected region (or covers the whole canvas
                    for full-image edits). Matches what compositeLayerStack
                    will produce on commit. */}
                {pendingLayer && (() => {
                  const r = pendingLayer.region
                  const style = r ? {
                    position: 'absolute',
                    left:   `${r.x * 100}%`,
                    top:    `${r.y * 100}%`,
                    width:  `${r.w * 100}%`,
                    height: `${r.h * 100}%`,
                    opacity: pendingIntensity / 100,
                    pointerEvents: 'none',
                  } : {
                    position: 'absolute', inset: 0,
                    width: '100%', height: '100%', objectFit: 'cover',
                    opacity: pendingIntensity / 100,
                    pointerEvents: 'none',
                  }
                  return (
                    <img
                      src={`data:image/png;base64,${pendingLayer.editedB64}`}
                      alt="" draggable={false}
                      style={style}
                    />
                  )
                })()}

                {/* Rect selection */}
                {selRect && mode === 'edit' && !pendingLayer && (
                  <div style={{
                    position: 'absolute',
                    left: selRect.left, top: selRect.top,
                    width: selRect.width, height: selRect.height,
                    border: `1.5px ${drag ? 'dashed' : 'solid'} #fff`,
                    boxShadow: '0 0 0 1px rgba(0,0,0,0.5)',
                    background: drag ? 'rgba(255,255,255,0.06)' : 'none',
                    pointerEvents: 'none', boxSizing: 'border-box',
                  }}>
                    {!drag && selLabel && (
                      <div style={{
                        position: 'absolute', bottom: '100%', left: 0,
                        background: '#1a1a1a', color: '#fff',
                        fontSize: '9px', letterSpacing: '0.06em', padding: '2px 5px',
                        whiteSpace: 'nowrap', marginBottom: '3px',
                      }}>
                        {selLabel}
                      </div>
                    )}
                  </div>
                )}

                {/* Lasso — live stroke while drawing */}
                {isLassoDrawing.current && mode === 'edit' && !pendingLayer && lassoPtsRef.current.length > 1 && (
                  <svg style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none', overflow: 'visible' }}>
                    <polyline
                      points={lassoPtsRef.current.map(p => `${p.x},${p.y}`).join(' ')}
                      fill="rgba(255,255,255,0.08)" stroke="white" strokeWidth="1.5"
                      strokeDasharray="5,3" strokeLinecap="round"
                    />
                  </svg>
                )}

                {/* Lasso — committed polygons (hidden while drawing a replacement stroke) */}
                {selection?.type === 'lasso' && mode === 'edit' && !pendingLayer && !isLassoDrawing.current && (
                  <svg style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none', overflow: 'visible' }}>
                    {selection.regions.map((region, ri) => (
                      <polygon key={ri}
                        points={region.points.map(p => `${p.x},${p.y}`).join(' ')}
                        fill="rgba(255,255,255,0.08)" stroke="white" strokeWidth="1.5"
                      />
                    ))}
                    {(() => {
                      const pts = selection.regions[0]?.points ?? []
                      if (!pts.length) return null
                      const cx = pts.reduce((s, p) => s + p.x, 0) / pts.length
                      const cy = pts.reduce((s, p) => s + p.y, 0) / pts.length
                      return (
                        <text x={cx} y={cy - 6} textAnchor="middle"
                          fill="white" fontSize="9" fontFamily="monospace"
                          letterSpacing="0.06em">
                          {selection.regions.length > 1 ? `LASSO ×${selection.regions.length}` : 'LASSO'}
                        </text>
                      )
                    })()}
                  </svg>
                )}

                {/* Cat loader */}
                {(isEditing || varyLoading) && (
                  <div style={{
                    position: 'absolute', inset: 0,
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    background: 'rgba(232,232,232,0.88)', pointerEvents: 'none',
                  }}>
                    <CatLoader progress={null} statusText={varyLoading ? 'GENERATING 3 VARIATIONS' : currentOp?.label} />
                  </div>
                )}
              </div>
            ) : (
              <div style={{ color: '#bbb', textTransform: 'uppercase', letterSpacing: '0.1em', textAlign: 'center' }}>
                <div style={{ fontSize: '48px', opacity: 0.3, marginBottom: '12px' }}>✦</div>
                NO IMAGE SELECTED
              </div>
            )}

            {/* Vary picker */}
            {varyOptions && (
              <div style={{
                position: 'absolute', inset: 0, background: 'rgba(232,232,232,0.92)',
                display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '16px',
              }}>
                <div style={{ fontSize: '10px', letterSpacing: '0.1em', textTransform: 'uppercase', color: '#888' }}>
                  SELECT A VARIATION — OR PRESS ESC TO CANCEL
                </div>
                <div style={{ display: 'flex', gap: '16px' }}>
                  {varyOptions.map((src, i) => (
                    <div key={i} onClick={() => acceptVary(src)}
                      style={{ width: '180px', height: '180px', overflow: 'hidden', cursor: 'pointer', boxShadow: '0 2px 16px rgba(0,0,0,0.18)', border: '2px solid transparent', transition: 'border-color 0.1s, transform 0.1s' }}
                      onMouseEnter={e => { e.currentTarget.style.borderColor = '#1a1a1a'; e.currentTarget.style.transform = 'scale(1.03)' }}
                      onMouseLeave={e => { e.currentTarget.style.borderColor = 'transparent'; e.currentTarget.style.transform = 'scale(1)' }}
                    >
                      <img src={src} alt={`Variation ${i + 1}`} style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }} />
                    </div>
                  ))}
                </div>
                <button onClick={() => setVaryOptions(null)}
                  style={{ border: '1px solid #bbb', padding: '6px 16px', background: 'none', color: '#888', cursor: 'pointer', letterSpacing: '0.06em', textTransform: 'uppercase' }}>
                  [ ] CANCEL
                </button>
              </div>
            )}

            {/* Error toast */}
            {error && (
              <div style={{
                position: 'absolute', bottom: '16px', left: '50%', transform: 'translateX(-50%)',
                background: '#1a1a1a', color: '#ff6b6b',
                padding: '7px 14px', fontSize: '10px', letterSpacing: '0.06em',
                textTransform: 'uppercase', whiteSpace: 'nowrap',
                maxWidth: '90%', overflow: 'hidden', textOverflow: 'ellipsis',
              }}>
                [ ERR ] {error}
              </div>
            )}
          </div>

          {/* ── Right panel ──────────────────────────────────────────────── */}
          <div style={{
            width: '190px', flexShrink: 0,
            borderLeft: '1px solid #d0d0d0', background: '#fff',
            display: 'flex', flexDirection: 'column', overflow: 'hidden',
          }}>

            {/* /HISTORY */}
            <div style={{ borderBottom: '1px solid #f0f0f0', flexShrink: 0 }}>
              <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', padding: '8px 12px 4px' }}>
                <span style={{ fontSize: '10px', letterSpacing: '0.08em', textTransform: 'uppercase', color: '#888' }}>/HISTORY</span>
                <span style={{ fontSize: '9px', color: '#bbb' }}>{imgHist.ptr + 1}/{imgHist.entries.length}</span>
              </div>
              <div style={{ overflowY: 'auto', maxHeight: '180px', padding: '0 8px 8px' }}>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px' }}>
                  {[...imgHist.entries].reverse().map((entry, ri) => {
                    const idx      = imgHist.entries.length - 1 - ri
                    const isActive = idx === imgHist.ptr
                    return (
                      <div key={idx} onClick={() => jumpToEntry(idx)} title={idx === 0 ? 'Original' : `State #${idx}`}
                        style={{ cursor: 'pointer', outline: isActive ? '2px solid #1a1a1a' : '2px solid transparent', position: 'relative' }}>
                        <img src={entry.src} alt=""
                          style={{ width: '100%', aspectRatio: '1', objectFit: 'cover', display: 'block' }} />
                        <div style={{
                          position: 'absolute', bottom: 0, left: 0, right: 0,
                          background: isActive ? '#1a1a1a' : 'rgba(0,0,0,0.45)',
                          color: '#fff', fontSize: '8px', padding: '2px 4px', textAlign: 'right',
                        }}>
                          {idx === 0 ? 'ORIG' : `#${idx}`}
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>
            </div>

            {/* /FILTER LAYERS */}
            <div style={{ borderBottom: '1px solid #f0f0f0', flexShrink: 0 }}>
              <SectionLabel>/FILTER LAYERS{effectLayers.length > 0 ? ` (${effectLayers.length})` : ''}</SectionLabel>
              {effectLayers.length === 0 ? (
                <div style={{ padding: '0 12px 8px', fontSize: '9px', color: '#ccc', textTransform: 'uppercase', letterSpacing: '0.04em' }}>NONE YET</div>
              ) : (
                <div style={{ maxHeight: '200px', overflowY: 'auto' }}>
                  {effectLayers.map(layer => {
                    const isAdj    = adjustLayerId === layer.id
                    const sliderVal = isAdj ? adjustIntensity : layer.intensity
                    return (
                      <div key={layer.id} style={{ padding: '4px 8px 6px', borderBottom: '1px solid #f8f8f8' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '4px', marginBottom: '3px' }}>
                          <EyeBtn visible={layer.visible} onClick={() => toggleLayerVisibility(layer.id)} />
                          <span style={{
                            fontSize: '9px', letterSpacing: '0.04em', textTransform: 'uppercase',
                            flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                            color: layer.visible ? '#333' : '#bbb',
                          }}>
                            {layer.effectName}
                          </span>
                          <span style={{ fontSize: '8px', color: '#bbb', flexShrink: 0 }}>{sliderVal ?? layer.intensity}%</span>
                        </div>
                        {layer.visible && (
                          <input type="range" min={0} max={100} value={sliderVal ?? layer.intensity}
                            onChange={e => handleLayerIntensity(layer.id, Number(e.target.value))}
                            className="ltnt-slider-light"
                          />
                        )}
                      </div>
                    )
                  })}
                </div>
              )}
            </div>

            {/* /REGION EDITS */}
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
              <SectionLabel>/REGION EDITS{regionLayers.length > 0 ? ` (${regionLayers.length})` : ''}</SectionLabel>
              {regionLayers.length === 0 ? (
                <div style={{ padding: '0 12px', fontSize: '9px', color: '#ccc', textTransform: 'uppercase', letterSpacing: '0.04em' }}>NONE YET</div>
              ) : (
                <div style={{ flex: 1, overflowY: 'auto' }}>
                  {regionLayers.map(layer => (
                    <div key={layer.id} style={{ padding: '5px 8px', display: 'flex', alignItems: 'flex-start', gap: '4px', borderBottom: '1px solid #f8f8f8' }}>
                      <EyeBtn visible={layer.visible} onClick={() => toggleLayerVisibility(layer.id)} />
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div style={{ fontSize: '8px', color: '#aaa', textTransform: 'uppercase', letterSpacing: '0.04em', marginBottom: '1px' }}>
                          {layer.selectionType === 'lasso'
                            ? `⌖ LASSO${(layer.normalizedRegions?.length ?? 1) > 1 ? ` ×${layer.normalizedRegions.length}` : ''}`
                            : '▢ RECT'}
                        </div>
                        <div style={{ fontSize: '9px', color: layer.visible ? '#444' : '#bbb', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                          {layer.prompt}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

          </div>{/* end right panel */}
        </div>{/* end middle row */}

        {/* ── Bottom bar ───────────────────────────────────────────────── */}
        {pendingLayer ? (
          <div style={{
            borderTop: '1px solid #d0d0d0', background: '#1a1a1a',
            display: 'flex', alignItems: 'center', padding: '0 16px',
            height: '56px', gap: '16px', flexShrink: 0,
          }}>
            <div style={{ color: 'rgba(255,255,255,0.5)', fontSize: '10px', letterSpacing: '0.06em', textTransform: 'uppercase', flexShrink: 0 }}>
              INTENSITY
            </div>
            <input type="range" min={0} max={100} value={pendingIntensity}
              onChange={e => handlePendingIntensity(Number(e.target.value))}
              className="ltnt-slider"
              style={{ flex: 1 }}
            />
            <div style={{ color: 'rgba(255,255,255,0.6)', fontSize: '11px', minWidth: '32px', textAlign: 'right', flexShrink: 0 }}>
              {pendingIntensity}%
            </div>
            <div style={{ color: 'rgba(255,255,255,0.25)', fontSize: '9px', letterSpacing: '0.06em', textTransform: 'uppercase', flexShrink: 0 }}>
              AUTO-SAVES
            </div>
            <button
              onClick={() => { clearTimeout(autoCommitTimer.current); setPendingLayer(null) }}
              style={{ border: '1px solid rgba(255,255,255,0.2)', padding: '6px 12px', background: 'transparent', color: 'rgba(255,255,255,0.5)', cursor: 'pointer', letterSpacing: '0.04em', flexShrink: 0 }}
            >
              [ ] DISCARD
            </button>
          </div>
        ) : (
          <div style={{
            borderTop: '1px solid #d0d0d0', background: '#1a1a1a',
            display: 'flex', alignItems: 'center', padding: '0 16px',
            height: '56px', gap: '12px', flexShrink: 0,
          }}>
            {mode === 'edit' && (
              <>
                {['rect', 'lasso'].map(tool => (
                  <button key={tool}
                    onClick={() => { setSelectionTool(tool); setSelection(null) }}
                    title={tool === 'rect' ? 'Rectangle select (R)' : 'Lasso select (L)'}
                    style={{
                      border: `1px solid ${selectionTool === tool ? 'rgba(255,255,255,0.55)' : 'rgba(255,255,255,0.15)'}`,
                      padding: '3px 8px', background: 'transparent',
                      color: selectionTool === tool ? 'white' : 'rgba(255,255,255,0.35)',
                      fontSize: '9px', letterSpacing: '0.06em', textTransform: 'uppercase',
                      cursor: 'pointer', flexShrink: 0,
                    }}>
                    {tool === 'rect' ? '▢ RECT' : '⌖ LASSO'}
                  </button>
                ))}
                <div style={{ color: selection ? 'rgba(255,255,255,0.55)' : 'rgba(255,255,255,0.2)', fontSize: '10px', letterSpacing: '0.06em', textTransform: 'uppercase', flexShrink: 0 }}>
                  {selection ? (selection.type === 'lasso' ? `⌖ ${selLabel}` : `▣ ${selLabel}`) : '— NONE'}
                </div>
              </>
            )}
            {mode === 'effect' && (
              <div style={{ color: effectName ? 'rgba(255,255,255,0.55)' : 'rgba(255,255,255,0.2)', fontSize: '10px', letterSpacing: '0.06em', textTransform: 'uppercase', flexShrink: 0 }}>
                {effectName ? `✦ ${effectName}` : '← PICK AN EFFECT'}
              </div>
            )}

            {(mode === 'edit' || mode === 'vary') && (
              <>
                <div style={{ width: '1px', height: '20px', background: 'rgba(255,255,255,0.1)', flexShrink: 0 }} />
                <input
                  value={prompt}
                  onChange={e => setPrompt(e.target.value)}
                  onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleApply() } }}
                  disabled={isEditing || varyLoading}
                  placeholder={mode === 'edit' ? '[ ] Describe the edit (optionally draw a region to target it)...' : '[ ] Optional variation hint...'}
                  style={{ flex: 1, color: (isEditing || varyLoading) ? 'rgba(255,255,255,0.4)' : 'rgba(255,255,255,0.85)' }}
                />
              </>
            )}

            <div style={{ flex: mode === 'edit' || mode === 'vary' ? 0 : 1 }} />

            {mode !== 'effect' && (
              <button onClick={handleApply} disabled={!canApply}
                style={{
                  border: `1px solid ${canApply ? 'rgba(255,255,255,0.4)' : 'rgba(255,255,255,0.1)'}`,
                  padding: '8px 18px', background: 'transparent',
                  color: canApply ? 'white' : 'rgba(255,255,255,0.3)',
                  whiteSpace: 'nowrap', display: 'flex', alignItems: 'center', gap: '6px',
                  flexShrink: 0, cursor: canApply ? 'pointer' : 'not-allowed',
                }}>
                ✦ [ ] {currentOp?.label}
              </button>
            )}
          </div>
        )}

      </div>{/* end main column */}

      <Tour
        steps={EDIT_TOUR_STEPS}
        open={tour.open}
        onClose={() => { tour.close(); onTourModeChange(false) }}
        storageKey="ltnt.tour.edit"
      />
    </div>
  )
}

// ── Global CSS (injected once as a static string, not re-evaluated on renders) ─
const GLOBAL_CSS = `
  .ltnt-slider {
    -webkit-appearance: none; appearance: none;
    background: transparent; cursor: pointer; height: 20px; flex: 1;
  }
  .ltnt-slider::-webkit-slider-runnable-track { background: rgba(255,255,255,0.25); height: 2px; border-radius: 1px; }
  .ltnt-slider::-webkit-slider-thumb { -webkit-appearance: none; background: white; width: 14px; height: 14px; border-radius: 50%; margin-top: -6px; }
  .ltnt-slider::-moz-range-track { background: rgba(255,255,255,0.25); height: 2px; border-radius: 1px; }
  .ltnt-slider::-moz-range-thumb { background: white; width: 14px; height: 14px; border-radius: 50%; border: none; }

  .ltnt-slider-light {
    -webkit-appearance: none; appearance: none;
    background: transparent; cursor: pointer; height: 16px; width: 100%;
  }
  .ltnt-slider-light::-webkit-slider-runnable-track { background: rgba(0,0,0,0.15); height: 2px; border-radius: 1px; }
  .ltnt-slider-light::-webkit-slider-thumb { -webkit-appearance: none; background: #555; width: 12px; height: 12px; border-radius: 50%; margin-top: -5px; }
  .ltnt-slider-light::-moz-range-track { background: rgba(0,0,0,0.15); height: 2px; border-radius: 1px; }
  .ltnt-slider-light::-moz-range-thumb { background: #555; width: 12px; height: 12px; border-radius: 50%; border: none; }
`
