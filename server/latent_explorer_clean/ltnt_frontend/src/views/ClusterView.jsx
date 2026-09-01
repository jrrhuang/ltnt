import { useRef, useState, useEffect, useCallback } from 'react'
import { submitSelection, pollJob, adjustLayout, pinToBoard, requestRefine, getJobStatus, getSessionState, finishRound } from '../api'
import Tour, { useTour, TourToggle } from '../Tour'
import Lightbox from '../components/Lightbox'

const CLUSTER_TOUR_STEPS = [
  {
    title: 'Your images',
    body: "Here are the images we made for you. Similar-looking ones are placed near each other, so it's easy to spot the styles you like most.",
    placement: 'center',
  },
  {
    target: '[data-tour="selection-panel"]',
    title: 'Pick your favorites',
    body: "Click the images you like. Hold Shift and drag to pick a whole group at once. These become the starting point for the next round.",
    placement: 'right',
  },
  {
    target: '[data-tour="layout-controls"]',
    title: 'Rearrange the view',
    body: "Feeling crowded? These sliders spread the images out or group similar ones closer together, so you can see everything clearly.",
    placement: 'right',
  },
  {
    target: '[data-tour="magnifier-btn"]',
    title: 'Take a closer look',
    body: "Turn this on (or press M) to zoom into any image just by hovering over it.",
    placement: 'right',
  },
  {
    target: '[data-tour="branch-btn"]',
    title: 'Make more like these',
    body: "Picked your favorites? Click Make More Like These — we'll generate new variations that build on what you chose, so you can keep narrowing in on the look you want.",
    placement: 'left',
  },
]

const SHORTCUTS = [
  { label: 'Pan canvas',    shortcut: 'Drag' },
  { label: 'Select region', shortcut: 'Shift + Drag' },
  { label: 'Toggle image',  shortcut: 'Click' },
  { label: 'Zoom',          shortcut: 'Scroll' },
  { label: 'Fit view',      shortcut: 'Shift + 0' },
  { label: 'Select all',    shortcut: 'Ctrl + A' },
  { label: 'Magnifier',     shortcut: 'M' },
]

const LOUPE_RADIUS = 110        // px — lens size on screen
const LOUPE_MAGNIFICATION = 3   // extra zoom on top of current canvas zoom

const IMG_SIZE_BASE = 132
// Shrink images as total count grows: 132px at ≤9, ~93px at 18, ~66px at 36, ~50px at 72.
// Bumped up from 96 so a fresh 6-image round renders comfortably large; the
// auto-fit-to-view (see fitToView) then scales the whole canvas so all current
// tiles fill the viewport without the artist having to zoom manually.
function getImgSize(totalCount) {
  if (totalCount <= 9) return IMG_SIZE_BASE
  return Math.max(44, IMG_SIZE_BASE * Math.sqrt(9 / totalCount))
}

/**
 * Override server-computed positions so anchors inherit their parent's position
 * and clones are placed as small offsets around the anchor.
 *
 * @param {Array} imgs - images from server, grouped as [a0, c0_1, ..., a1, c1_1, ...]
 * @param {Array} parents - [{x, y}, ...] saved positions of selected parents (in order)
 * @returns {Array} imgs with updated x, y positions
 */
function applyParentPositions(imgs, parents) {
  const N = parents.length
  if (N === 0) return imgs
  const C = Math.max(Math.round(imgs.length / N), 1)

  return imgs.map((img, i) => {
    const groupIdx = Math.floor(i / C)
    const posInGroup = i % C
    if (groupIdx >= N) return img // safety

    const parent = parents[groupIdx]

    if (posInGroup === 0) {
      // Anchor: inherit exact parent position
      return { ...img, x: parent.x, y: parent.y }
    } else {
      // Clone: offset from parent position
      // Use server-computed offset (difference between clone and its anchor in server coords)
      // Find this group's anchor in the original imgs
      const anchorIdx = groupIdx * C
      const serverAnchor = imgs[anchorIdx]
      const dx = img.x - serverAnchor.x
      const dy = img.y - serverAnchor.y
      return { ...img, x: parent.x + dx, y: parent.y + dy }
    }
  })
}

/**
 * Frontend de-overlap pass. Spawned siblings often land on near-identical
 * positions (the backend offsets are tiny), so they stack on top of each other
 * and the artist can't inspect them individually. This runs a few cheap
 * relaxation iterations: any two tiles whose centers are closer than the
 * desired spacing are pushed directly apart along the line between them. Clusters
 * stay grouped (we only move the minimum needed to un-stack), but no two tiles
 * remain fully overlapping.
 *
 * Operates in screen-pixel space (where tile size lives), then converts the
 * adjusted centers back to the normalized 0..1 coordinates the rest of the
 * view uses. Pure — returns new objects, never mutates the input. If the
 * canvas isn't measured yet (w/h === 0) it's a no-op.
 *
 * @param {Array} imgs - images with normalized {x, y}
 * @param {{w:number,h:number}} canvas - measured canvas size in px
 * @param {(img:Object)=>number} sizeForPx - tile edge length in px for an img
 * @returns {Array} imgs with de-overlapped normalized x, y
 */
function deOverlap(imgs, canvas, sizeForPx) {
  const { w, h } = canvas
  if (!w || !h || imgs.length < 2) return imgs

  // Work on pixel centers + radii.
  const pts = imgs.map(img => ({
    px: img.x * w,
    py: img.y * h,
    r: sizeForPx(img) / 2,
  }))

  // A few relaxation passes are enough to separate even tightly stacked groups.
  const PAD = 6           // px of breathing room between tile edges
  const ITERS = 12
  const JITTER = 0.5      // tiny deterministic nudge when two centers coincide
  for (let it = 0; it < ITERS; it++) {
    let moved = false
    for (let i = 0; i < pts.length; i++) {
      for (let j = i + 1; j < pts.length; j++) {
        const a = pts[i], b = pts[j]
        let dx = b.px - a.px
        let dy = b.py - a.py
        let dist = Math.hypot(dx, dy)
        const minDist = a.r + b.r + PAD
        if (dist >= minDist) continue
        // Coincident centers: nudge apart deterministically (use index to fan out).
        if (dist < 1e-3) {
          const ang = (i * 53 + j * 131) % 360 * Math.PI / 180
          dx = Math.cos(ang); dy = Math.sin(ang); dist = JITTER
        }
        const overlap = (minDist - dist) / 2
        const ux = dx / dist, uy = dy / dist
        a.px -= ux * overlap; a.py -= uy * overlap
        b.px += ux * overlap; b.py += uy * overlap
        moved = true
      }
    }
    if (!moved) break
  }

  return imgs.map((img, i) => ({ ...img, x: pts[i].px / w, y: pts[i].py / h }))
}

/**
 * Span-batch collision avoidance: move ONLY the tiles without `_placed` so a
 * new batch never buries the existing spread (and never disturbs tiles the
 * user dragged or that were already laid out). Pixel-space repulsion of new
 * tiles off ALL tiles; placed tiles are immovable obstacles.
 */
function spreadNewTiles(imgs, canvas) {
  const { w, h } = canvas || {}
  if (!w || !h || imgs.length < 2) return imgs
  const size = getImgSize(imgs.length)
  const PAD = 6
  const minDist = size + PAD
  const pts = imgs.map(img => ({
    px: img.x * w, py: img.y * h, fixed: !!img._placed,
  }))
  for (let it = 0; it < 30; it++) {
    let moved = false
    for (let i = 0; i < pts.length; i++) {
      if (pts[i].fixed) continue
      for (let j = 0; j < pts.length; j++) {
        if (i === j) continue
        const a = pts[i], b = pts[j]
        let dx = a.px - b.px, dy = a.py - b.py
        let dist = Math.hypot(dx, dy)
        if (dist >= minDist) continue
        if (dist < 1e-3) {
          const ang = ((i * 47 + j * 101) % 360) * Math.PI / 180
          dx = Math.cos(ang); dy = Math.sin(ang); dist = 1
        }
        const push = (minDist - dist)
        a.px += (dx / dist) * push
        a.py += (dy / dist) * push
        // Keep on-canvas (normalized 0.02..0.98)
        a.px = Math.min(Math.max(a.px, 0.02 * w), 0.98 * w)
        a.py = Math.min(Math.max(a.py, 0.02 * h), 0.98 * h)
        moved = true
      }
    }
    if (!moved) break
  }
  return imgs.map((img, i) => (
    pts[i].fixed ? img : { ...img, x: pts[i].px / w, y: pts[i].py / h }
  ))
}

export default function ClusterView({
  images = [],
  jobId,
  interval,
  totalIntervals,
  isFinal,
  onImages,
  onBack,
  onNext,
  sessionPrompt = '',
  onBoard = () => {},
  editProvider = 'reve',
  onEditProviderChange = () => {},
  // Frozen (faded) images are owned by App so they survive the round-trip
  // through EditView. We seed local state from the last known value on
  // mount and mirror every write back up via onFrozenImagesChange.
  initialFrozenImages = [],
  onFrozenImagesChange = () => {},
  // Undo/redo stack is owned by App so it survives an EditView round-trip.
  // We seed our local state from these on mount and mirror every write
  // back via onHistoryChange(history, ptr).
  initialHistory = [],
  initialHistoryPtr = -1,
  onHistoryChange = () => {},
  tourMode = false,
  onTourModeChange = () => {},
  theme,
  onToggleTheme,
  // Refined-render provenance restored by App from the session-state endpoint
  // ({particleIndex: url}); seeds refinedUrls so REFINED badges + compare
  // survive a page refresh. Additive - defaults keep live behavior identical.
  initialRefinedUrls = {},
  // True when App rebuilt this canvas from a saved session (page refresh /
  // shared link) — drives a one-time plain-words explanation of the dimmed
  // earlier-round images.
  restored = false,
}) {
  // SESSION RESTORED banner (critic-6 #3): shown briefly after a restore, but
  // must NOT stay pinned across round transitions. Auto-dismiss after a few
  // seconds AND on the next round change so it never stacks with round banners.
  const [restoreNoticeDismissed, setRestoreNoticeDismissed] = useState(false)
  useEffect(() => {
    if (!restored || restoreNoticeDismissed) return
    const t = setTimeout(() => setRestoreNoticeDismissed(true), 6000)
    return () => clearTimeout(t)
  }, [restored, restoreNoticeDismissed])
  // Dismiss the instant the round changes (a batch lands / user breeds) so the
  // banner never persists through a transition.
  const restoreDismissIntervalRef = useRef(null)
  useEffect(() => {
    if (restoreDismissIntervalRef.current === null) {
      restoreDismissIntervalRef.current = interval
      return
    }
    if (interval !== restoreDismissIntervalRef.current) {
      restoreDismissIntervalRef.current = interval
      setRestoreNoticeDismissed(true)
    }
  }, [interval])
  const canvasRef = useRef(null)
  const [canvasSize, setCanvasSize] = useState({ w: 0, h: 0 })
  const [drag, setDrag] = useState(null)
  // Selection is keyed by image URL so it works uniformly for active AND
  // frozen images. URL is guaranteed unique across rounds (server paths
  // include interval number).
  const [selectedIds, setSelectedIds] = useState(new Set())  // Set<string> of URLs
  // Save-to-board feedback: 'idle' | 'saving' | 'saved'
  const [boardSaveState, setBoardSaveState] = useState('idle')
  async function handleSaveToBoard() {
    const selected = [...images, ...frozenImages].filter(img => selectedIds.has(img.url))
    if (selected.length === 0 || boardSaveState === 'saving') return
    setBoardSaveState('saving')
    // Each pin must carry a caption: prefer a per-image prompt if the image
    // object already has one, otherwise fall back to the session prompt the
    // cluster was generated from (threaded down from App via sessionPrompt).
    const caption = (sessionPrompt || '').trim()
    try {
      await Promise.all(selected.map(img => pinToBoard(img.url, ((img.prompt || '').trim()) || caption)))
      setBoardSaveState('saved')
      setTimeout(() => setBoardSaveState('idle'), 1800)
      setSelectedIds(new Set())  // clear selection so the count/feedback updates
    } catch (e) {
      console.error('Save to board failed:', e)
      setBoardSaveState('idle')
    }
  }
  const [branching, setBranching] = useState(false)
  // REFINE: per-image full-quality render. refining = shimmer while the server
  // integrates the full fine schedule; refinedUrls = swapped-in image sources.
  const [refining, setRefining] = useState({})        // {imgId: true}
  // Mousedown origin on a REFINE button — a drag that starts there must not
  // fire refine on release (click-after-drag guard).
  const refineBtnDown = useRef(null)
  const [refinedUrls, setRefinedUrls] = useState(() => ({ ...initialRefinedUrls }))  // {imgId: url}
  // Live refine % from the server (real per-step; refines are serialized so
  // one number is enough).
  const [refinePct, setRefinePct] = useState(null)
  // Which image URLs have finished loading in the browser — drives the
  // "LOADING PREVIEWS m/n" pill and per-tile skeletons (the network fetch of
  // the previews is itself a wait the artist can see and count).
  const [loadedUrls, setLoadedUrls] = useState(() => new Set())
  const markLoaded = useCallback((url) => {
    setLoadedUrls(prev => {
      if (prev.has(url)) return prev
      const next = new Set(prev); next.add(url); return next
    })
  }, [])
  // HONEST end-state label: a restored session whose worker is gone
  // (cancelled by the 5-min selection timeout / superseded / stale-from-disk)
  // must NOT claim "GENERATION COMPLETE". Derived from the session-state
  // endpoint using the session id embedded in the image URLs.
  const [sessionEnded, setSessionEnded] = useState(false)
  useEffect(() => {
    if (!isFinal || images.length === 0) { setSessionEnded(false); return }
    const m = images[0]?.url?.match(/^\/images\/([^/]+)\//)
    if (!m) return
    let on = true
    getSessionState(m[1]).then(st => {
      if (on) setSessionEnded(['cancelled', 'error', 'stale'].includes(st.status))
    }).catch(() => {})
    return () => { on = false }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isFinal, images])
  const [branchStatus, setBranchStatus] = useState('')
  // Dismissable, plain-words error toast for a failed breed round (the old
  // code stashed the error in branchStatus, which nothing rendered once the
  // overlay closed — failures were INVISIBLE).
  const [branchError, setBranchError] = useState('')
  const [branchProgress, setBranchProgress] = useState(0)
  const [repulsion, setRepulsion] = useState(0)
  const [cohesion, setCohesion] = useState(0)
  // cohesionSpring kept in state (always 0) so the backend API call signature
  // and props stay unchanged if we re-enable the spring slider later.
  const [cohesionSpring] = useState(0)
  const adjustTimer = useRef(null)
  const adjustSeq = useRef(0)
  const [adjusting, setAdjusting] = useState(false)
  // Progressive disclosure: hide /LAYOUT + /SHORTCUTS behind an OPTIONS toggle
  // so a first-timer sees just selection state + the primary action.
  const [optionsOpen, setOptionsOpen] = useState(false)
  const [draggingImg, setDraggingImg] = useState(null) // {id, type:'active'|'frozen', startMouse, startPos}
  // SYNCHRONOUS drag-vs-click guard (critic-5 #2: a drag must NEVER toggle
  // selection). The `moved` flag on draggingImg is React state and the window
  // mouseup handler nulls draggingImg BEFORE the tile's onClick fires, so the
  // click sometimes read moved=false and selected a just-dragged tile. This
  // ref is set the instant the pointer crosses the movement threshold and is
  // read (then reset) by onClick — it survives the mouseup→click sequence.
  const dragMovedRef = useRef(false)
  const DRAG_SELECT_THRESHOLD = 4  // px in screen space before a press is a drag
  // Frozen images from previous intervals (unselected, kept on canvas)
  // Seed from the App-held snapshot so frozen images survive an EditView
  // round trip. Writes go through a wrapper that also notifies the parent.
  const [frozenImages, _setFrozenImagesRaw] = useState(() =>
    initialFrozenImages.map(img => ({ ...img }))
  )
  const setFrozenImages = useCallback((updater) => {
    _setFrozenImagesRaw(prev => {
      const next = typeof updater === 'function' ? updater(prev) : updater
      onFrozenImagesChange(next)
      return next
    })
  }, [onFrozenImagesChange])
  // Saved parent positions: maps selection order -> {x, y} for position inheritance
  const parentPositions = useRef(null) // array of {x, y} or null

  // Zoom & pan state
  const [zoom, setZoom] = useState(1)
  const [pan, setPan] = useState({ x: 0, y: 0 })
  const [panning, setPanning] = useState(false)
  const panStart = useRef(null)

  // Magnifier (loupe) — toggleable follow-the-cursor lens
  const [loupeOn, setLoupeOn] = useState(false)
  const [loupePos, setLoupePos] = useState(null) // { x, y } in canvas-client coords, null when outside

  // Lightbox — full-size viewer for a tile (⤢ button or double-click).
  // Index into `images` (active round only — those are the pickable ones).
  const [lightboxIdx, setLightboxIdx] = useState(null)

  // Record the live session id (from the served image path
  // '/images/<session>/...') so BoardView can scope its default view to
  // this session's pins.
  useEffect(() => {
    const m = images[0]?.url?.match(/^\/images\/([^/]+)\//)
    if (m) {
      try { localStorage.setItem('ltnt.current_session', m[1]) } catch {}
    }
  }, [images])

  // ── SPAN MODE (round 1): live streaming of the prompt's space ─────────────
  // While the server streams span batches, this view polls the job and folds
  // each new batch onto the canvas (already-shown images NEVER move — user
  // drags included). A meter narrates novelty; "SPREAD LOOKS GOOD" ends the
  // round early via POST /finish_round. When the job flips to
  // waiting_selection the normal picking flow takes over.
  const [span, setSpan] = useState(null)
  const [spanStopping, setSpanStopping] = useState(false)
  // Live per-step paint progress between batch landings (the backend already
  // publishes "painting images 9–16 (solve step k/14)" + a real percent on
  // the job while a batch integrates — surface it so the strip is never a
  // 15-25s static line of dead air).
  const [spanPaint, setSpanPaint] = useState(null)  // {stage, pct} | null
  // ZERO-DEAD-AIR (critic-5 #1): a mid-stream page refresh restores the canvas
  // header instantly but the live job's progress strip is blank until the
  // first poll reattaches (~15-20s). Show a RECONNECTING strip from the FIRST
  // restored frame: true whenever we were restored into a non-final round and
  // no live poll has landed yet; cleared on the first successful tick.
  const [reconnecting, setReconnecting] = useState(() => !!restored && !isFinal)
  const spanLive = !!(span && span.active)
  const imagesRef = useRef(images)
  useEffect(() => { imagesRef.current = images }, [images])
  const canvasSizeRef = useRef(canvasSize)
  useEffect(() => { canvasSizeRef.current = canvasSize }, [canvasSize])
  useEffect(() => {
    if (!jobId || isFinal) return
    let stopped = false
    async function tick() {
      if (stopped) return
      try {
        const d = await getJobStatus(jobId)
        if (stopped) return
        // First live poll landed — the strip below now has real data.
        setReconnecting(false)
        if (d.span) setSpan(d.span)
        const streaming = d.span && d.span.active && d.status === 'running'
        // Between landings the job's stage/progress ARE the live painting
        // telemetry (per solve step). Show them; clear once the batch lands.
        if (streaming && typeof d.stage === 'string' && d.stage.startsWith('painting')) {
          setSpanPaint({ stage: d.stage, pct: Math.min(99, d.progress ?? 0) })
        } else {
          setSpanPaint(null)
        }
        if (streaming || (d.span && d.status === 'waiting_selection')) {
          const serverImgs = d.images || []
          if (serverImgs.length !== imagesRef.current.length) {
            // Merge: keep the position of anything already on the canvas
            // (server coords are only trusted for NEW images) — zero reshuffle.
            const prevByUrl = new Map(imagesRef.current.map(ci => [ci.url, ci]))
            let merged = serverImgs.map(si => {
              const cur = prevByUrl.get(si.url)
              return cur ? { ...si, x: cur.x, y: cur.y, _placed: true } : { ...si }
            })
            // Collision-avoid the NEW tiles only: push them clear of every
            // already-placed tile (user drags included) so batches never
            // stack into an unclickable pile at the canvas center. Placed
            // tiles NEVER move.
            merged = spreadNewTiles(merged, canvasSizeRef.current)
            merged = merged.map(({ _placed, ...img }) => img)
            onImages(merged, jobId, d.interval || 1, d.total_intervals, false)
          }
        }
        if (streaming) {
          setTimeout(tick, 1200)
        } else if (d.span) {
          // Span round over (presented for picking / moved on) — release the
          // reconnect pointer so a refresh doesn't bounce through GenerateView.
          try { localStorage.removeItem('ltnt_active_job') } catch {}
          setSpanStopping(false)
        }
      } catch {
        if (!stopped) setTimeout(tick, 3000)
      }
    }
    tick()
    return () => { stopped = true }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId])

  // First span batch lands via the GenerateView handoff (no merge pass), so
  // its tiles carry raw backend MDS coords that can overlap at tile scale.
  // De-overlap them ONCE, immediately at mount — before the user can have
  // touched anything — so no batch ever piles up (critic blocker #4a).
  const spanInitSpreadRef = useRef(null)
  useEffect(() => {
    // Runs for a LIVE first batch and for a restored/settled span canvas
    // (both arrive with raw backend coords). Once per job, always before any
    // user interaction is possible.
    const spanCanvas = spanLive || (!!span && interval === 1 && !isFinal)
    if (!spanCanvas || images.length === 0 || !canvasSize.w || !canvasSize.h) return
    if (spanInitSpreadRef.current === jobId) return
    spanInitSpreadRef.current = jobId
    const total = images.length + frozenImages.length
    const sizeForPx = (img) => getImgSize(total) * (img.size ?? 1.0)
    const spread = deOverlap(images.map(i => ({ ...i })), canvasSize, sizeForPx)
    onImages(spread, jobId, interval, totalIntervals, isFinal)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [spanLive, images, canvasSize, jobId])

  async function handleSpreadLooksGood() {
    if (!jobId || spanStopping) return
    setSpanStopping(true)
    try { await finishRound(jobId) } catch (e) {
      console.error('finish_round failed:', e)
      setSpanStopping(false)
    }
  }

  const tour = useTour('ltnt.tour.cluster', { autoOpen: tourMode })

  // ── Undo/redo history ─────────────────────────────────────────────────────
  // Frontend-only — each snapshot captures the canvas state after a branch.
  // Seeded from App-held state so it survives an EditView round-trip; all
  // writes are mirrored back up via onHistoryChange(history, ptr).
  // The server's sampler is always at the "latest" round, so branching from
  // a past snapshot is disabled (user must redo to latest first).
  const [history, _setHistoryRaw] = useState(() => initialHistory.map(s => ({ ...s })))
  const [historyPtr, _setHistoryPtrRaw] = useState(() => initialHistoryPtr)
  const setHistory = useCallback((updater) => {
    _setHistoryRaw(prev => {
      const next = typeof updater === 'function' ? updater(prev) : updater
      onHistoryChange(next, historyPtrRef.current)
      return next
    })
  }, [onHistoryChange])
  const setHistoryPtr = useCallback((updater) => {
    _setHistoryPtrRaw(prev => {
      const next = typeof updater === 'function' ? updater(prev) : updater
      historyPtrRef.current = next
      onHistoryChange(historyRef.current, next)
      return next
    })
  }, [onHistoryChange])
  const historyRef = useRef(history)
  const historyPtrRef = useRef(historyPtr)
  useEffect(() => { historyRef.current = history }, [history])
  useEffect(() => { historyPtrRef.current = historyPtr }, [historyPtr])
  const atLatest = historyPtr === history.length - 1 || history.length === 0

  // Seed the history with the initial round on mount (only when App hasn't
  // already given us a snapshot stack).
  useEffect(() => {
    if (history.length === 0 && images.length > 0) {
      const seed = [{
        images: images.map(img => ({ ...img })),
        frozenImages: [],
        interval, totalIntervals, isFinal,
        parentPositions: null,
      }]
      _setHistoryRaw(seed)
      _setHistoryPtrRaw(0)
      historyRef.current = seed
      historyPtrRef.current = 0
      onHistoryChange(seed, 0)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  function pushSnapshot(snap) {
    const truncated = historyPtrRef.current >= 0
      ? historyRef.current.slice(0, historyPtrRef.current + 1)
      : []
    const next = [...truncated, snap]
    const nextPtr = next.length - 1
    _setHistoryRaw(next)
    _setHistoryPtrRaw(nextPtr)
    historyRef.current = next
    historyPtrRef.current = nextPtr
    onHistoryChange(next, nextPtr)
  }

  function loadSnapshot(idx) {
    const snap = history[idx]
    if (!snap) return
    setFrozenImages(snap.frozenImages.map(img => ({ ...img })))
    parentPositions.current = snap.parentPositions
    setSelectedIds(new Set())
    onImages(
      snap.images.map(img => ({ ...img })),
      jobId, snap.interval, snap.totalIntervals, snap.isFinal,
    )
  }

  function undo() {
    if (historyPtr > 0) {
      const p = historyPtr - 1
      setHistoryPtr(p)
      loadSnapshot(p)
    }
  }
  function redo() {
    if (historyPtr < history.length - 1) {
      const p = historyPtr + 1
      setHistoryPtr(p)
      loadSnapshot(p)
    }
  }

  // ── Live layout sliders: debounced + stale-response-safe ──────────────────
  // Called on every slider onChange. Debounces for ~120 ms then fires a single
  // /adjust request with the latest values. If multiple requests overlap
  // (adjust is slow), only the newest response is applied.
  function scheduleAdjust(r, c, cs) {
    if (!jobId || !atLatest) return
    if (adjustTimer.current) clearTimeout(adjustTimer.current)
    adjustTimer.current = setTimeout(async () => {
      const mySeq = ++adjustSeq.current
      setAdjusting(true)
      try {
        const data = await adjustLayout(jobId, r, c, cs)
        if (mySeq !== adjustSeq.current) return  // stale
        if (data.images && data.images.length > 0) {
          onImages(data.images, jobId, interval, totalIntervals, isFinal)
        }
        if (data.frozen_updates && data.frozen_updates.length > 0) {
          const urlMap = {}
          data.frozen_updates.forEach(u => { if (u.url) urlMap[u.url] = { x: u.x, y: u.y } })
          setFrozenImages(prev => prev.map(img => {
            const u = urlMap[img.url]
            return u ? { ...img, x: u.x, y: u.y } : img
          }))
        }
      } catch (err) {
        console.error('Adjust failed:', err)
      } finally {
        if (mySeq === adjustSeq.current) setAdjusting(false)
      }
    }, 120)
  }

  // Track canvas size
  useEffect(() => {
    const el = canvasRef.current
    if (!el) return
    const ro = new ResizeObserver(([entry]) => {
      setCanvasSize({ w: entry.contentRect.width, h: entry.contentRect.height })
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  // ── Keyboard shortcuts ────────────────────────────────────────────────────
  useEffect(() => {
    function onKeyDown(e) {
      if ((e.ctrlKey || e.metaKey) && e.key === 'a') {
        e.preventDefault()
        setSelectedIds(new Set(images.map(img => img.url)))
      }
      // Shift+0 — fit view (reset zoom/pan)
      if (e.shiftKey && (e.key === ')' || e.key === '0')) {
        setZoom(1)
        setPan({ x: 0, y: 0 })
      }
      // M — toggle magnifier
      if ((e.key === 'm' || e.key === 'M') && !e.ctrlKey && !e.metaKey && !e.shiftKey) {
        // Don't intercept if typing in a text input
        const tag = (e.target && e.target.tagName) || ''
        if (tag !== 'INPUT' && tag !== 'TEXTAREA') {
          setLoupeOn(v => !v)
        }
      }
      // Cmd/Ctrl+Z — undo, Cmd/Ctrl+Shift+Z — redo
      if ((e.ctrlKey || e.metaKey) && (e.key === 'z' || e.key === 'Z')) {
        e.preventDefault()
        if (e.shiftKey) redo(); else undo()
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [images, historyPtr, history])

  // ── Scroll to zoom ────────────────────────────────────────────────────────
  useEffect(() => {
    const el = canvasRef.current
    if (!el) return
    function onWheel(e) {
      e.preventDefault()
      const rect = el.getBoundingClientRect()
      const mx = e.clientX - rect.left
      const my = e.clientY - rect.top

      const factor = e.deltaY < 0 ? 1.05 : 1 / 1.05
      const newZoom = Math.min(Math.max(zoom * factor, 0.3), 10)

      // Zoom centered on cursor position
      const scale = newZoom / zoom
      setPan(p => ({
        x: mx - scale * (mx - p.x),
        y: my - scale * (my - p.y),
      }))
      setZoom(newZoom)
    }
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => el.removeEventListener('wheel', onWheel)
  }, [zoom])

  // ── Convert screen coords to canvas-space coords ─────────────────────────
  function screenToCanvas(sx, sy) {
    return {
      x: (sx - pan.x) / zoom,
      y: (sy - pan.y) / zoom,
    }
  }

  // ── Fit-to-view ───────────────────────────────────────────────────────────
  // Compute the bounding box of all current tiles (active + frozen) in canvas-
  // pixel space and set zoom + pan so they fill the viewport with a margin.
  // This is the same intent as the Shift+0 shortcut, but instead of snapping to
  // 100%/origin it scales to the actual content so a fresh round of 6 images
  // shows up large and centered. Clamped to the existing zoom range (0.3..10).
  const fitToView = useCallback((imgsForFit, frozenForFit) => {
    const { w, h } = canvasSize
    const all = [...imgsForFit, ...frozenForFit]
    if (!w || !h || all.length === 0) return
    const total = imgsForFit.length + frozenForFit.length
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity
    all.forEach(img => {
      const cx = img.x * w
      const cy = img.y * h
      const s = getImgSize(total) * (img.size ?? 1.0)
      minX = Math.min(minX, cx - s / 2); maxX = Math.max(maxX, cx + s / 2)
      minY = Math.min(minY, cy - s / 2); maxY = Math.max(maxY, cy + s / 2)
    })
    const bw = Math.max(maxX - minX, 1)
    const bh = Math.max(maxY - minY, 1)
    const MARGIN = 0.88   // leave ~12% breathing room around the content
    const z = Math.min(Math.max(Math.min((w * MARGIN) / bw, (h * MARGIN) / bh), 0.3), 10)
    // Center the bounding box in the viewport.
    const cx = (minX + maxX) / 2
    const cy = (minY + maxY) / 2
    setZoom(z)
    setPan({ x: w / 2 - z * cx, y: h / 2 - z * cy })
  }, [canvasSize])

  // Auto fit-to-view when a NEW round of images arrives (and on first arrival),
  // so the artist immediately sees comfortably-large images without zooming.
  // Keyed on the active-image URL set + canvas size: the effect only re-fits
  // when the round actually changes, so it never fights a user's pan/zoom mid-
  // round (dragging tiles / moving sliders keeps the same URLs => no re-fit).
  const lastFitKeyRef = useRef('')
  // SPAN camera contract (critic blocker #4b): fit ONCE on the first batch,
  // then never re-fit per batch — only expand (zoom out) when a new tile
  // actually lands outside the current viewport. Placed tiles must not
  // visually drift (90→94→97→98% refits).
  const spanFitDoneRef = useRef(false)
  useEffect(() => { spanFitDoneRef.current = false }, [jobId])
  const zoomRef = useRef(zoom); useEffect(() => { zoomRef.current = zoom }, [zoom])
  const panRef = useRef(pan); useEffect(() => { panRef.current = pan }, [pan])
  useEffect(() => {
    // Wait until the canvas has REAL dimensions AND images exist — on a
    // session restore this effect can fire before the ResizeObserver has
    // measured the canvas (mount race -> "empty" canvas until Shift+0).
    if (!canvasSize.w || !canvasSize.h || images.length === 0) return
    const key = `${canvasSize.w}x${canvasSize.h}|${images.map(i => i.url).join(',')}`
    if (key === lastFitKeyRef.current) return
    // Fit on the next frame so layout is committed; the key is only marked
    // fitted once the fit actually ran (guards against a cancelled frame).
    const raf = requestAnimationFrame(() => {
      lastFitKeyRef.current = key
      if (spanLive && spanFitDoneRef.current) {
        // Mid-span batch landing: keep the camera unless something new is
        // off-screen — then (and only then) zoom out to contain everything.
        const { w, h } = canvasSize
        const z = zoomRef.current, p = panRef.current
        const total = images.length + frozenImages.length
        const anyOutside = images.some(img => {
          const s = getImgSize(total) * (img.size ?? 1.0)
          const sx = img.x * w * z + p.x
          const sy = img.y * h * z + p.y
          const half = (s * z) / 2
          return sx - half < 0 || sx + half > w || sy - half < 0 || sy + half > h
        })
        if (!anyOutside) return
      }
      fitToView(images, frozenImages)
      if (spanLive) spanFitDoneRef.current = true
    })
    return () => cancelAnimationFrame(raf)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [images, canvasSize, fitToView, spanLive])

  // ── Mouse handlers (pan vs. drag-select) ──────────────────────────────────
  function canvasXY(e) {
    const rect = canvasRef.current.getBoundingClientRect()
    return { x: e.clientX - rect.left, y: e.clientY - rect.top }
  }

  function onMouseDown(e) {
    if (e.button !== 0) return
    const { x, y } = canvasXY(e)

    if (e.shiftKey) {
      // Shift+drag = select region
      const c = screenToCanvas(x, y)
      setDrag({ x0: c.x, y0: c.y, x1: c.x, y1: c.y, sx0: x, sy0: y, sx1: x, sy1: y })
    } else {
      // Plain drag = pan
      setPanning(true)
      panStart.current = { x: x - pan.x, y: y - pan.y }
    }
  }

  function onMouseMove(e) {
    const { x, y } = canvasXY(e)
    if (loupeOn) setLoupePos({ x, y })

    // Image dragging takes priority
    if (draggingImg) {
      const dx = (e.clientX - draggingImg.startMouse.x) / zoom / canvasSize.w
      const dy = (e.clientY - draggingImg.startMouse.y) / zoom / canvasSize.h
      const newX = draggingImg.startPos.x + dx
      const newY = draggingImg.startPos.y + dy

      // Threshold measured in raw SCREEN px (not canvas-scaled) so a real drag
      // is detected consistently at any zoom. Set the synchronous ref the
      // instant we cross it — onClick reads this, not the async `moved` state.
      const screenDx = e.clientX - draggingImg.startMouse.x
      const screenDy = e.clientY - draggingImg.startMouse.y
      if (Math.abs(screenDx) > DRAG_SELECT_THRESHOLD ||
          Math.abs(screenDy) > DRAG_SELECT_THRESHOLD) {
        dragMovedRef.current = true
        setDraggingImg(d => ({ ...d, moved: true }))
      }

      if (draggingImg.type === 'active') {
        onImages(
          images.map(img => img.id === draggingImg.idx ? { ...img, x: newX, y: newY } : img),
          jobId, interval, totalIntervals, isFinal
        )
      } else {
        setFrozenImages(prev => prev.map((img, i) => {
          const matches = draggingImg.url ? img.url === draggingImg.url : i === draggingImg.idx
          return matches ? { ...img, x: newX, y: newY } : img
        }))
      }
      return
    }

    if (panning && panStart.current) {
      setPan({ x: x - panStart.current.x, y: y - panStart.current.y })
      return
    }

    if (drag) {
      const c = screenToCanvas(x, y)
      setDrag(d => ({ ...d, x1: c.x, y1: c.y, sx1: x, sy1: y }))
    }
  }

  const onMouseUp = useCallback(() => {
    if (draggingImg) {
      setDraggingImg(null)
      return
    }

    if (panning) {
      setPanning(false)
      panStart.current = null
      return
    }

    if (!drag) return

    const selLeft   = Math.min(drag.x0, drag.x1)
    const selRight  = Math.max(drag.x0, drag.x1)
    const selTop    = Math.min(drag.y0, drag.y1)
    const selBottom = Math.max(drag.y0, drag.y1)

    // Check if drag was meaningful (> 10px in screen space)
    const screenDx = Math.abs(drag.sx1 - drag.sx0)
    const screenDy = Math.abs(drag.sy1 - drag.sy0)

    if (screenDx > 10 || screenDy > 10) {
      const newSelected = new Set()
      const all = [...images, ...frozenImages]
      all.forEach(img => {
        const cx = img.x * canvasSize.w
        const cy = img.y * canvasSize.h
        const s = getImgSize(images.length + frozenImages.length) * (img.size ?? 1.0)
        if (cx + s / 2 > selLeft && cx - s / 2 < selRight &&
            cy + s / 2 > selTop && cy - s / 2 < selBottom) {
          newSelected.add(img.url)
        }
      })
      setSelectedIds(newSelected)
    }

    setDrag(null)
  }, [drag, panning, draggingImg, images, frozenImages, canvasSize])

  useEffect(() => {
    window.addEventListener('mouseup', onMouseUp)
    return () => window.removeEventListener('mouseup', onMouseUp)
  }, [onMouseUp])

  // ── Refine: request a full-quality render of one particle ────────────────
  // Refined URLs are per-round (indices shift each round) — reset when a new
  // round of images arrives.
  const roundKey = images.map(i => i.url).join(',')
  // Reset on round CHANGE only — NOT on mount, or a restored session's seeded
  // refinedUrls (initialRefinedUrls, from the rehydration state) is wiped.
  const roundKeyRef = useRef(null)
  useEffect(() => {
    if (roundKeyRef.current === null) { roundKeyRef.current = roundKey; return }
    if (roundKeyRef.current === roundKey) return
    roundKeyRef.current = roundKey
    setRefining({}); setRefinedUrls({})
  }, [roundKey])

  // Clear the round-1 SPAN state once we leave round 1 (critic-5 #4): the
  // green "…hit the 64-image cap" spread banner is gated on span+interval===1,
  // but stale `span` state let it flash into round-2 breeding. Drop it the
  // moment interval advances so no round-1 banner survives a round change.
  useEffect(() => {
    if (interval > 1 && span) { setSpan(null); setSpanPaint(null) }
  }, [interval, span])

  async function handleRefine(e, img) {
    e.stopPropagation()
    if (refining[img.id] || refinedUrls[img.id] || isFinal || branching || !atLatest || spanLive) return
    setRefining(prev => ({ ...prev, [img.id]: true }))
    setRefinePct(0)
    const clear = () => {
      setRefinePct(null)
      setRefining(prev => {
        const next = { ...prev }; delete next[img.id]; return next
      })
    }
    try {
      await requestRefine(jobId, [img.id])
      // Poll until the refined URL shows up in job state (the worker refines
      // then returns to waiting_selection). refine_progress is the server's
      // REAL per-integration-step fraction for the in-flight refine.
      const poll = async () => {
        try {
          const data = await getJobStatus(jobId)
          if (data.refine_progress != null) setRefinePct(data.refine_progress)
          const url = data.refined && data.refined[String(img.id)]
          if (url) {
            setRefinedUrls(prev => ({ ...prev, [img.id]: url }))
            clear()
          } else if (data.status === 'waiting_selection' || data.status === 'running') {
            setTimeout(poll, 700)
          } else {
            clear()  // job moved on (done/error) — drop the shimmer
          }
        } catch { clear() }
      }
      setTimeout(poll, 700)
    } catch (err) {
      console.error('Refine failed:', err)
      clear()
    }
  }

  // ── Branch: send selection -> resume polling ──────────────────────────────
  async function handleBranch() {
    if (branching || spanLive) return   // span still streaming — picks open when it settles
    const activeSelected = images.filter(img => selectedIds.has(img.url))
    if (activeSelected.length === 0) return
    setBranching(true)
    setBranchStatus('Sending selection...')
    setBranchProgress(0)

    try {
      const indices = images
        .map((img, i) => (selectedIds.has(img.url) ? i : -1))
        .filter(i => i >= 0)

      // Freeze unselected images before advancing.
      // Dedup by URL so no image can be accidentally removed or duplicated
      // across multiple branch rounds — frozen history is append-only.
      const unselected = images
        .filter(img => !selectedIds.has(img.url))
        .map(img => ({ ...img, frozen: true }))
      setFrozenImages(prev => {
        const existingUrls = new Set(prev.map(f => f.url))
        const fresh = unselected.filter(u => !existingUrls.has(u.url))
        return [...prev, ...fresh]
      })

      // Save positions of selected images (in selection order)
      // These become the anchor positions for the next interval
      const selectedImgs = indices.map(i => images[i]).filter(Boolean)
      parentPositions.current = selectedImgs.map(img => ({ x: img.x, y: img.y }))

      await submitSelection(jobId, indices)

      setBranchStatus('Branching \u2014 generating new variations...')

      const data = await pollJob(jobId, (status, d) => {
        if (d?.stage) setBranchStatus(d.stage)
        if (d?.progress != null) setBranchProgress(d.progress)
      })

      setBranching(false)
      setBranchStatus('')
      setBranchProgress(0)
      setSelectedIds(new Set())

      const final = data.status === 'done'
      let imgs = final ? data.result.images : data.images

      // Apply position inheritance: anchors keep their parent's (user-adjusted)
      // position, clones get offset relative to that inherited position.
      // Runs in both tree and normal incremental-layout modes — guarantees that
      // positions from the previous round (UMAP + any drags / slider moves)
      // are preserved, with no server-side re-normalization or auto-repulsion.
      if (parentPositions.current && imgs.length > 0) {
        imgs = applyParentPositions(imgs, parentPositions.current)
      }

      // frozenImages in the closure is pre-branch; we append the just-frozen
      // unselected to match what the user now sees. Computed before the
      // de-overlap pass so new tiles can also be spread clear of frozen ones.
      const newFrozen = (() => {
        const existingUrls = new Set(frozenImages.map(f => f.url))
        return [...frozenImages, ...unselected.filter(u => !existingUrls.has(u.url))]
      })()

      // De-overlap pass: spawned siblings land on near-identical positions
      // (tiny backend offsets) and stack on top of each other. Nudge any tiles
      // that overlap apart so every child is individually inspectable. We
      // de-overlap the new active tiles together with the frozen ones (so new
      // children don't bury history), then keep only the active slice — frozen
      // positions are owned by their own state and left untouched.
      if (canvasSize.w && imgs.length > 0) {
        const total = imgs.length + newFrozen.length
        const sizeForPx = (img) => getImgSize(total) * (img.size ?? 1.0)
        const spread = deOverlap([...imgs, ...newFrozen], canvasSize, sizeForPx)
        imgs = spread.slice(0, imgs.length)
      }

      onImages(imgs, jobId, data.interval, data.total_intervals, final)
      pushSnapshot({
        images: imgs.map(img => ({ ...img })),
        frozenImages: newFrozen.map(img => ({ ...img })),
        interval: data.interval,
        totalIntervals: data.total_intervals,
        isFinal: final,
        parentPositions: parentPositions.current ? [...parentPositions.current] : null,
      })
    } catch (err) {
      setBranching(false)
      setBranchProgress(0)
      setBranchStatus('')
      // Server errors are already plain-words (_friendly_error) — never a
      // raw traceback. Selection is kept so RETRY can resubmit.
      setBranchError(err.message || "Sorry — that didn't work. Please try again.")
    }
  }

  // ── Derived ───────────────────────────────────────────────────────────────
  // DETERMINISTIC dimmed set (critic-6 #2): the raw frozenImages list can
  // over-count across restores (server-side _all_urls accumulation + the
  // reconnect merge), so the "N DIMMED" tally drifted 81→64→84 on the SAME
  // session. Derive the displayed dimmed tiles = frozen images that are
  // DISTINCT by URL and NOT currently active. This makes both the rendered
  // dimmed tiles and the count = the true number of distinct earlier-round
  // images, identical on every reload.
  const dimmedImages = (() => {
    const activeUrls = new Set(images.map(i => i.url))
    const seen = new Set()
    const out = []
    for (const f of frozenImages) {
      if (!f || !f.url) continue
      if (activeUrls.has(f.url) || seen.has(f.url)) continue
      seen.add(f.url)
      out.push(f)
    }
    return out
  })()

  const selectedImages = [...images, ...dimmedImages].filter(img => selectedIds.has(img.url))
  // Spawn can only use ACTIVE (latest-round) images as seeds — frozen URLs
  // no longer correspond to live latents on the server.
  const spawnableCount = images.filter(img => selectedIds.has(img.url)).length

  // Drag rect in screen space for the overlay
  const selRect = drag ? {
    left:   Math.min(drag.sx0, drag.sx1),
    top:    Math.min(drag.sy0, drag.sy1),
    width:  Math.abs(drag.sx1 - drag.sx0),
    height: Math.abs(drag.sy1 - drag.sy0),
  } : null

  // The round-1 SPAN strip must NOT show once we start breeding the next
  // round (critic-5 #4: the green "…64-image cap" banner flashed into round-2
  // breeding). Gate it off while branching too, not just on interval.
  const spanRound = !!span && interval === 1 && !isFinal && !branching
  const isSketchRound = interval === 1 && !isFinal && !spanRound
  const stageLabel = isFinal
    ? (sessionEnded ? 'ROUND COMPLETE — IMAGES SAVED' : 'FINAL RESULTS')
    : branching
    ? `MAKING ROUND ${Math.min(interval + 1, totalIntervals || interval + 1)} OF ${totalIntervals} \u2014 BREEDING FROM YOUR PICKS`
    : spanLive
    ? `MAPPING YOUR PROMPT'S SPACE \u2014 ${images.length} IMAGES SO FAR`
    : spanRound
    ? `ROUND 1 OF ${totalIntervals} \u2014 THE SPREAD: PICK DIRECTIONS`
    : isSketchRound
    ? `ROUND ${interval} OF ${totalIntervals} \u2014 ROUGH SKETCHES: PICK DIRECTIONS`
    : `ROUND ${interval} OF ${totalIntervals} \u2014 PICK THE ONES YOU LIKE`

  const cursorStyle = panning ? 'grabbing' : drag ? 'crosshair' : 'grab'

  return (
    <div style={{ height: '100vh', display: 'flex', background: 'var(--bg)' }}>

      {/* ── Sidebar ──────────────────────────────────────────────────────── */}
      <aside style={{
        width: '265px',
        flexShrink: 0,
        background: 'var(--panel)',
        borderRight: '1px solid var(--border)',
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden',
      }}>
        <div style={{ padding: '14px 20px 18px', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div style={{ textTransform: 'uppercase' }}>[ x ] LTNT.APP</div>
          <button
            onClick={onToggleTheme}
            title="Toggle light / dark theme"
            style={{
              border: '1px solid var(--border)', background: 'transparent',
              color: 'var(--muted)', padding: '3px 8px', fontSize: '9px',
              letterSpacing: '0.08em', textTransform: 'uppercase', cursor: 'pointer',
            }}
          >
            {theme === 'dark' ? '☀ LIGHT' : '☾ DARK'}
          </button>
        </div>

        {/* Selection panel */}
        <div data-tour="selection-panel" style={{ marginBottom: '18px' }}>
          <SectionLabel>/SELECTION</SectionLabel>
          <div style={{ margin: '0 16px', border: '1px solid var(--border)', background: 'var(--panel-2)', padding: '10px' }}>
            <div style={{
              fontSize: '10px', letterSpacing: '0.06em',
              textTransform: 'uppercase', fontWeight: 700, marginBottom: '10px',
            }}>
              {selectedImages.length > 0
                ? `${selectedImages.length} IMAGE${selectedImages.length === 1 ? '' : 'S'} SELECTED`
                : 'NO SELECTION \u2014 CLICK IMAGES YOU LIKE'}
            </div>
            <div style={{
              display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)',
              gap: '4px', minHeight: '60px',
            }}>
              {selectedImages.slice(0, 9).map(img => (
                <div
                  key={img.url}
                  className={loadedUrls.has(img.url) ? undefined : 'ltnt-tile-loading'}
                  style={{ aspectRatio: '1', overflow: 'hidden', background: 'var(--tile-empty)' }}
                >
                  {/* Same URL as the canvas tile -> served from browser cache
                      once the tile has loaded (no gray re-fetch flash). */}
                  <img
                    src={img.url}
                    alt={`#${img.id}`}
                    decoding="async"
                    style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }}
                    onLoad={() => markLoaded(img.url)}
                    onError={() => markLoaded(img.url)}
                  />
                </div>
              ))}
              {selectedImages.length === 0 && (
                <div style={{
                  gridColumn: '1 / -1', display: 'flex',
                  alignItems: 'center', justifyContent: 'center',
                  fontSize: '10px', color: 'var(--muted)', letterSpacing: '0.06em',
                  textTransform: 'uppercase', height: '60px',
                }}>
                  [ EMPTY ]
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Zoom info */}
        <div style={{ marginBottom: '18px' }}>
          <SectionLabel>/VIEW</SectionLabel>
          <div style={{ padding: '0 20px', fontSize: '11px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ color: 'var(--muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
              {Math.round(zoom * 100)}%
            </span>
            <button
              onClick={() => { setZoom(1); setPan({ x: 0, y: 0 }) }}
              style={{
                border: '1px solid var(--border)', padding: '3px 10px',
                background: 'none', cursor: 'pointer',
                fontSize: '10px', textTransform: 'uppercase', letterSpacing: '0.04em',
              }}
            >
              Reset
            </button>
            <button
              data-tour="magnifier-btn"
              onClick={() => setLoupeOn(v => !v)}
              title="Toggle magnifier (M)"
              style={{
                border: '1px solid var(--border)', padding: '3px 10px',
                background: loupeOn ? 'var(--text)' : 'none',
                color: loupeOn ? 'var(--bg)' : 'var(--text)',
                cursor: 'pointer',
                fontSize: '10px', textTransform: 'uppercase', letterSpacing: '0.04em',
              }}
            >
              {loupeOn ? '⊙ Magnifier On' : '⊙ Magnifier'}
            </button>
          </div>
        </div>

        {/* ── Progressive-disclosure OPTIONS toggle (collapsed by default) ── */}
        <button
          onClick={() => setOptionsOpen(o => !o)}
          style={{
            display: 'block', width: '100%', textAlign: 'left',
            border: 'none', background: 'none', cursor: 'pointer',
            padding: '0 20px', marginBottom: optionsOpen ? '8px' : '18px',
            fontSize: '10px', letterSpacing: '0.08em',
            textTransform: 'uppercase', color: 'var(--muted)',
            fontFamily: 'inherit',
          }}
        >
          [ {optionsOpen ? '−' : '+'} ] OPTIONS
        </button>

        {optionsOpen && (<>
        <div data-tour="layout-controls">
          <SectionLabel>/LAYOUT</SectionLabel>
          <div style={{ padding: '0 16px', marginBottom: '12px' }}>
            <div style={{ fontSize: '11px', color: 'var(--muted)', marginBottom: '4px' }}>
              Spread {Math.round(repulsion * 100)}%
            </div>
            <input
              type="range" min="0" max="100" value={Math.round(repulsion * 100)}
              onChange={e => { const v = e.target.value / 100; setRepulsion(v); scheduleAdjust(v, cohesion, cohesionSpring) }}
              disabled={!atLatest}
              style={{ width: '100%' }}
            />
            <div style={{ fontSize: '11px', color: 'var(--muted)', marginBottom: '4px', marginTop: '8px' }}>
              Group {Math.round(cohesion * 100)}%
            </div>
            <input
              type="range" min="0" max="100" value={Math.round(cohesion * 100)}
              onChange={e => { const v = e.target.value / 100; setCohesion(v); scheduleAdjust(repulsion, v, cohesionSpring) }}
              disabled={!atLatest}
              style={{ width: '100%' }}
            />
            {/*
              Spring cohesion — hidden for now. Backend still receives a
              cohesion_spring value (0) so the API stays stable. To re-enable,
              uncomment this block and switch cohesionSpring back to useState.
              <div style={{ fontSize: '11px', color: '#555', marginBottom: '4px', marginTop: '8px' }}>
                Cohesion (spring) {Math.round(cohesionSpring * 100)}%
              </div>
              <input
                type="range" min="0" max="100" value={Math.round(cohesionSpring * 100)}
                onChange={e => { const v = e.target.value / 100; setCohesionSpring(v); scheduleAdjust(repulsion, cohesion, v) }}
                disabled={!atLatest}
                style={{ width: '100%' }}
              />
            */}
            {adjusting && (
              <div style={{ marginTop: '8px', fontSize: '10px', color: 'var(--muted)',
                letterSpacing: '0.06em', textTransform: 'uppercase' }}>
                Updating layout...
              </div>
            )}
            {!atLatest && (
              <div style={{ marginTop: '8px', fontSize: '10px', color: 'var(--muted)',
                letterSpacing: '0.06em', textTransform: 'uppercase' }}>
                Redo to latest to edit
              </div>
            )}
          </div>
        </div>

        <div>
          <SectionLabel>/SHORTCUTS</SectionLabel>
          {SHORTCUTS.map(s => (
            <SidebarRow key={s.label} label={s.label} right={s.shortcut} />
          ))}
        </div>
        </>)}

        {branching && branchStatus && (
          <div style={{
            margin: '18px 16px', padding: '10px',
            background: 'var(--panel-2)', border: '1px solid var(--border)',
            fontSize: '10px', letterSpacing: '0.06em',
            textTransform: 'uppercase', color: 'var(--muted)',
          }}>
            {branchStatus}
          </div>
        )}
      </aside>

      {/* ── Main area ────────────────────────────────────────────────────── */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>

        {/* Top bar */}
        <div style={{
          height: '56px', display: 'flex', alignItems: 'center',
          justifyContent: 'space-between', padding: '0 20px',
          borderBottom: '1px solid var(--border)', flexShrink: 0,
        }}>
          <div style={{ display: 'flex', gap: '8px' }}>
            <TourToggle tourMode={tourMode} onTourModeChange={onTourModeChange} />
            <button onClick={() => {
              // A running/populated session must never be silently destroyed:
              // one plain confirm; the old session stays reachable (App pushes
              // its ?session= URL into history + lists it under RECENT).
              if (images.length > 0 || frozenImages.length > 0 || spanLive || branching) {
                if (!window.confirm('Start over? Your current images stay saved at this link.')) return
              }
              onBack()
            }} style={{
              border: '1px solid var(--text)', padding: '6px 14px',
              background: 'none', cursor: 'pointer',
            }}>
            ← [ ] NEW PROMPT
          </button>
            <button
              onClick={undo}
              disabled={historyPtr <= 0}
              title="Undo (Cmd/Ctrl+Z)"
              style={{
                border: '1px solid var(--text)', padding: '6px 12px',
                background: 'none', cursor: historyPtr <= 0 ? 'not-allowed' : 'pointer',
                opacity: historyPtr <= 0 ? 0.35 : 1,
                fontSize: '11px', textTransform: 'uppercase', letterSpacing: '0.04em',
              }}
            >
              ↶ Undo
            </button>
            <button
              onClick={redo}
              disabled={historyPtr >= history.length - 1}
              title="Redo (Cmd/Ctrl+Shift+Z)"
              style={{
                border: '1px solid var(--text)', padding: '6px 12px',
                background: 'none',
                cursor: historyPtr >= history.length - 1 ? 'not-allowed' : 'pointer',
                opacity: historyPtr >= history.length - 1 ? 0.35 : 1,
                fontSize: '11px', textTransform: 'uppercase', letterSpacing: '0.04em',
              }}
            >
              ↷ Redo
            </button>
          </div>
          <div style={{
            textTransform: 'uppercase', color: 'var(--muted)',
            letterSpacing: '0.08em', fontSize: '11px',
          }}>
            {/* ONE truthful count (critic-5 #4): the header and the canvas
                footer both read images.length and both say "IN THIS ROUND",
                so they can never disagree (was header "8 IMAGES" vs footer
                "6 IN THIS ROUND"). Hidden while span-live/breeding (their
                own strips carry the live count). */}
            {stageLabel}{spanLive || branching ? '' : ` • ${images.length} IN THIS ROUND`}
          </div>
          <button onClick={onBoard} title="Open your saved board" style={{
            border: '1px solid var(--text)', padding: '6px 14px', background: 'none',
            cursor: 'pointer', fontSize: '11px', textTransform: 'uppercase',
            letterSpacing: '0.04em',
          }}>
            ✦ /BOARD
          </button>
          {isFinal && (
            <div style={{
              fontSize: '10px', letterSpacing: '0.08em',
              textTransform: 'uppercase', color: 'var(--muted)',
            }}>
              {sessionEnded ? 'SESSION ENDED' : 'GENERATION COMPLETE'}
            </div>
          )}
        </div>

        {/* Honest end-state strip: this session's worker is gone (timed out /
            superseded / restored from disk) — say so, and point at the two
            real ways forward instead of claiming completion. */}
        {isFinal && sessionEnded && (
          <div style={{
            padding: '8px 20px', background: 'var(--info)',
            borderBottom: '1px solid var(--border)', flexShrink: 0,
            fontSize: '10px', letterSpacing: '0.06em', textTransform: 'uppercase',
            color: 'var(--text)', display: 'flex', alignItems: 'center', gap: '10px',
          }}>
            <span style={{ fontWeight: 700 }}>✓ ROUND COMPLETE — ALL IMAGES SAVED</span>
            <span style={{ color: 'var(--muted)' }}>
              Keep exploring: save an image to the board and use EXPLORE FROM THIS
              to start a fresh round near it, or begin a new prompt.
            </span>
          </div>
        )}

        {/* Restored-session explainer (critic blocker #5): plain words for
            what the dimmed tiles are and what clicking does — shown once
            after a restore, dismissible. */}
        {restored && !restoreNoticeDismissed && dimmedImages.length > 0 && (
          <div style={{
            padding: '8px 20px', background: 'var(--accent-soft)',
            borderBottom: '1px solid var(--border)', flexShrink: 0,
            fontSize: '10px', letterSpacing: '0.06em', textTransform: 'uppercase',
            color: 'var(--text)', display: 'flex', alignItems: 'center', gap: '10px',
          }}>
            <span style={{ fontWeight: 700 }}>↺ SESSION RESTORED</span>
            <span style={{ color: 'var(--muted)' }}>
              The bright images are your latest round; dimmed ones are earlier
              rounds. Click any image — dimmed too — to select it for saving
              to the board or editing.
            </span>
            <button
              onClick={() => setRestoreNoticeDismissed(true)}
              title="Dismiss"
              style={{
                marginLeft: 'auto', border: 'none', background: 'none',
                color: 'var(--text)', cursor: 'pointer', fontSize: '13px',
              }}
            >
              ✕
            </button>
          </div>
        )}

        {/* RECONNECTING strip (critic-5 #1) — bridges the gap between a
            mid-stream refresh restoring the header and the first live poll
            landing, so the progress area is NEVER blank. Shown only until the
            first tick resolves (then the SPAN / paint strip takes over). */}
        {reconnecting && !spanRound && (
          <div style={{
            padding: '8px 20px', background: 'var(--info)',
            borderBottom: '1px solid var(--border)', flexShrink: 0,
            fontSize: '10px', letterSpacing: '0.06em', textTransform: 'uppercase',
            color: 'var(--accent)', display: 'flex', alignItems: 'center', gap: '12px',
          }}>
            <span style={{ fontWeight: 700 }}>⟳ RECONNECTING TO YOUR RUN…</span>
            <span style={{ color: 'var(--muted)' }}>
              Picking your session back up — the live progress will appear in a moment.
            </span>
          </div>
        )}

        {/* SPAN MODE strip — live novelty meter + early-stop while round 1
            streams, then a "pick your directions" cue once it settles. */}
        {spanRound && (
          <div style={{
            padding: '8px 20px', background: spanLive ? 'var(--info)' : 'var(--good)',
            borderBottom: `1px solid var(--border)`,
            flexShrink: 0, fontSize: '10px', letterSpacing: '0.06em',
            textTransform: 'uppercase',
            color: spanLive ? 'var(--accent)' : 'var(--text)',
            display: 'flex', alignItems: 'center', gap: '12px',
          }}>
            {spanLive ? (
              <>
                <span style={{ fontWeight: 700 }}>◍ MAPPING YOUR PROMPT'S SPACE</span>
                <span>
                  {span.n_images} of up to {span.max_images} images ·
                  batch {span.batch_n} landed
                </span>
                {/* LIVE paint progress between batch landings — the server's
                    real per-solve-step stage + percent, so the strip is never
                    a static line while the next batch integrates. */}
                {spanPaint && (
                  <span style={{
                    display: 'inline-flex', alignItems: 'center', gap: '8px',
                    background: 'var(--panel)',
                    border: '1px solid var(--border)', padding: '2px 10px',
                  }}>
                    <span>⟳ {spanPaint.stage}</span>
                    <span style={{
                      position: 'relative', width: '64px', height: '4px',
                      background: 'var(--panel-2)', display: 'inline-block',
                    }}>
                      <span style={{
                        position: 'absolute', left: 0, top: 0, bottom: 0,
                        width: `${spanPaint.pct}%`, background: 'var(--accent)',
                        transition: 'width 0.4s ease',
                      }} />
                    </span>
                    <span style={{ fontWeight: 700 }}>{spanPaint.pct}%</span>
                  </span>
                )}
                {/* The meter IS the state phrase (plain words > unlabeled
                    bars): the server derives it from the real novelty series
                    — 'still finding new directions' until novelty goes flat
                    for 2 consecutive batches, then it's a good time to pick. */}
                <span style={{
                  fontWeight: 700,
                  color: span.saturating ? 'var(--text)' : 'var(--accent)',
                  background: span.saturating ? 'var(--good)' : 'var(--panel)',
                  border: `1px solid var(--border)`,
                  padding: '2px 10px',
                }}>
                  {spanStopping
                    ? '⏳ finishing after this batch…'
                    : (span.message ||
                      (span.saturating
                        ? 'directions repeating — good time to pick'
                        : 'still finding new directions'))}
                </span>
                <button
                  onClick={handleSpreadLooksGood}
                  disabled={spanStopping}
                  style={{
                    marginLeft: 'auto', border: '1px solid var(--accent)',
                    background: spanStopping ? 'var(--panel-2)' : 'var(--panel)',
                    color: 'var(--accent)', padding: '4px 12px',
                    cursor: spanStopping ? 'default' : 'pointer',
                    fontSize: '10px', letterSpacing: '0.06em',
                    textTransform: 'uppercase', fontWeight: 700,
                  }}
                >
                  {spanStopping ? '⏳ FINISHING…' : '✓ SPREAD LOOKS GOOD — START PICKING'}
                </button>
              </>
            ) : (
              <>
                <span style={{ fontWeight: 700 }}>◍ THE SPREAD</span>
                <span style={{ color: 'var(--muted)' }}>
                  {span.n_images} full quick renders of different directions
                  {span.stop_reason === 'auto' && ' — stopped on its own when new batches stopped covering new ground'}
                  {span.stop_reason === 'user' && ' — you called the spread'}
                  {span.stop_reason === 'cap' && ` — hit the ${span.max_images}-image cap`}
                  . Click the ones worth exploring, then MAKE MORE LIKE THESE.
                </span>
              </>
            )}
          </div>
        )}

        {/* Round-1 sketch strip — sets expectations for the coarse previews.
            Round 1 shows quick sketches of DIFFERENT DIRECTIONS the prompt can
            go; the picks steer the search and every later round gets sharper. */}
        {isSketchRound && (
          <div style={{
            padding: '8px 20px', background: 'var(--warn)',
            borderBottom: '1px solid var(--border)', flexShrink: 0,
            fontSize: '10px', letterSpacing: '0.06em', textTransform: 'uppercase',
            color: 'var(--muted)', display: 'flex', alignItems: 'center', gap: '10px',
          }}>
            <span style={{ fontWeight: 700 }}>✎ COARSE SKETCHES</span>
            <span style={{ color: 'var(--muted)' }}>
              These are quick previews of different directions — not final images.
              Click the ones you like and hit MAKE MORE: each round gets sharper.
            </span>
          </div>
        )}

        {/* Canvas */}
        <main
          ref={canvasRef}
          onMouseDown={onMouseDown}
          onMouseMove={onMouseMove}
          onMouseLeave={() => setLoupePos(null)}
          style={{
            flex: 1, position: 'relative', overflow: 'hidden',
            cursor: cursorStyle,
            userSelect: 'none',
          }}
        >
          {images.length === 0 && (
            <div style={{
              position: 'absolute', inset: 0, display: 'flex',
              alignItems: 'center', justifyContent: 'center',
              flexDirection: 'column', gap: '12px',
              color: 'var(--muted)', fontSize: '11px', letterSpacing: '0.1em',
              textTransform: 'uppercase',
            }}>
              <div style={{ fontSize: '48px', opacity: 0.3 }}>✦</div>
              <div>NO IMAGES</div>
            </div>
          )}

          {/* Zoomable/pannable content layer */}
          <div style={{
            position: 'absolute', inset: 0,
            transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
            transformOrigin: '0 0',
            pointerEvents: 'none',
          }}>
            {/* Frozen (dimmed) images from previous intervals — render the
                deterministic, deduped set so no earlier-round tile is drawn
                twice or drawn while also active (critic-6 #2). */}
            {canvasSize.w > 0 && dimmedImages.map((img, fi) => {
              const cx = img.x * canvasSize.w
              const cy = img.y * canvasSize.h
              const imgSize = getImgSize(images.length + dimmedImages.length) * (img.size ?? 1.0)
              const isSelected = selectedIds.has(img.url)
              return (
                <div
                  key={`frozen-${img.url || fi}`}
                  onMouseDown={e => {
                    e.stopPropagation()
                    dragMovedRef.current = false
                    setDraggingImg({ idx: fi, url: img.url, type: 'frozen', startMouse: { x: e.clientX, y: e.clientY }, startPos: { x: img.x, y: img.y }, moved: false })
                  }}
                  onClick={e => {
                    e.stopPropagation()
                    // A drag NEVER toggles selection (synchronous ref, reset by
                    // this click so the next plain click still selects).
                    if (dragMovedRef.current) { dragMovedRef.current = false; return }
                    setSelectedIds(prev => {
                      const next = new Set(prev)
                      next.has(img.url) ? next.delete(img.url) : next.add(img.url)
                      return next
                    })
                  }}
                  style={{
                    position: 'absolute',
                    left: cx - imgSize / 2,
                    top: cy - imgSize / 2,
                    width: imgSize,
                    height: imgSize,
                    boxShadow: isSelected
                      ? '0 0 0 2.5px var(--text), 0 2px 12px rgba(0,0,0,0.25)'
                      : '0 1px 4px rgba(0,0,0,0.1)',
                    overflow: 'hidden',
                    // When selected, un-fade so the user can clearly see which
                    // history image they picked for editing.
                    opacity: isSelected ? 0.95 : 0.45,
                    filter: isSelected ? 'none' : 'grayscale(40%)',
                    pointerEvents: 'auto',
                    cursor: 'pointer',
                  }}
                >
                  <img
                    src={img.url}
                    alt="frozen"
                    style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }}
                    draggable={false}
                  />
                  {isSelected && (
                    <div style={{
                      position: 'absolute', bottom: '5px', right: '5px',
                      width: '10px', height: '10px', borderRadius: '50%',
                      background: 'var(--text)', border: '1.5px solid #fff',
                    }} />
                  )}
                </div>
              )
            })}

            {/* Active images */}
            {canvasSize.w > 0 && images.map(img => {
              const cx = img.x * canvasSize.w
              const cy = img.y * canvasSize.h
              const isSelected = selectedIds.has(img.url)
              const imgSize = getImgSize(images.length + frozenImages.length) * (img.size ?? 1.0)
              return (
                <div
                  key={img.id}
                  className={loadedUrls.has(refinedUrls[img.id] || img.url)
                    ? 'ltnt-tile'
                    : 'ltnt-tile-loading ltnt-tile'}
                  onMouseDown={e => {
                    e.stopPropagation()
                    dragMovedRef.current = false
                    setDraggingImg({ idx: img.id, type: 'active', startMouse: { x: e.clientX, y: e.clientY }, startPos: { x: img.x, y: img.y }, moved: false })
                  }}
                  onClick={(e) => {
                    e.stopPropagation()
                    // A drag NEVER toggles selection (synchronous ref survives
                    // the mouseup→click race; reset so the next click selects).
                    if (dragMovedRef.current) { dragMovedRef.current = false; return }
                    setSelectedIds(prev => {
                      const next = new Set(prev)
                      next.has(img.url) ? next.delete(img.url) : next.add(img.url)
                      return next
                    })
                  }}
                  onDoubleClick={(e) => {
                    e.stopPropagation()
                    const idx = images.findIndex(i => i.id === img.id)
                    if (idx >= 0) setLightboxIdx(idx)
                  }}
                  style={{
                    position: 'absolute',
                    left: cx - imgSize / 2,
                    top: cy - imgSize / 2,
                    width: imgSize,
                    height: imgSize,
                    boxShadow: isSelected
                      ? '0 0 0 2.5px var(--text), 0 2px 12px rgba(0,0,0,0.25)'
                      : '0 2px 8px rgba(0,0,0,0.18)',
                    cursor: 'grab',
                    overflow: 'hidden',
                    transition: 'box-shadow 0.1s',
                    pointerEvents: 'auto',
                  }}
                >
                  <img
                    src={refinedUrls[img.id] || img.url}
                    alt={`#${img.id}`}
                    style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }}
                    draggable={false}
                    onLoad={() => markLoaded(refinedUrls[img.id] || img.url)}
                    onError={() => markLoaded(refinedUrls[img.id] || img.url)}
                  />
                  {/* REFINE — full-quality render of this preview. EXPLICIT
                      hover affordance ONLY, and only when the tile is big
                      enough that the button cannot be mistaken for the tile
                      itself (on dense span canvases the corner buttons used
                      to cover most of a 44px tile, so a "select" click fired
                      REFINE). Small tiles refine via the lightbox instead. */}
                  {/* Hover ✦ refine affordance on EVERY tile size (critic
                      polish #6): big tiles get the labeled button; small
                      (dense-span) tiles get a compact icon-only ✦ so it can't
                      cover the tile and steal select-clicks. Both are
                      hover-revealed (CSS) with the same drag-guard. */}
                  {!isFinal && !refinedUrls[img.id] && (refining[img.id] || (!branching && atLatest)) && (() => {
                    // ENLARGED hit-target (critic-6 #1). The old ✦ was ~10px —
                    // a few px off SELECTED the tile. Give the button a real
                    // clickable region that scales with the tile but is always
                    // comfortably larger than before (≥24px each side), while
                    // leaving the tile CENTRE clear for select-clicks. The
                    // full "REFINE" label only shows once the tile is big
                    // enough to hold it (≥72px); smaller tiles get a bigger ✦.
                    const showLabel = imgSize >= 72
                    const btnH = showLabel ? 28 : Math.max(24, Math.round(imgSize * 0.34))
                    const btnW = showLabel ? 46 : Math.max(24, Math.round(imgSize * 0.34))
                    const iconPx = imgSize >= 72 ? 13 : Math.max(11, Math.round(imgSize * 0.22))
                    return (
                    <button
                      className="ltnt-refine-btn"
                      onMouseDown={e => {
                        e.stopPropagation()
                        refineBtnDown.current = { x: e.clientX, y: e.clientY }
                      }}
                      onClick={e => {
                        // Drag-guard: a drag that starts on this button must
                        // NOT fire refine on release. stopPropagation on both
                        // mousedown + click means a click ANYWHERE in this
                        // enlarged region refines and NEVER selects the tile.
                        e.stopPropagation()
                        const d = refineBtnDown.current
                        if (d && (Math.abs(e.clientX - d.x) > 4 || Math.abs(e.clientY - d.y) > 4)) {
                          return
                        }
                        handleRefine(e, img)
                      }}
                      disabled={!!refining[img.id] || branching || !atLatest}
                      title="Render this image at full quality"
                      style={{
                        position: 'absolute', top: '3px', left: '3px',
                        display: 'flex', flexDirection: 'column',
                        alignItems: 'center', justifyContent: 'center', gap: '1px',
                        width: `${btnW}px`, height: `${btnH}px`,
                        boxSizing: 'border-box',
                        border: '1px solid var(--text)',
                        background: 'var(--panel)',
                        padding: 0,
                        lineHeight: 1.05,
                        letterSpacing: '0.06em', textTransform: 'uppercase',
                        cursor: refining[img.id] ? 'wait' : 'pointer',
                      }}
                    >
                      {refining[img.id] ? (
                        <span style={{ fontSize: showLabel ? '9px' : '8px', fontWeight: 700 }}>
                          ⟳ {refinePct != null ? refinePct : 0}%
                        </span>
                      ) : (
                        <>
                          <span style={{ fontSize: `${iconPx}px`, lineHeight: 1 }}>✦</span>
                          {/* Hover-revealed label for discoverability. */}
                          {showLabel && (
                            <span className="ltnt-refine-label" style={{
                              fontSize: '7px', fontWeight: 700,
                            }}>
                              REFINE
                            </span>
                          )}
                        </>
                      )}
                    </button>
                    )
                  })()}
                  {/* Enlarge — open the full-size viewer (also: double-click).
                      Hidden on tiny tiles for the same reason as REFINE:
                      single-click on a tile must mean SELECT, period. */}
                  {imgSize >= 72 && (
                  <button
                    className="ltnt-refine-btn"
                    onMouseDown={e => e.stopPropagation()}
                    onClick={e => {
                      e.stopPropagation()
                      const idx = images.findIndex(i => i.id === img.id)
                      if (idx >= 0) setLightboxIdx(idx)
                    }}
                    title="View full size (double-click also works)"
                    style={{
                      position: 'absolute', top: '4px', right: '4px',
                      border: '1px solid var(--text)',
                      background: 'var(--panel)',
                      padding: '2px 6px', fontSize: '9px',
                      letterSpacing: '0.06em', textTransform: 'uppercase',
                      cursor: 'zoom-in',
                    }}
                  >
                    ⤢
                  </button>
                  )}
                  {/* Round-1 tiles are honest sketches — label them as such */}
                  {isSketchRound && !refinedUrls[img.id] && (
                    <div style={{
                      position: 'absolute', bottom: '4px', left: '4px',
                      background: 'var(--warn)', color: 'var(--muted)',
                      border: '1px solid var(--border)',
                      padding: '1px 5px', fontSize: '8px',
                      letterSpacing: '0.08em', textTransform: 'uppercase',
                      pointerEvents: 'none',
                    }}>
                      ✎ SKETCH
                    </div>
                  )}
                  {refining[img.id] && <div className="ltnt-refine-shimmer" />}
                  {/* Visible refine-in-progress overlay (critic-5 #3): a
                      centered spinner + live % on the tile from the FIRST
                      frame after clicking ✦ — not hover-gated, not a late
                      corner badge. Covers every tile size. */}
                  {refining[img.id] && (
                    <div style={{
                      position: 'absolute', inset: 0,
                      display: 'flex', flexDirection: 'column',
                      alignItems: 'center', justifyContent: 'center', gap: '6px',
                      background: 'rgba(26,26,26,0.42)',
                      pointerEvents: 'none',
                    }}>
                      <div className="ltnt-refine-spinner" style={{
                        width: imgSize >= 72 ? '26px' : '16px',
                        height: imgSize >= 72 ? '26px' : '16px',
                        border: '2.5px solid rgba(255,255,255,0.4)',
                        borderTopColor: '#fff', borderRadius: '50%',
                      }} />
                      <div style={{
                        color: '#fff', fontSize: imgSize >= 72 ? '10px' : '8px',
                        fontWeight: 700, letterSpacing: '0.08em',
                        textTransform: 'uppercase',
                        textShadow: '0 1px 2px rgba(0,0,0,0.6)',
                      }}>
                        {imgSize >= 72 ? 'Refining ' : ''}{refinePct != null ? refinePct : 0}%
                      </div>
                    </div>
                  )}
                  {refinedUrls[img.id] && (
                    <div style={{
                      position: 'absolute', top: '4px', left: '4px',
                      background: 'var(--text)', color: 'var(--bg)',
                      padding: '2px 6px', fontSize: '8px',
                      letterSpacing: '0.08em', textTransform: 'uppercase',
                      pointerEvents: 'none',
                    }}>
                      ✦ REFINED
                    </div>
                  )}
                  {isSelected && (
                    <div style={{
                      position: 'absolute', bottom: '5px', right: '5px',
                      width: '10px', height: '10px', borderRadius: '50%',
                      background: 'var(--text)', border: '1.5px solid #fff',
                    }} />
                  )}
                </div>
              )
            })}
          </div>

          {/* Preview image-load pill — the network fetch of a fresh round is a
              real wait; show a live count instead of silent gray cards. */}
          {(() => {
            const total = images.length
            if (total === 0 || branching) return null
            const loaded = images.filter(i => loadedUrls.has(refinedUrls[i.id] || i.url)).length
            if (loaded >= total) return null
            return (
              <div style={{
                position: 'absolute', bottom: '16px', left: '50%',
                transform: 'translateX(-50%)',
                background: 'var(--panel)',
                border: '1px solid var(--border)',
                padding: '6px 14px', zIndex: 15,
                fontSize: '10px', letterSpacing: '0.08em',
                textTransform: 'uppercase', color: 'var(--muted)',
                pointerEvents: 'none',
              }}>
                ⟳ LOADING PREVIEWS {loaded}/{total}
              </div>
            )
          })()}

          {/* Drag selection rectangle (in screen space, above the transform) */}
          {selRect && (
            <div style={{
              position: 'absolute',
              left: selRect.left, top: selRect.top,
              width: selRect.width, height: selRect.height,
              border: '1.5px dashed var(--text)',
              background: 'rgba(26,26,26,0.04)',
              pointerEvents: 'none',
              zIndex: 10,
            }} />
          )}

          {/* Corner label */}
          {images.length > 0 && (
            <div style={{
              position: 'absolute', bottom: '14px', right: '20px',
              fontSize: '10px', letterSpacing: '0.1em',
              textTransform: 'uppercase', color: 'var(--muted)', pointerEvents: 'none',
            }}>
              {images.length} IN THIS ROUND{dimmedImages.length > 0 ? ` \u2022 ${dimmedImages.length} DIMMED (EARLIER ROUNDS)` : ''} • {Math.round(zoom * 100)}%
            </div>
          )}

          {/* Magnifier lens — follows the cursor while toggled on */}
          {loupeOn && loupePos && canvasSize.w > 0 && (() => {
            const mx = loupePos.x, my = loupePos.y
            const mc_x = (mx - pan.x) / zoom
            const mc_y = (my - pan.y) / zoom
            const S = zoom * LOUPE_MAGNIFICATION
            const tx = LOUPE_RADIUS - S * mc_x
            const ty = LOUPE_RADIUS - S * mc_y
            const total = images.length + dimmedImages.length
            const baseSize = getImgSize(total)
            return (
              <div style={{
                position: 'absolute',
                left: mx - LOUPE_RADIUS,
                top: my - LOUPE_RADIUS,
                width: LOUPE_RADIUS * 2,
                height: LOUPE_RADIUS * 2,
                borderRadius: '50%',
                border: '2px solid var(--text)',
                overflow: 'hidden',
                boxShadow: '0 6px 24px rgba(0,0,0,0.3)',
                background: 'var(--bg)',
                pointerEvents: 'none',
                zIndex: 30,
              }}>
                <div style={{
                  position: 'absolute', left: 0, top: 0,
                  width: canvasSize.w, height: canvasSize.h,
                  transform: `translate(${tx}px, ${ty}px) scale(${S})`,
                  transformOrigin: '0 0',
                }}>
                  {dimmedImages.map((img, fi) => {
                    const cx = img.x * canvasSize.w
                    const cy = img.y * canvasSize.h
                    const s = baseSize * (img.size ?? 1.0)
                    return (
                      <img
                        key={`lf-${img.url || fi}`}
                        src={img.url}
                        alt=""
                        draggable={false}
                        style={{
                          position: 'absolute',
                          left: cx - s / 2, top: cy - s / 2,
                          width: s, height: s,
                          objectFit: 'cover',
                          opacity: 0.45,
                          filter: 'grayscale(40%)',
                        }}
                      />
                    )
                  })}
                  {images.map(img => {
                    const cx = img.x * canvasSize.w
                    const cy = img.y * canvasSize.h
                    const s = baseSize * (img.size ?? 1.0)
                    return (
                      <img
                        key={`la-${img.id}`}
                        src={img.url}
                        alt=""
                        draggable={false}
                        style={{
                          position: 'absolute',
                          left: cx - s / 2, top: cy - s / 2,
                          width: s, height: s,
                          objectFit: 'cover',
                        }}
                      />
                    )
                  })}
                </div>
                {/* Crosshair */}
                <div style={{
                  position: 'absolute',
                  left: LOUPE_RADIUS - 5, top: LOUPE_RADIUS - 5,
                  width: 10, height: 10,
                  border: '1px solid rgba(26,26,26,0.6)',
                  borderRadius: '50%',
                }} />
              </div>
            )
          })()}

          {/* Branching progress — compact centered card, no full-canvas
              backdrop, so frozen + active images stay visible the whole time. */}
          {branching && (
            <div style={{
              position: 'absolute',
              top: '50%', left: '50%',
              transform: 'translate(-50%, -50%)',
              background: 'var(--panel)',
              border: '1px solid var(--border)',
              padding: '22px 28px',
              width: '360px',
              zIndex: 20,
              boxShadow: '0 4px 24px rgba(0,0,0,0.14)',
              pointerEvents: 'none',
              textAlign: 'center',
            }}>
              <div style={{
                fontSize: '11px', letterSpacing: '0.1em',
                textTransform: 'uppercase', color: 'var(--muted)',
                marginBottom: '16px',
              }}>
                {branchStatus || 'Branching...'}
              </div>
              <PixelCatProgress progress={branchProgress} />
              <div style={{
                marginTop: '8px', fontSize: '10px',
                letterSpacing: '0.08em', color: 'var(--muted)',
              }}>
                {branchProgress}%
              </div>
            </div>
          )}

          {/* Breed-failure toast — dismissable, plain words, retry keeps the
              same selection. */}
          {branchError && (
            <div style={{
              position: 'absolute', bottom: '16px', left: '50%',
              transform: 'translateX(-50%)', zIndex: 30,
              display: 'flex', alignItems: 'center', gap: '10px',
              background: 'var(--panel)', border: '1px solid var(--danger-border)',
              padding: '8px 14px', fontSize: '11px', color: 'var(--danger)',
              letterSpacing: '0.04em', boxShadow: '0 4px 18px rgba(0,0,0,0.18)',
            }}>
              <span>⚠ {branchError}</span>
              <button
                onClick={() => { setBranchError(''); handleBranch() }}
                style={{
                  border: '1px solid var(--danger)', background: 'none', color: 'var(--danger)',
                  padding: '3px 10px', fontSize: '10px', cursor: 'pointer',
                  letterSpacing: '0.06em', textTransform: 'uppercase',
                }}
              >
                ↻ Retry
              </button>
              <button
                onClick={() => setBranchError('')}
                title="Dismiss"
                style={{
                  border: 'none', background: 'none', color: 'var(--danger)',
                  cursor: 'pointer', fontSize: '13px',
                }}
              >
                ✕
              </button>
            </div>
          )}
        </main>

        {/* Bottom toolbar */}
        {!isFinal && (
          <div style={{
            height: '72px', display: 'flex', alignItems: 'center',
            justifyContent: 'center', gap: '12px',
            borderTop: '1px solid var(--border)', flexShrink: 0,
          }}>
            <ToolbarButton onClick={() => setSelectedIds(new Set(images.map(i => i.url)))}>
              □ SELECT ALL
            </ToolbarButton>
            <ToolbarButton onClick={() => setSelectedIds(new Set())}>
              × CLEAR
            </ToolbarButton>
            <ToolbarButton
              onClick={handleSaveToBoard}
              disabled={selectedIds.size === 0 || boardSaveState === 'saving'}
              title="Copy the selected images into your persistent board"
            >
              {boardSaveState === 'saved'
                ? '\u2713 SAVED'
                : boardSaveState === 'saving'
                ? '\u27F3 SAVING...'
                : `+ SAVE TO BOARD (${selectedIds.size})`}
            </ToolbarButton>
            <div data-tour="branch-btn">
            <ToolbarButton
              primary
              onClick={handleBranch}
              disabled={spawnableCount === 0 || branching || !atLatest || spanLive}
              title={spanLive ? "The spread is still streaming — you can already click images; breeding opens when the round settles (or hit SPREAD LOOKS GOOD)" : !atLatest ? "Redo to latest to spawn again" : (spawnableCount === 0 ? "Click one or more images first — then we'll generate new variations near them" : "Generates a new round of variations bred from your selected images")}
            >
              {spanLive
                ? `\u25CD SPREADING\u2026 PICK AS THEY LAND (${spawnableCount})`
                : branching
                ? '\u27F3 MAKING MORE...'
                : `\u2726 MAKE MORE LIKE THESE (${spawnableCount})`
              }
            </ToolbarButton>
            </div>
            {/* MVP: image-editing entry point removed — this is a
                latent-exploration tool only. The ✎ EDIT button + its
                edit-provider dropdown lived here (see *.bak-noedit). */}
          </div>
        )}
        {isFinal && (
          <div style={{
            height: '72px', display: 'flex', alignItems: 'center',
            justifyContent: 'center', gap: '12px',
            borderTop: '1px solid var(--border)', flexShrink: 0,
          }}>
            <ToolbarButton onClick={() => setSelectedIds(new Set(images.map(i => i.url)))}>
              □ SELECT ALL
            </ToolbarButton>
            <ToolbarButton onClick={() => setSelectedIds(new Set())}>
              × CLEAR
            </ToolbarButton>
            <ToolbarButton
              onClick={handleSaveToBoard}
              disabled={selectedIds.size === 0 || boardSaveState === 'saving'}
              title="Copy the selected images into your persistent board"
            >
              {boardSaveState === 'saved'
                ? '\u2713 SAVED'
                : boardSaveState === 'saving'
                ? '\u27F3 SAVING...'
                : `+ SAVE TO BOARD (${selectedIds.size})`}
            </ToolbarButton>
            {/* MVP: ✦ EDIT button + edit-provider dropdown removed here too
                (latent-exploration-only). See *.bak-noedit. */}
          </div>
        )}
      </div>

      {/* Full-size viewer — arrow keys browse, ESC closes; refined renders
          get a SKETCH/REFINED before-after toggle. */}
      {lightboxIdx != null && images.length > 0 && (
        <Lightbox
          items={images.map(img => ({
            url: img.url,
            refinedUrl: refinedUrls[img.id] || null,
            label: isSketchRound && !refinedUrls[img.id]
              ? 'Coarse sketch — pick directions, not final detail'
              : (isFinal ? 'Final image' : `Round ${interval} preview`),
          }))}
          index={Math.min(lightboxIdx, images.length - 1)}
          onNavigate={setLightboxIdx}
          onClose={() => setLightboxIdx(null)}
          onRefine={isFinal ? null : (i) => {
            const img = images[i]
            if (img) handleRefine({ stopPropagation: () => {} }, img)
          }}
          refiningIndex={(() => {
            const img = images[Math.min(lightboxIdx, images.length - 1)]
            return img && refining[img.id] ? Math.min(lightboxIdx, images.length - 1) : null
          })()}
        />
      )}

      <Tour
        steps={CLUSTER_TOUR_STEPS}
        open={tour.open}
        onClose={() => { tour.close(); onTourModeChange(false) }}
        storageKey="ltnt.tour.cluster"
      />
    </div>
  )
}

function SectionLabel({ children }) {
  return (
    <div style={{
      fontSize: '10px', letterSpacing: '0.08em',
      textTransform: 'uppercase', color: 'var(--muted)',
      padding: '0 20px', marginBottom: '8px',
    }}>
      {children}
    </div>
  )
}

function SidebarRow({ label, right }) {
  return (
    <div style={{
      display: 'flex', justifyContent: 'space-between', alignItems: 'center',
      padding: '5px 20px', fontSize: '11px', letterSpacing: '0.04em',
      textTransform: 'uppercase',
    }}>
      <span>[ ] {label}</span>
      <span style={{ color: 'var(--muted)', fontSize: '10px' }}>{right}</span>
    </div>
  )
}

function ToolbarButton({ children, primary, onClick, disabled, title }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      title={title}
      style={{
        border: '1px solid var(--text)',
        padding: '10px 20px',
        background: primary ? 'var(--text)' : 'none',
        color: primary ? 'var(--bg)' : 'var(--text)',
        display: 'flex', alignItems: 'center', gap: '6px',
        cursor: disabled ? 'not-allowed' : 'pointer',
        opacity: disabled ? 0.4 : 1,
      }}
    >
      {children}
    </button>
  )
}

// Edit-provider picker. Delete this block (and the two usages above) to
// strip the OpenAI path.
function EditProviderSelect({ value, onChange }) {
  return (
    <label style={{
      display: 'flex', alignItems: 'center', gap: '6px',
      fontSize: '10px', letterSpacing: '0.06em',
      textTransform: 'uppercase', color: 'var(--muted)',
    }}>
      <span>Edit with:</span>
      <select
        value={value}
        onChange={e => onChange(e.target.value)}
        style={{
          border: '1px solid var(--text)',
          padding: '5px 8px',
          background: 'var(--panel)',
          fontSize: '11px',
          letterSpacing: '0.04em',
          textTransform: 'uppercase',
          cursor: 'pointer',
        }}
      >
        <option value="reve">Reve</option>
        <option value="nanobanana">Nano Banana</option>
        {/* GPT Image stashed — uncomment to restore:
        <option value="openai">GPT Image</option>
        */}
      </select>
    </label>
  )
}

// ── Pixel cat progress bar (shared with GenerateView) ─────────────────────────
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

function PixelCatProgress({ progress }) {
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
        height: '2px', background: 'var(--panel-2)',
      }} />
      <div style={{
        position: 'absolute', bottom: '0', left: '0',
        width: `${progress}%`, height: '2px', background: 'var(--text)',
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
