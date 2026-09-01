// CREATE MAP — DINO-anchored latent atlas.
// Every node sits at its DINO-UMAP coordinate (screen position ≈ semantic
// similarity). A one-time relaxation pass gently de-overlaps near-identical
// images so they sit almost-adjacent instead of fully merging (Jerry's spec:
// "balance between force repulsion and DINO ... almost next to each other"),
// while a weak pull-back keeps every node near its true DINO position. All nodes
// are pinned + visible at once (dense / expansive), rendered on canvas so pan/
// zoom stay smooth at ~1000 nodes. Faint lineage edges show the tree.
import { useEffect, useMemo, useRef, useState } from 'react'
import ForceGraph2D from 'react-force-graph-2d'

const SPREAD = 1500

const nodeSize = (d) => (d === 0 ? 56 : Math.max(30, 48 - d * 4))
const depthAlpha = () => 1

export default function ForceMapView({ manifest, onPick = () => {} }) {
  const wrapRef = useRef(null)
  const fgRef = useRef(null)
  const [size, setSize] = useState({ w: 0, h: 0 })
  const [loadPct, setLoadPct] = useState(0)
  // layoutReady gates the heavy graph render. It flips true only AFTER the
  // ~40-iteration O(n²) de-overlap relaxation finishes — and that relaxation is
  // DEFERRED out of the render path (see effect below) so the first paint is
  // instant and the "PREPARING MAP…" overlay is visible from frame 0. Doing the
  // relaxation synchronously in a useMemo blocked first paint → black screen.
  const [layoutReady, setLayoutReady] = useState(false)
  const hoverRef = useRef(null)
  const [, force] = useState(0)

  useEffect(() => {
    const el = wrapRef.current
    if (!el) return
    const ro = new ResizeObserver(() => {
      const r = el.getBoundingClientRect()
      setSize({ w: Math.max(1, r.width), h: Math.max(1, r.height) })
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  // CHEAP memo: just the raw nodes/links at their DINO coords + the adjacency.
  // No relaxation here — this stays out of the critical path so mount is instant.
  const { nodes: baseNodes, links, childrenOf } = useMemo(() => {
    const imgs = manifest?.images || []
    const nodes = imgs.map((im) => {
      const dx = (im.x - 0.5) * SPREAD
      const dy = (im.y - 0.5) * SPREAD
      return { id: im.id, depth: im.depth, url: im.url, parent: im.parent, dinoX: dx, dinoY: dy, x: dx, y: dy }
    })
    const links = imgs
      .filter((im) => im.parent !== null && im.parent !== undefined)
      .map((im) => ({ source: im.parent, target: im.id }))
    const kids = {}
    links.forEach((l) => { (kids[l.source] = kids[l.source] || []).push(l.target) })
    return { nodes, links, childrenOf: kids }
  }, [manifest])

  // graphData holds the RELAXED, pinned nodes. Empty until the deferred
  // relaxation completes, so we never hand the heavy graph to ForceGraph2D
  // before it's ready.
  const [graphData, setGraphData] = useState({ nodes: [], links: [] })

  // DEFER the heavy relaxation to AFTER first paint. requestAnimationFrame lets
  // the browser paint the "PREPARING MAP…" overlay first; the nested
  // setTimeout(…,0) yields once more so the overlay is actually on-screen before
  // we hog the main thread with the O(n²) relaxation. When done, publish the
  // pinned graph and flip layoutReady → the graph renders.
  useEffect(() => {
    setLayoutReady(false)
    setGraphData({ nodes: [], links: [] })
    if (!baseNodes.length) { setLayoutReady(true); return }
    let cancelled = false
    const cleanup = { t: null }
    // Clone so the relaxation mutates its own copy (baseNodes memo stays clean).
    const nodes = baseNodes.map((nd) => ({ ...nd }))
    const raf = requestAnimationFrame(() => {
      const t = setTimeout(() => {
        if (cancelled) return
        const n = nodes.length
        const rad = nodes.map((nd) => nodeSize(nd.depth) * 0.62)
        const ITERS = n > 1400 ? 24 : 40
        for (let it = 0; it < ITERS; it++) {
          for (let i = 0; i < n; i++) {
            const a = nodes[i]
            a.x += (a.dinoX - a.x) * 0.03
            a.y += (a.dinoY - a.y) * 0.03
          }
          for (let i = 0; i < n; i++) {
            const a = nodes[i]
            for (let j = i + 1; j < n; j++) {
              const b = nodes[j]
              let dx = b.x - a.x, dy = b.y - a.y
              let dist = Math.sqrt(dx * dx + dy * dy) || 0.01
              const minD = rad[i] + rad[j]
              if (dist < minD) {
                const push = (minD - dist) / 2
                const ux = dx / dist, uy = dy / dist
                a.x -= ux * push; a.y -= uy * push
                b.x += ux * push; b.y += uy * push
              }
            }
          }
        }
        // Pin every node at its relaxed position — no live simulation.
        nodes.forEach((nd) => { nd.fx = nd.x; nd.fy = nd.y })
        if (cancelled) return
        setGraphData({ nodes, links })
        setLayoutReady(true)
      }, 0)
      cleanup.t = t
    })
    return () => { cancelled = true; cancelAnimationFrame(raf); if (cleanup.t) clearTimeout(cleanup.t) }
  }, [baseNodes, links])

  // Preload images; refresh the canvas as they arrive (no blocking gate).
  useEffect(() => {
    let live = true; let done = 0; let tick = 0
    const total = graphData.nodes.length || 1
    graphData.nodes.forEach((nd) => {
      const img = new Image()
      const fin = () => {
        if (!live) return
        done += 1; tick += 1; setLoadPct(Math.round(done / total * 100))
        if (tick % 10 === 0 || done >= total) { try { fgRef.current?.refresh() } catch { /* noop */ } }
      }
      img.onload = fin; img.onerror = fin; img.src = nd.url; nd.__img = img
    })
    return () => { live = false }
  }, [graphData])

  // Frame the whole atlas on load (static — pinned nodes never move).
  useEffect(() => {
    const fg = fgRef.current
    if (!fg) return
    const fits = [300, 1000, 2200].map((ms) =>
      setTimeout(() => { try { fg.zoomToFit(700, 55) } catch { /* noop */ } }, ms))
    return () => fits.forEach(clearTimeout)
  }, [graphData, size])

  const isChildOfHover = (id) => {
    const h = hoverRef.current
    return h != null && (childrenOf[h] || []).includes(id)
  }

  return (
    <div ref={wrapRef} style={{ position: 'absolute', inset: 0, background: '#000000' }}>
      {size.w > 0 && layoutReady && (
        <ForceGraph2D
          ref={fgRef}
          width={size.w} height={size.h}
          graphData={graphData}
          backgroundColor="#000000"
          cooldownTicks={0}
          warmupTicks={0}
          nodeRelSize={6}
          enableZoomInteraction enablePanInteraction enableNodeDrag={false}
          minZoom={0.05} maxZoom={40}
          onNodeHover={(node) => { hoverRef.current = node ? node.id : null; force((v) => v + 1); try { fgRef.current?.refresh() } catch { /* noop */ } }}
          onNodeClick={(node) => onPick(node)}
          linkColor={(l) => {
            const h = hoverRef.current
            const s = typeof l.source === 'object' ? l.source.id : l.source
            if (h != null) return s === h ? 'rgba(195,228,255,0.92)' : 'rgba(140,185,255,0.05)'
            return 'rgba(140,185,255,0.16)'
          }}
          linkWidth={(l) => { const h = hoverRef.current; const s = typeof l.source === 'object' ? l.source.id : l.source; return h != null && s === h ? 2 : 0.6 }}
          nodeCanvasObject={(node, ctx) => {
            if (!Number.isFinite(node.x) || !Number.isFinite(node.y)) return
            const s = nodeSize(node.depth)
            const img = node.__img
            const hot = isChildOfHover(node.id) || hoverRef.current === node.id
            const alpha = hoverRef.current != null && !hot ? 0.32 : depthAlpha(node.depth)
            const x0 = node.x - s / 2, y0 = node.y - s / 2
            const r = s * 0.16
            const roundRect = (pad) => {
              const rr = r + pad, xx = x0 - pad, yy = y0 - pad, ss = s + pad * 2
              ctx.beginPath(); ctx.moveTo(xx + rr, yy)
              ctx.arcTo(xx + ss, yy, xx + ss, yy + ss, rr); ctx.arcTo(xx + ss, yy + ss, xx, yy + ss, rr)
              ctx.arcTo(xx, yy + ss, xx, yy, rr); ctx.arcTo(xx, yy, xx + ss, yy, rr)
              ctx.closePath()
            }
            // Original glowing-organism aesthetic (Jerry): additive blue halo behind each tile.
            {
              const gr = s * (hot ? 1.25 : 0.98)
              const gc = hot ? '160,215,255' : node.depth === 0 ? '120,190,255' : '90,150,235'
              const grd = ctx.createRadialGradient(node.x, node.y, s * 0.25, node.x, node.y, gr)
              grd.addColorStop(0, `rgba(${gc},1)`); grd.addColorStop(1, `rgba(${gc},0)`)
              ctx.save(); ctx.globalCompositeOperation = 'lighter'
              ctx.globalAlpha = (hot ? 0.6 : 0.22) * (hoverRef.current != null && !hot ? 0.4 : 1)
              ctx.fillStyle = grd; ctx.beginPath(); ctx.arc(node.x, node.y, gr, 0, 2 * Math.PI); ctx.fill(); ctx.restore()
            }
            ctx.save(); ctx.globalAlpha = alpha
            if (img && img.complete && img.naturalWidth) {
              ctx.save(); roundRect(0); ctx.clip(); ctx.drawImage(img, x0, y0, s, s); ctx.restore()
              roundRect(0.5)
              ctx.lineWidth = hot ? 2 : Math.max(0.6, s * 0.02)
              ctx.strokeStyle = hot ? 'rgba(205,232,255,0.95)' : (node.depth === 0 ? 'rgba(175,210,255,0.75)' : 'rgba(150,190,250,0.4)')
              ctx.stroke()
            } else {
              ctx.fillStyle = node.depth === 0 ? 'rgba(150,190,255,0.6)' : 'rgba(120,160,230,0.4)'
              ctx.beginPath(); ctx.arc(node.x, node.y, s * 0.35, 0, 2 * Math.PI); ctx.fill()
            }
            ctx.restore()
          }}
          nodePointerAreaPaint={(node, color, ctx) => {
            if (!Number.isFinite(node.x) || !Number.isFinite(node.y)) return
            const s = nodeSize(node.depth); ctx.fillStyle = color; ctx.fillRect(node.x - s / 2, node.y - s / 2, s, s)
          }}
        />
      )}
      {/* Visible from the FIRST frame on entry: while the deferred relaxation
          runs (and there'd otherwise be a black-no-indicator gap), a centered
          pulsing "PREPARING MAP…" overlay assures the user something's happening.
          Clears the instant layoutReady flips → the graph paints. */}
      {!layoutReady && (
        <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', pointerEvents: 'none' }}>
          <style>{`@keyframes ltntPrepPulse { 0%, 100% { opacity: 0.42 } 50% { opacity: 0.95 } }`}</style>
          <div style={{ color: '#9fb4e8', fontFamily: 'inherit', fontSize: 12, letterSpacing: '0.16em', textTransform: 'uppercase', animation: 'ltntPrepPulse 1.6s ease-in-out infinite', background: 'rgba(5,6,12,0.6)', padding: '8px 18px', borderRadius: 20, backdropFilter: 'blur(6px)' }}>
            Preparing map…
          </div>
        </div>
      )}
      {layoutReady && loadPct < 60 && (
        <div style={{ position: 'absolute', bottom: 16, left: 16, color: 'rgba(159,180,232,0.7)', fontFamily: 'inherit', fontSize: 11, letterSpacing: '0.1em', textTransform: 'uppercase', pointerEvents: 'none' }}>
          loading images… {loadPct}%
        </div>
      )}
      <div style={{ position: 'absolute', bottom: 16, right: 16, color: 'rgba(159,180,232,0.7)', fontFamily: 'inherit', fontSize: 11, letterSpacing: '0.08em', textTransform: 'uppercase', pointerEvents: 'none' }}>
        scroll to zoom · drag to pan · click to open
      </div>
    </div>
  )
}
