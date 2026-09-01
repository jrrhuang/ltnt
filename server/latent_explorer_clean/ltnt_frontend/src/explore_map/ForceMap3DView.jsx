// CREATE MAP — Mode B: 3D latent-space fly-through. The owner's beloved
// "zoom through space" aesthetic: perspective fly-through + FogExp2 + UnrealBloom
// glow + a dust field. Layout is VOLUMETRIC and HIERARCHICAL:
//   • x,y  = DINO manifold coords (im.x, im.y in [0,1] -> world units)
//   • z    = -depth * DEPTH_SPACING  (roots at z=0, each generation recedes
//            into the fog) -> hierarchy becomes legible as depth planes while
//            you keep the expansive fly-through.
// Nodes are PINNED (fx,fy,fz) — no unstable force sim. Lineage edges are
// hover/select-only (always-on would be an ugly spiderweb across the volume).
//
// HANG ROOT-CAUSE (fixed here): react-force-graph-3d@1.29 mounts its own
// <canvas>; if ANYTHING in its init or in our post-mount bloom/scene setup
// throws, the canvas never appears and the view hangs on the loader forever
// with NO surfaced error. Fixes: (1) an ErrorBoundary around <ForceGraph3D>
// that surfaces the real error + a usable fallback instead of a silent hang;
// (2) every scene/bloom mutation wrapped so a bloom failure degrades to
// "renders without bloom" rather than killing the whole view; (3) the loader
// is a small non-blocking pill (pointerEvents:none) that auto-clears — it can
// never gate the canvas.
import { Component, useEffect, useMemo, useRef, useState } from 'react'
import ForceGraph3D from 'react-force-graph-3d'
import * as THREE from 'three'
// Static import resolves at build time (verified present in three@0.185). Bloom
// is optional eye-candy: the IMPORT is safe, but its runtime INSTANTIATION is
// wrapped below so an UnrealBloomPass/composer API change degrades to "no bloom"
// rather than hanging the whole view.
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js'

const DEPTH_SPACING = 300          // world units each generation recedes into z
const XY_SPREAD = 2100             // maps DINO [0,1] -> world units on x/y
const depthOpacity = (d) => Math.max(0.34, 1 - d * 0.15)
const depthSize = (d) => (d === 0 ? 140 : Math.max(30, 96 - d * 10))

// ---- Error boundary: turn a silent 3D hang into a visible, reported error ----
class GL3DBoundary extends Component {
  constructor(p) { super(p); this.state = { err: null } }
  static getDerivedStateFromError(err) { return { err } }
  componentDidCatch(err, info) {
    // Surface the EXACT error (this is what was being swallowed on the hang).
    // eslint-disable-next-line no-console
    console.error('[ForceMap3D] renderer threw:', err && err.message, err, info)
  }
  render() {
    if (this.state.err) {
      return (
        <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', flexDirection: 'column', gap: 10, background: '#000000', color: '#9fb4e8', fontSize: 12, letterSpacing: '0.08em', textAlign: 'center', padding: 24 }}>
          <div style={{ textTransform: 'uppercase', opacity: 0.9 }}>3D renderer failed to start</div>
          <div style={{ opacity: 0.6, maxWidth: 520, fontFamily: 'monospace', fontSize: 11 }}>{String(this.state.err && this.state.err.message || this.state.err)}</div>
          <div style={{ opacity: 0.5 }}>Switch to ◉ NEBULA (2D) — it works.</div>
        </div>
      )
    }
    return this.props.children
  }
}

export default function ForceMap3DView({ manifest, cinematic = false, onPick = () => {} }) {
  const wrapRef = useRef(null)
  const fgRef = useRef(null)
  const [size, setSize] = useState({ w: 0, h: 0 })
  const hoverRef = useRef(null)
  const [loaded, setLoaded] = useState(false)   // any texture decoded -> drop loader
  const framedRef = useRef(false)

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

  // Build graph with PINNED positions: x,y from DINO coords, z = -depth*spacing.
  const graphData = useMemo(() => {
    const imgs = manifest?.images || []
    let maxDepth = 0
    imgs.forEach((im) => { if ((im.depth || 0) > maxDepth) maxDepth = im.depth || 0 })
    const nodes = imgs.map((im) => {
      const wx = ((im.x ?? 0.5) - 0.5) * XY_SPREAD
      const wy = ((im.y ?? 0.5) - 0.5) * XY_SPREAD
      // Center the z-stack around 0 so the camera framing sits inside the volume.
      const wz = ((maxDepth / 2) - (im.depth || 0)) * DEPTH_SPACING
      return {
        id: im.id, depth: im.depth || 0, url: im.url, parent: im.parent,
        // PIN — react-force-graph honors fx/fy/fz and skips the force sim for them.
        fx: wx, fy: wy, fz: wz, x: wx, y: wy, z: wz,
      }
    })
    const links = imgs
      .filter((im) => im.parent !== null && im.parent !== undefined)
      .map((im) => ({ source: im.parent, target: im.id }))
    // Adjacency for hover-lineage (parent -> children, and child -> parent).
    const kids = {}, parentOf = {}
    links.forEach((l) => {
      (kids[l.source] = kids[l.source] || []).push(l.target)
      parentOf[l.target] = l.source
    })
    return { nodes, links, kids, parentOf }
  }, [manifest])

  const totalNodes = graphData.nodes.length || 1
  const texCache = useRef(new Map())

  const FADE_MS = 650   // per-sprite fog-in: fade up from 0 over this window

  // Start a sprite fading UP from its current opacity to its depth target. The
  // rAF loop below (fadeTick) advances it every frame. Sprites therefore rise
  // OUT OF THE FOG as their textures stream in, so the first ~1-2s of texture
  // streaming reads as intentional atmosphere instead of a black gate.
  const beginFade = (node) => {
    if (!node.__mat) return
    if (hoverRef.current != null) { node.__mat.opacity = depthOpacity(node.depth); return }
    node.__fadeFrom = node.__mat.opacity || 0
    node.__fadeTo = depthOpacity(node.depth)
    node.__fadeStart = performance.now()
  }

  const makeSprite = (node) => {
    let tex = texCache.current.get(node.url)
    if (!tex) {
      tex = new THREE.TextureLoader().load(node.url, () => {
        setLoaded(true)          // first decode -> clear the loader pill
        beginFade(node)          // fog-in this sprite (never snaps to visible)
        try { fgRef.current?.refresh() } catch {}
      })
      tex.colorSpace = THREE.SRGBColorSpace
      texCache.current.set(node.url, tex)
    }
    const ready = !!(tex.image && tex.image.width)
    // Always birth at 0 opacity; even an already-decoded texture (cached from a
    // prior mount / a re-refresh) fades up rather than popping in hard.
    const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false, opacity: 0 })
    node.__mat = mat
    if (ready) beginFade(node)   // texture already decoded -> fog it in now
    const sprite = new THREE.Sprite(mat)
    const s = depthSize(node.depth)
    sprite.scale.set(s, s, 1)
    // ---- Z=0 ROOT-CAUSE FIX (measured in-browser) --------------------------
    // react-force-graph-3d only copies node.x/y/z -> obj.position DURING its
    // simulation tick loop. With every node PINNED (fx/fy/fz), the sim reaches
    // alpha≈0 essentially instantly and the custom sprites are created lazily
    // AFTER that — so the tick loop that would place them has already stopped
    // and every sprite is left at its default (0,0,0). Verified live: 972/972
    // sprites at world (0,0,0) while node data held the correct fz (±440 etc),
    // and a reheat did NOT move them. Because positions are fully pinned and
    // deterministic, we place the object ourselves at creation. The cold sim
    // never fights this (confirmed: manual positions held across ticks).
    sprite.position.set(node.fx ?? node.x ?? 0, node.fy ?? node.y ?? 0, node.fz ?? node.z ?? 0)
    node.__sprite = sprite
    return sprite
  }

  // Belt-and-suspenders: after mount, force every sprite's position from its
  // pinned node coords (in case any object was created before RFG attached it,
  // or a refresh recreates objects while the sim is cold). Cheap + idempotent.
  const syncPositions = () => {
    const fg = fgRef.current
    if (!fg) return
    try {
      graphData.nodes.forEach((n) => {
        const o = n.__sprite || n.__threeObj
        if (o && o.position) o.position.set(n.fx ?? n.x ?? 0, n.fy ?? n.y ?? 0, n.fz ?? n.z ?? 0)
      })
    } catch {}
  }

  // Fog + bloom + dust (the "glowing organism" layer). Guarded so a bloom
  // failure degrades to "no bloom" instead of hanging the whole view.
  useEffect(() => {
    const fg = fgRef.current
    if (!fg) return
    let dust
    const t = setTimeout(() => {
      // Fog — cheap + the single biggest depth cue; do it first, on its own.
      try {
        const scene = fg.scene()
        // Density tuned so the near/root generation is crisp and deeper gens
        // dissolve softly into the fog — NOT so dense that the framed view blacks
        // out. Pair this with the clamped camera pullback in the framing effect.
        scene.fog = new THREE.FogExp2(0x000000, 0.0006)
      } catch (e) { /* keep going */ }
      // Bloom — optional; isolate so a composer/UnrealBloomPass API change can't
      // take down fog/dust/canvas.
      try {
        const composer = fg.postProcessingComposer && fg.postProcessingComposer()
        if (typeof UnrealBloomPass === 'function' && composer && !composer.__bloom) {
          const bloom = new UnrealBloomPass(new THREE.Vector2(size.w || 1200, size.h || 800), 1.3, 0.8, 0.12)
          composer.addPass(bloom)
          composer.__bloom = bloom
        }
      } catch (e) {
        // eslint-disable-next-line no-console
        console.warn('[ForceMap3D] bloom disabled (degraded gracefully):', e && e.message)
      }
      // Dust field — starfield vastness.
      try {
        const scene = fg.scene()
        const N = 1400
        const pos = new Float32Array(N * 3)
        for (let i = 0; i < N * 3; i++) pos[i] = (Math.random() - 0.5) * 2200
        const geo = new THREE.BufferGeometry()
        geo.setAttribute('position', new THREE.BufferAttribute(pos, 3))
        dust = new THREE.Points(geo, new THREE.PointsMaterial({ color: 0x5878ff, size: 1.7, transparent: true, opacity: 0.5, sizeAttenuation: true, depthWrite: false }))
        scene.add(dust)
      } catch (e) { /* dust is pure flavor */ }
    }, 200)
    return () => { clearTimeout(t); try { if (dust) fgRef.current?.scene().remove(dust) } catch {} }
  }, [size.w, size.h])

  // Controls + frame the volume so you START inside it (never a black void),
  // looking down the z-axis so generations recede into fog.
  useEffect(() => {
    const fg = fgRef.current
    if (!fg) return
    try {
      const controls = fg.controls()
      controls.autoRotate = false
      // Momentum / inertia: damping makes drags & zooms GLIDE to a stop instead
      // of snapping. Lower factor = longer, smoother coast. Slightly slower
      // rotate/zoom speeds make the fly-through feel weighty, not twitchy.
      controls.enableDamping = true
      controls.dampingFactor = 0.06
      controls.rotateSpeed = 0.6
      controls.zoomSpeed = 0.8
      controls.panSpeed = 0.7
    } catch {}
    // Frame the volume from the pinned data. The camera/canvas may not be wired
    // the instant this effect runs, so we RETRY until the camera actually reaches
    // the target — a single fire-once frame() was leaving the camera stuck at a
    // default pose (black screen: the volume sat off-axis / behind fog).
    const ns = graphData.nodes
    let cx = 0, cy = 0, cz = 0
    ns.forEach((n) => { cx += n.fx; cy += n.fy; cz += n.fz })
    if (ns.length) { cx /= ns.length; cy /= ns.length; cz /= ns.length }
    const dists = ns.map((n) => Math.hypot(n.fx - cx, n.fy - cy)).sort((a, b) => a - b)
    const rCore = dists[Math.floor(dists.length * 0.72)] || 300
    const zMax = ns.length ? Math.max(...ns.map((n) => n.fz)) : 0
    // Pull back to see the x/y core, but CLAMP the distance: FogExp2 (density
    // 0.0006) makes anything past ~2500 units fog out to pure black, so an
    // unclamped pullback = black screen (measured: camDist 2900 -> fog factor
    // 0.99). We sit ~800-1300 units in FRONT of the nearest (max-z) generation
    // so the roots are crisp and deeper gens recede softly into the fog.
    const dist = Math.min(1300, Math.max(800, rCore * 1.5))
    // Camera sits in front of the root plane, not behind the whole z-span.
    const camZ = zMax + dist
    const timers = []
    // Re-issue the framing across the whole mount window. react-force-graph may
    // not have its camera/canvas wired for the first few hundred ms; a single
    // fire-once frame() was silently no-op'ing and leaving the camera at a
    // default pose (the black screen). We snap the target every retry (instant,
    // duration 0) so whenever the camera finally exists, it lands correctly; the
    // very first attempt animates for a gentle ease-in if the camera is ready.
    const doFrame = (animate) => {
      if (!ns.length) return
      try {
        const ctr = fg.controls()
        if (ctr) ctr.target.set(cx, cy, cz)
        fg.cameraPosition({ x: cx, y: cy, z: camZ }, { x: cx, y: cy, z: cz }, animate ? 1400 : 0)
      } catch {}
      syncPositions()
    }
    // Stop re-framing the instant the user grabs the scene, so we never yank
    // the camera out of their hands.
    let userTook = false
    let dom
    const onUser = () => { userTook = true }
    try { dom = fg.renderer() && fg.renderer().domElement; if (dom) { dom.addEventListener('pointerdown', onUser, { passive: true }); dom.addEventListener('wheel', onUser, { passive: true }) } } catch {}
    syncPositions()
    doFrame(true)
    // Belt-and-suspenders re-frames spanning the ~0.1-1.6s mount window: whenever
    // the camera finally exists it lands correctly, killing the black screen.
    ;[150, 350, 600, 1000, 1600].forEach((ms) => timers.push(setTimeout(() => {
      if (!framedRef.current && !userTook) doFrame(false)
    }, ms)))
    timers.push(setTimeout(() => { framedRef.current = true }, 1800))
    return () => { timers.forEach(clearTimeout); try { if (dom) { dom.removeEventListener('pointerdown', onUser); dom.removeEventListener('wheel', onUser) } } catch {} }
  }, [graphData, size.w])

  // ---- Streaming mount + navigation feel (idle drift) -----------------------
  // react-force-graph's own animation loop (_animationCycle, which reschedules
  // itself via rAF EVERY frame regardless of the sim cooldown — verified in the
  // lib source) does the rendering, hover-raycasting AND controls.update(). This
  // light loop only layers on the FEEL: (a) subtle idle ambient auto-rotate so the
  // scene breathes when untouched, and (b) keeps sprite positions synced for the
  // first few seconds as textures stream in. It renders nothing itself.
  useEffect(() => {
    let raf, running = true
    const start = performance.now()
    let lastUserMove = start
    let wired = false
    let ctr, dom
    const markMove = () => { lastUserMove = performance.now() }
    // fgRef is attached imperatively AFTER this effect first runs, so poll for it.
    const wire = () => {
      const fg = fgRef.current
      if (!fg) return false
      try {
        ctr = fg.controls(); if (!ctr) return false
        dom = fg.renderer() && fg.renderer().domElement
        if (dom) {
          dom.addEventListener('pointerdown', markMove, { passive: true })
          dom.addEventListener('wheel', markMove, { passive: true })
          dom.addEventListener('pointermove', (e) => { if (e.buttons) markMove() }, { passive: true })
        }
        return true
      } catch { return false }
    }
    const loop = (now) => {
      if (!running) return
      if (!wired) wired = wire()
      if (wired) {
        const t = (now - start) / 1000
        if (t < 3.5) syncPositions()   // sprites stream in over the first seconds
        // Fog-in: advance any sprite currently fading up. Only touches nodes
        // with an active __fadeStart, so it costs ~nothing once everything has
        // faded in and never fights the hover-lineage opacity (hover clears
        // __fadeStart). Guarded so a stray node never throws in the loop.
        if (hoverRef.current == null) {
          const nds = graphData.nodes
          for (let i = 0; i < nds.length; i++) {
            const n = nds[i]
            if (n.__fadeStart == null || !n.__mat) continue
            const p = Math.min(1, (now - n.__fadeStart) / FADE_MS)
            const e = 1 - (1 - p) * (1 - p)   // easeOutQuad — settles gently
            n.__mat.opacity = (n.__fadeFrom || 0) + ((n.__fadeTo ?? 1) - (n.__fadeFrom || 0)) * e
            if (p >= 1) { n.__mat.opacity = n.__fadeTo ?? 1; n.__fadeStart = null }
          }
        }
        try {
          if (ctr) {
            // Idle "breathing" orbit after 2.2s untouched; user touch cuts it off.
            ctr.autoRotate = (now - lastUserMove) > 2200
            ctr.autoRotateSpeed = 0.16
            // No ctr.update() here — RFG's own loop calls it every frame. (Calling
            // it twice would double the damping/rotate speed.)
          }
        } catch {}
      }
      raf = requestAnimationFrame(loop)
    }
    raf = requestAnimationFrame(loop)
    return () => {
      running = false
      cancelAnimationFrame(raf)
      try { if (dom) { dom.removeEventListener('pointerdown', markMove); dom.removeEventListener('wheel', markMove) } } catch {}
    }
  }, [graphData, size.w])

  // Safety: the loader pill must never linger even if textures are slow/blocked.
  useEffect(() => {
    const t = setTimeout(() => setLoaded(true), 1500)
    return () => clearTimeout(t)
  }, [])

  // ---- Hover/select lineage highlight (edges appear ONLY for the family) ----
  const applyFamily = (id) => {
    hoverRef.current = id
    const { kids, parentOf } = graphData
    let fam = null
    if (id != null) {
      fam = new Set([id, ...(kids[id] || [])])
      if (parentOf[id] != null) fam.add(parentOf[id])
    }
    graphData.nodes.forEach((n) => {
      if (!n.__mat) return
      n.__fadeStart = null   // hover opacity wins over any in-flight fog-in fade
      n.__mat.opacity = fam == null ? depthOpacity(n.depth) : (fam.has(n.id) ? 1 : 0.06)
    })
    try { fgRef.current?.refresh() } catch {}
  }

  const isLineageOf = (l, id) => {
    if (id == null) return false
    const sId = typeof l.source === 'object' ? l.source.id : l.source
    const tId = typeof l.target === 'object' ? l.target.id : l.target
    return sId === id || tId === id
  }

  return (
    <div ref={wrapRef} style={{ position: 'absolute', inset: 0, background: '#000000' }}>
      {size.w > 0 && (
        <GL3DBoundary>
          <ForceGraph3D
            ref={fgRef}
            width={size.w}
            height={size.h}
            graphData={graphData}
            backgroundColor="#000000"
            showNavInfo={false}
            numDimensions={3}
            // Positions are fully PINNED, so run ZERO sim ticks — cheap, and it
            // takes the force sim out of the picture entirely. (react-force-graph's
            // render + hover-raycast loop, _animationCycle, is NOT gated by the sim
            // cooldown — it reschedules itself via rAF every frame regardless — so
            // rendering, hover-lineage and click->editor keep working with the sim
            // off; verified against the lib source.)
            warmupTicks={0}
            cooldownTicks={0}
            onEngineStop={syncPositions}
            nodeThreeObject={makeSprite}
            // Edges: barely-there at rest, bright for the hovered/selected lineage.
            linkColor={(l) => isLineageOf(l, hoverRef.current) ? 'rgba(200,232,255,0.95)' : 'rgba(120,170,255,0.0)'}
            linkWidth={(l) => isLineageOf(l, hoverRef.current) ? 3 : 0}
            linkDirectionalParticles={(l) => isLineageOf(l, hoverRef.current) ? 3 : 0}
            linkDirectionalParticleWidth={2.4}
            linkDirectionalParticleColor={() => 'rgba(210,235,255,1)'}
            enableNodeDrag={false}
            onBackgroundClick={() => applyFamily(null)}
            onNodeHover={(node) => applyFamily(node ? node.id : null)}
            onNodeClick={(node) => {
              // PRIMARY action: single-click OPENS the view-only lightbox — same
              // as 2D NEBULA, so the "click to open" footer hint is accurate in
              // both views. Fire onPick FIRST so the lightbox appears instantly
              // and isn't gated on the camera move.
              onPick(node)   // node has .url — opens the lightbox
              // Subtle, short camera ease toward the picked node (kept because
              // it's trivial and reads nice) — NOT the primary action.
              const fg = fgRef.current
              try {
                if (fg && node.fx !== undefined) {
                  const dist = 320
                  fg.cameraPosition(
                    { x: node.fx, y: node.fy, z: node.fz + dist },
                    { x: node.fx, y: node.fy, z: node.fz },
                    600,
                  )
                }
              } catch {}
            }}
          />
        </GL3DBoundary>
      )}
      {!loaded && (
        <div style={{ position: 'absolute', bottom: 20, left: 0, right: 0, display: 'flex', justifyContent: 'center', pointerEvents: 'none' }}>
          <style>{`@keyframes ltntPulse { 0%, 100% { opacity: 0.42 } 50% { opacity: 0.95 } }`}</style>
          <div style={{ color: '#9fb4e8', fontFamily: 'inherit', fontSize: 11, letterSpacing: '0.14em', textTransform: 'uppercase', animation: 'ltntPulse 1.6s ease-in-out infinite', background: 'rgba(5,6,12,0.6)', padding: '6px 14px', borderRadius: 20, backdropFilter: 'blur(6px)' }}>Materializing the latent space…</div>
        </div>
      )}
    </div>
  )
}
