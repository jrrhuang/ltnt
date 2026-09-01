import { useState, useEffect } from 'react'
import { startGenerate, pollJob, getJobStatus, getSessionState } from './api'
import GenerateView from './views/GenerateView'
import ClusterView from './views/ClusterView'
import ExploreView from './views/ExploreView'
import BoardView from './views/BoardView'
// MVP: clicking a map node opens this view-only lightbox (NOT the editor).
import Lightbox from './components/Lightbox'

import EditView from './views/EditView'
// Nano Banana (Google Gemini) editor — isolated variant.
import EditViewNanoBanana from './views/EditViewNanoBanana'
// OpenAI editor — stashed for now (option removed from the dropdown).
// Left imported so re-enabling is just a one-line dropdown change.
import EditViewOpenAI from './views/EditViewOpenAI' // eslint-disable-line no-unused-vars
// === CREATE MAP (delete import + view + handler + case to remove) ===
import CreateMapView from './explore_map/CreateMapView'
// === END CREATE MAP ===

export default function App() {
  const [view, setView] = useState('generate')
  // Which view to return to when leaving BoardView. Set whenever the board is
  // opened so 'back' lands the user where they came from (landing vs cluster).
  const [boardFrom, setBoardFrom] = useState('cluster')
  const [clusterImages, setClusterImages] = useState([])
  // Prompt the current cluster session was generated from — threaded to
  // ClusterView so 'save to board' can attach it to each pin.
  const [sessionPrompt, setSessionPrompt] = useState('')
  const [activeJobId, setActiveJobId] = useState(null)
  // === SESSION REHYDRATION === durable backend session id for the current
  // exploration. Mirrored into the URL (?session=) so BACK / refresh / shared
  // links land back on the live canvas instead of the landing page.
  const [sessionId, setSessionId] = useState(null)
  // Bumped to force-remount GenerateView so its ltnt_active_job reconnect
  // effect re-runs when we rehydrate into an in-flight (running) job.
  const [genKey, setGenKey] = useState(0)
  // True while the on-load state fetch is in flight — renders a visible
  // "RESTORING YOUR SESSION..." overlay (zero-dead-air invariant).
  const [restoring, setRestoring] = useState(false)
  // Refined-render provenance ({particleIndex: url}) recovered from the
  // session state — seeds ClusterView's refinedUrls on a restored mount so
  // REFINED badges + sketch/refined compare survive a refresh.
  const [restoredRefined, setRestoredRefined] = useState({})
  // True when the current cluster canvas was rebuilt from a saved session —
  // ClusterView uses it to explain the dimmed earlier-round images once.
  const [wasRestored, setWasRestored] = useState(false)
  // === END SESSION REHYDRATION ===
  const [interval, setInterval_] = useState(0)
  const [totalIntervals, setTotalIntervals] = useState(0)
  const [isFinal, setIsFinal] = useState(false)
  const [exploreManifest, setExploreManifest] = useState(null)
  const [editImages, setEditImages] = useState([])
  // MVP: view-only lightbox opened by clicking a map node ({ url } | null).
  const [mapLightbox, setMapLightbox] = useState(null)
  // === CREATE MAP === prompt + model the map was launched with + manifest
  // (lifted out of CreateMapView so leaving for EditView and coming back
  // doesn't re-run the precompute).
  const [createMapPrompt, setCreateMapPrompt] = useState('')
  const [createMapModel, setCreateMapModel] = useState('flux')
  const [createMapAugment, setCreateMapAugment] = useState(true)
  const [createMapManifest, setCreateMapManifest] = useState(null)
  // === SESSION REHYDRATION ===
  // Keep ?session=<id> in the URL + remember the last session id. Passing
  // null clears the param (intentional reset). Also strips stale ?map/?view
  // params so a refresh reattaches to the live session, not an old map.
  function writeSessionUrl(sid, { push = false } = {}) {
    try {
      const u = new URL(window.location.href)
      u.searchParams.delete('map')
      u.searchParams.delete('view')
      if (sid) u.searchParams.set('session', sid)
      else u.searchParams.delete('session')
      // push=true keeps the PREVIOUS url (e.g. ?session=<old>) as a history
      // entry, so the browser BACK button can return to an abandoned session.
      if (push) window.history.pushState({}, '', u)
      else window.history.replaceState({}, '', u)
    } catch {}
    try {
      if (sid) localStorage.setItem('ltnt_last_session', sid)
      else localStorage.removeItem('ltnt_last_session')
    } catch {}
  }

  // Rolling RECENT list: last few session ids, newest first, deduped.
  // GenerateView reads this to offer one-click return to an old session
  // (thumbnails/prompts come from GET /api/sessions/{id}/state — no new
  // backend endpoint needed).
  function recordRecentSession(sid) {
    if (!sid) return
    try {
      const cur = JSON.parse(localStorage.getItem('ltnt_recent_sessions') || '[]')
      const next = [sid, ...cur.filter(s => s !== sid)].slice(0, 8)
      localStorage.setItem('ltnt_recent_sessions', JSON.stringify(next))
    } catch {}
  }
  useEffect(() => { recordRecentSession(sessionId) }, [sessionId])

  // BACK/FORWARD across pushState boundaries (e.g. leaving a session via NEW
  // PROMPT) must actually restore what the URL says — the rehydration effect
  // only runs on mount, so reload on popstate.
  useEffect(() => {
    const onPop = () => { try { window.location.reload() } catch {} }
    window.addEventListener('popstate', onPop)
    return () => window.removeEventListener('popstate', onPop)
  }, [])

  // Whenever a job becomes active, resolve its durable session id and mirror
  // it into the URL (jobs carry session_id from the very first poll).
  useEffect(() => {
    if (!activeJobId) return
    let ok = true
    getJobStatus(activeJobId)
      .then(d => {
        if (ok && d.session_id) {
          setSessionId(d.session_id)
          writeSessionUrl(d.session_id)
        }
      })
      .catch(() => {})
    return () => { ok = false }
  }, [activeJobId])

  // On load: rehydrate from ?session=<id> (or, as a fallback, the last-session
  // id in localStorage if that session is still live). Rebuilds the canvas
  // from GET /api/sessions/{id}/state instead of dumping the user on landing;
  // in-flight jobs resume polling via GenerateView's reconnect effect.
  useEffect(() => {
    const qs = new URLSearchParams(window.location.search)
    const hasMap = !!qs.get('map')
    const urlSid = qs.get('session')
    // DETERMINISTIC bare "/" (critic-5 #5): a plain visit ALWAYS shows the
    // start screen — restore happens ONLY via an explicit ?session=<id> URL
    // (the RECENT list on the start screen is the one-click way back). This
    // removes the old localStorage fallback that restored-or-not depending on
    // the last session's server state.
    const sid = urlSid
    if (!sid) return
    setRestoring(true)
    getSessionState(sid).then(st => {
      const inFlight = st.status === 'pending' || st.status === 'running'
      setSessionId(st.session_id || sid)
      if (!hasMap) writeSessionUrl(st.session_id || sid)
      if (Array.isArray(st.frozen) && st.frozen.length) setClusterFrozen(st.frozen)
      if (typeof st.prompt === 'string' && st.prompt) setSessionPrompt(st.prompt)
      // Board session-scoping key (ClusterView also re-derives this from
      // image URLs on mount; setting it here covers every restore path).
      try { localStorage.setItem('ltnt.current_session', st.session_id || sid) } catch {}
      if (inFlight && st.job_id) {
        // Reattach to the running job: GenerateView's mount effect polls
        // ltnt_active_job and hands off to the cluster view when ready.
        // Mark restored so ClusterView shows a RECONNECTING strip from the
        // first frame after the handoff (critic-5 #1 zero-dead-air).
        try { localStorage.setItem('ltnt_active_job', st.job_id) } catch {}
        setWasRestored(true)
        setGenKey(k => k + 1)
        setRestoring(false)
        return
      }
      if (st.refined && typeof st.refined === 'object' && Object.keys(st.refined).length) {
        setRestoredRefined(st.refined)
      }
      // Only waiting_selection is interactive; done/stale/cancelled/error
      // restore as a viewable final canvas (their worker is gone).
      const interactive = st.status === 'waiting_selection'
      let imgs = interactive
        ? (st.images || [])
        : ((st.result && st.result.images) ? st.result.images : (st.images || []))
      if (!imgs.length && Array.isArray(st.rounds) && st.rounds.length) {
        // Last resort: rebuild the latest round from the per-round URL list
        // (grid layout — no coords survive in this path).
        const urls = st.rounds[st.rounds.length - 1].image_urls || []
        const cols = Math.max(1, Math.ceil(Math.sqrt(urls.length)))
        const rows = Math.max(1, Math.ceil(urls.length / cols))
        imgs = urls.map((u, i) => ({
          id: i, url: u,
          x: 0.05 + 0.90 * (((i % cols) + 0.5) / cols),
          y: 0.05 + 0.90 * ((Math.floor(i / cols) + 0.5) / rows),
        }))
      }
      // Server-rehydrated (post-restart) sessions arrive with frozen:[] even
      // though every earlier round's URLs survive on disk in st.rounds —
      // the rehydrator never rebuilds history (server.py:2037), so a restart
      // silently vanished the dimmed canvas history. Derive it client-side:
      // all pre-latest round URLs, minus the active set, on a loose grid
      // (ClusterView's compact+deOverlap pass arranges them on mount).
      if (!(Array.isArray(st.frozen) && st.frozen.length)
          && Array.isArray(st.rounds) && st.rounds.length > 1) {
        const activeUrls = new Set(imgs.map(im => im.url))
        const seen = new Set()
        const hist = []
        st.rounds.slice(0, -1).forEach(r => (r.image_urls || []).forEach(u => {
          if (!activeUrls.has(u) && !seen.has(u)) { seen.add(u); hist.push(u) }
        }))
        if (hist.length) {
          const hc = Math.max(1, Math.ceil(Math.sqrt(hist.length)))
          setClusterFrozen(hist.map((u, i) => ({
            id: `hist-${i}`, url: u, frozen: true,
            x: 0.06 + 0.88 * (((i % hc) + 0.5) / hc),
            y: 0.06 + 0.88 * ((Math.floor(i / hc) + 0.5) / Math.ceil(hist.length / hc)),
          })))
        }
      }
      setRestoring(false)
      if (!imgs.length) return
      setClusterImages(imgs)
      setActiveJobId(st.job_id)
      setInterval_(st.interval || 0)
      setTotalIntervals(st.total_intervals || 0)
      setIsFinal(!interactive)
      setWasRestored(true)
      if (!hasMap) setView('cluster')
    }).catch(() => { setRestoring(false) /* unknown session — stay on landing */ })
  }, [])
  // === END SESSION REHYDRATION ===

  // Instant-load a precomputed tree by URL: ?map=<id> (e.g. ?map=a1057d8c).
  // Fetches the on-disk manifest via the /images static mount (no backend/GPU).
  useEffect(() => {
    const mid = new URLSearchParams(window.location.search).get('map')
    if (!mid) return
    fetch('/images/' + mid + '/manifest.json')
      .then(function (r) { return r.json() })
      .then(function (m) {
        (m.images || []).forEach(function (im) { im.url = '/images/' + mid + '/' + im.path })
        setCreateMapManifest(m)
        setView('create_map')
      })
      .catch(function (e) { console.error('map load failed', e) })
  }, [])
  // === END CREATE MAP ===
  // Where the user came from when entering EditView (so the back button
  // returns to that view, not always 'cluster').
  const [editFrom, setEditFrom] = useState('cluster')
  // Frozen (faded history) images, lifted out of ClusterView so they
  // survive the round-trip through EditView. Reset only on NEW PROMPT.
  const [clusterFrozen, setClusterFrozen] = useState([])
  // Undo/redo history + pointer for the ClusterView, also lifted to App so
  // the stack is preserved when the user dips into EditView and returns.
  const [clusterHistory, setClusterHistory] = useState([])
  const [clusterHistoryPtr, setClusterHistoryPtr] = useState(-1)
  // Which edit backend to route to — persisted across sessions. Accepts
  // 'reve' | 'nanobanana'. Anything else (e.g. a stale 'openai' from a
  // previous session) falls back to 'reve'.
  const [editProvider, setEditProvider] = useState(() => {
    try {
      const v = localStorage.getItem('ltnt.edit_provider')
      if (v === 'reve' || v === 'nanobanana') return v
      return 'reve'
    } catch { return 'reve' }
  })
  function changeEditProvider(p) {
    setEditProvider(p)
    try { localStorage.setItem('ltnt.edit_provider', p) } catch {}
  }

  // Tour Mode — when ON, every view auto-opens its guided tour on mount so
  // first-time visitors always see the onboarding hints. Persists across
  // sessions. Toggled via the "? TOUR" button in each view's top bar.
  const [tourMode, setTourMode] = useState(() => {
    try {
      const v = localStorage.getItem('ltnt.tour_mode')
      return v == null ? true : v === '1'  // default ON for demos
    } catch { return true }
  })
  function changeTourMode(on) {
    setTourMode(on)
    try { localStorage.setItem('ltnt.tour_mode', on ? '1' : '0') } catch {}
  }

  // THEME — the regular-usage session UI is light/dark themeable and DEFAULTS
  // TO DARK to match the flagship create-map aesthetic (no brand whiplash).
  // Persisted in localStorage; applied to <html data-theme> so the semantic CSS
  // vars in index.css re-skin the whole session flow on a single attribute swap.
  const [theme, setTheme] = useState(() => {
    try {
      const v = localStorage.getItem('ltnt.theme')
      return v === 'light' || v === 'dark' ? v : 'dark'  // default DARK
    } catch { return 'dark' }
  })
  useEffect(() => {
    try { document.documentElement.setAttribute('data-theme', theme) } catch {}
    try { localStorage.setItem('ltnt.theme', theme) } catch {}
  }, [theme])
  function toggleTheme() {
    setTheme(t => (t === 'dark' ? 'light' : 'dark'))
  }

  function handleImages(images, jobId, intervalNum, totalInts, final, prompt) {
    if (typeof prompt === 'string') setSessionPrompt(prompt)
    setClusterImages(images)
    setActiveJobId(jobId)
    setInterval_(intervalNum)
    setTotalIntervals(totalInts)
    setIsFinal(final)
    setView('cluster')
  }

  function handleExplore(manifest) {
    setExploreManifest(manifest)
    setView('explore')
  }

  // EXPLORE FROM THIS PIN — start a NEW exploration SEEDED from a saved board
  // image (image-seeded SDEdit). Reuses the exact generate-start machinery
  // GenerateView uses (startGenerate -> pollJob -> onImages -> cluster view).
  // The defaults mirror GenerateView's initial state so a pin-seeded session
  // behaves like a normal Generate, just starting from the chosen image.
  // `onProgress(stage, pct)` lets the caller (BoardView) show a launching state.
  async function handleExploreFromPin({ prompt, init_image_url, init_strength }, onProgress = () => {}) {
    const params = {
      model: 'flux',
      n_images: 6,
      num_steps: 28,
      num_clones: 3,
      num_resampling_steps: 3,
      guidance_scale: 3.5,
      rho: 0.4,
      spawn_mode: 'glass',
      schedule_mode: 'aggressive',
      tree_mode: false,
      seed: null,
      height: 512,
      width: 512,
      clone_mode: 'glass',
      init_image_url,
      init_strength: init_strength == null ? 0.55 : init_strength,
    }
    const jobId = await startGenerate(prompt || '', params)
    localStorage.setItem('ltnt_active_job', jobId)
    const data = await pollJob(jobId, (status, d) => {
      if (d?.stage || d?.progress != null) onProgress(d?.stage || '', d?.progress ?? 0)
    })
    localStorage.removeItem('ltnt_active_job')
    const isFinal = data.status === 'done'
    const images = isFinal ? data.result.images : data.images
    handleImages(images, jobId, data.interval, data.total_intervals, isFinal, prompt || '')
  }

  // === CREATE MAP ===
  function handleCreateMap(prompt, model = 'flux', augment = true) {
    setCreateMapAugment(augment)
    // Reset manifest only when launching with a different prompt or model.
    if (prompt !== createMapPrompt || model !== createMapModel)
      setCreateMapManifest(null)
    setCreateMapPrompt(prompt)
    setCreateMapModel(model)
    setView('create_map')
  }
  // MVP (latent-exploration only): clicking a map node opens a simple
  // view-only lightbox (big image + close) — NOT the editor. The old
  // setView('edit') path is intentionally gone.
  function handleCreateMapPick(img) {
    if (img && img.url) setMapLightbox({ url: img.url, label: img.label || '' })
  }
  // === END CREATE MAP ===

  function handleNext(selectedImages) {
    // Ensure images have both 'url' and 'src' for EditView compatibility
    const mapped = selectedImages.map(img => ({ ...img, src: img.src || img.url }))
    setEditImages(mapped)
    setEditFrom('cluster')
    setView('edit')
  }

  // === SESSION REHYDRATION === BACK from the map returns to the LIVE session
  // (canvas + rounds intact) instead of resetting to the landing page. Strips
  // the stale ?map/?view params and restores ?session=. Only falls back to a
  // full reset when there is no session to return to.
  function handleMapBack() {
    if (clusterImages.length > 0) {
      writeSessionUrl(sessionId)
      setView('cluster')
      return
    }
    if (sessionId) {
      writeSessionUrl(sessionId)
      setView('generate')
      return
    }
    handleReset()
  }
  // === END SESSION REHYDRATION ===

  function handleReset() {
    // Abandon any lingering active-job pointer so GenerateView doesn't try to
    // reconnect to it on mount and bounce the user back.
    try { localStorage.removeItem('ltnt_active_job') } catch {}
    // Keep the abandoned session reachable: remember it under RECENT and
    // leave its ?session= URL in browser history (pushState) so BACK returns
    // to it (critic blocker #3 — NEW PROMPT must not destroy a session).
    recordRecentSession(sessionId)
    setWasRestored(false)
    // Intentional reset: drop the session URL param + last-session pointer.
    setSessionId(null)
    writeSessionUrl(null, { push: !!sessionId })
    setRestoredRefined({})
    setView('generate')
    setClusterImages([])
    setActiveJobId(null)
    setIsFinal(false)
    setExploreManifest(null)
    setEditImages([])
    setClusterFrozen([])
    setClusterHistory([])
    setClusterHistoryPtr(-1)
    setCreateMapManifest(null)
  }

  return (
    <>
      {/* === SESSION REHYDRATION === zero-dead-air: visible state while the
          on-load session fetch is in flight (never a silent landing page). */}
      {restoring && (
        <div style={{
          position: 'fixed', inset: 0, zIndex: 1000, background: 'var(--bg)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}>
          <div style={{ textAlign: 'center' }}>
            <div style={{
              fontSize: '11px', letterSpacing: '0.1em',
              textTransform: 'uppercase', color: 'var(--muted)',
            }}>
              [ RESTORING YOUR SESSION... ]
            </div>
          </div>
        </div>
      )}
      {/* === END SESSION REHYDRATION === */}
      {view === 'generate' && (
        <GenerateView
          key={`gen-${genKey}`}
          onImages={handleImages}
          onExplore={handleExplore}
          onCreateMap={handleCreateMap}
          onBoard={() => { setBoardFrom('generate'); setView('board') }}
          tourMode={tourMode}
          onTourModeChange={changeTourMode}
        />
      )}
      {view === 'cluster' && (
        <ClusterView
          images={clusterImages}
          jobId={activeJobId}
          interval={interval}
          totalIntervals={totalIntervals}
          isFinal={isFinal}
          onImages={handleImages}
          onBack={handleReset}
          sessionPrompt={sessionPrompt}
          onBoard={() => { setBoardFrom('cluster'); setView('board') }}
          editProvider={editProvider}
          onEditProviderChange={changeEditProvider}
          initialRefinedUrls={restoredRefined}
          restored={wasRestored}
          initialFrozenImages={clusterFrozen}
          onFrozenImagesChange={setClusterFrozen}
          initialHistory={clusterHistory}
          initialHistoryPtr={clusterHistoryPtr}
          onHistoryChange={(h, p) => { setClusterHistory(h); setClusterHistoryPtr(p) }}
          tourMode={tourMode}
          onTourModeChange={changeTourMode}
          theme={theme}
          onToggleTheme={toggleTheme}
        />
      )}
      {view === 'explore' && (
        <ExploreView
          manifest={exploreManifest}
          onBack={handleReset}
        />
      )}
      {view === 'board' && (
        <BoardView
          onBack={() => setView(boardFrom)}
          onExploreFromPin={handleExploreFromPin}
        />
      )}
      {view === 'edit' && editProvider === 'nanobanana' && (
        <EditViewNanoBanana
          images={editImages}
          onBack={() => setView(editFrom)}
          tourMode={tourMode}
          onTourModeChange={changeTourMode}
        />
      )}
      {/* === CREATE MAP view (delete block to remove) === */}
      {view === 'create_map' && (
        <CreateMapView
          prompt={createMapPrompt}
          params={{ model: createMapModel, augment: createMapAugment, num_resampling_steps: 3 }}
          initialManifest={createMapManifest}
          onManifestReady={setCreateMapManifest}
          onBack={handleMapBack}
          onPick={handleCreateMapPick}
        />
      )}
      {/* === END CREATE MAP === */}
      {/* MVP: view-only lightbox for a clicked map node (no edit tools). */}
      {mapLightbox && (
        <Lightbox
          items={[{ url: mapLightbox.url, label: mapLightbox.label }]}
          index={0}
          onClose={() => setMapLightbox(null)}
        />
      )}
      {view === 'edit' && editProvider === 'reve' && (
        <EditView
          images={editImages}
          onBack={() => setView(editFrom)}
          tourMode={tourMode}
          onTourModeChange={changeTourMode}
        />
      )}
    </>
  )
}
