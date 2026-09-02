import { useState, useEffect } from 'react'
import { startGenerate, pollJob, getServerStatus, getSessionState } from '../api'
import Tour, { useTour, TourToggle } from '../Tour'

const GENERATE_TOUR_STEPS = [
  {
    title: 'Welcome to LTNT!',
    body: "LTNT helps you find your perfect image. Start with a rough idea, pick the ones you like best, and we'll keep generating variations that build on your taste — until you land on exactly what you're looking for.",
    placement: 'center',
  },
  {
    target: '[data-tour="prompt-input"]',
    title: 'Describe your image',
    body: "Type a few words about what you'd like to see. It can be as simple as \"a cozy cabin in the snow\" or as detailed as you want.",
    placement: 'top',
  },
  {
    target: '[data-tour="settings-toggle"]',
    title: 'Settings (optional)',
    body: "Want more images per round, or more rounds of refinement? You can change those here. The defaults work well — feel free to skip this.",
    placement: 'top',
  },
  {
    target: '[data-tour="generate-btn"]',
    title: 'Start generating',
    body: "Click here and watch the whole process live: first the AI's different readings of your prompt appear, then every image sharpens out of noise on screen, completing one at a time. Set SPREAD to choose how many (type any number 2-18) and PREVIEW to trade speed vs detail.",
    placement: 'top',
  },
  // Explore Map tour step stashed with the button it points at.
  // Uncomment along with the EXPLORE MAP button below to restore.
  // {
  //   target: '[data-tour="explore-btn"]',
  //   title: 'Or explore freely',
  //   body: "Not sure what you want yet? Try this instead — it builds a zoomable map of images you can wander through until something catches your eye.",
  //   placement: 'top',
  // },
]

export default function GenerateView({ onImages, onExplore, onCreateMap, onBoard = () => {}, tourMode = false, onTourModeChange = () => {} }) {
  // Prompt persistence is per-TAB (sessionStorage): a refresh mid-typing keeps
  // your text, but a fresh visit NEVER starts with someone-else's/old text in
  // the box (the "mystery text splices with my typing" class of bug). Examples
  // live in the placeholder only — the input value is always exactly what the
  // user typed.
  const [prompt, setPrompt] = useState(() => {
    try { return sessionStorage.getItem('ltnt_prompt') || '' } catch { return '' }
  })
  useEffect(() => {
    try { sessionStorage.setItem('ltnt_prompt', prompt) } catch {}
  }, [prompt])
  const [loading, setLoading] = useState(false)
  const [statusText, setStatusText] = useState('')
  const [progress, setProgress] = useState(0)
  // Live solve previews streamed by the backend ({urls, seq, step}).
  const [stepPreviews, setStepPreviews] = useState(null)
  // The LLM's prompt readings, shown as they land during generation.
  const [genReadings, setGenReadings] = useState(null)
  const [error, setError] = useState('')
  const [showSettings, setShowSettings] = useState(false)
  // Advanced generation knobs (steps/CFG/rho/…) hidden by default — plain
  // labels, engineer dials only on request.
  const [showAdvanced, setShowAdvanced] = useState(false)
  const tour = useTour('ltnt.tour.generate', { autoOpen: tourMode })

  // RECENT sessions — one-click return to a previous exploration. Ids come
  // from localStorage (App records every session); prompt + thumbnail come
  // from GET /api/sessions/{id}/state. Unobtrusive: small list, start screen
  // only, silently skips sessions the server no longer knows.
  const [recent, setRecent] = useState([])
  useEffect(() => {
    let on = true
    let ids = []
    try { ids = JSON.parse(localStorage.getItem('ltnt_recent_sessions') || '[]') } catch {}
    ids = ids.slice(0, 5)
    if (!ids.length) return
    Promise.all(ids.map(id =>
      getSessionState(id).then(st => {
        const rounds = Array.isArray(st.rounds) ? st.rounds : []
        const lastUrls = rounds.length ? (rounds[rounds.length - 1].image_urls || []) : []
        const thumb = (st.images && st.images[0] && st.images[0].url) || lastUrls[0] || null
        return { id, prompt: st.prompt || '', thumb }
      }).catch(() => null)
    )).then(items => { if (on) setRecent(items.filter(Boolean)) })
    return () => { on = false }
  }, [])

  // Generation parameters
  const [model, setModel] = useState('fluxfm')  // 'flux' | 'sd35' | 'sana' | 'fluxfm' | 'krea2'
  // Only relevant when model === 'fluxfm'. 'glass' = exact GLASS bridge
  // (M_inner NFE per clone). 'flow_map' = Weighted-Diamond-Maps-style
  // 1-NFE amortized clone (biased but ~9× faster).
  const [cloneMode, setCloneMode] = useState('flow_map')
  // Sequential round-1 makes this cheap: first image pops just as early —
  // a bigger spread only extends the arrivals, never the wait for the first.
  const [nImages, setNImages] = useState(12)
  // PREVIEW depth = Euler lookahead steps for each round's preview tiles.
  // roughest previews; STUDY = today's balance; RENDER = slowest, richest.
  // Sent as lookahead_steps (+ a richer round-1 value) on every generate.
  const PREVIEW_MODES = {
    SKETCH: { la: 2, laFirst: 4 },
    STUDY: { la: 4, laFirst: 10 },
    RENDER: { la: 8, laFirst: 12 },
  }
  const [previewMode, setPreviewMode] = useState('STUDY')
  const [numSteps, setNumSteps] = useState(28)
  const [numClones, setNumClones] = useState(3)
  const [numResamplingSteps, setNumResamplingSteps] = useState(3)
  const [guidanceScale, setGuidanceScale] = useState(3.5)
  const [rho, setRho] = useState(0.8)
  // Diversity dial — a simple artist-facing control that maps to the backend
  // spawn_mode + rho. Default 'balanced' preserves current behavior (glass, 0.4).
  // 'wild' = sdedit + lower rho (more divergent), 'tight' = glass + higher rho.
  // multiple different diversity toggles. Just do max diversity").
  // WDM hop, rho 0.95, 4-step descent — the measured-good config.
  const DIVERSITY_PRESETS = [
    { key: 'max', label: 'MAX', spawn_mode: 'flow_map', rho: 0.95 },
  ]
  const [diversityIdx, setDiversityIdx] = useState(0)
  const [augmentOn, setAugmentOn] = useState(true)  // LLM prompt-interpretation diversity
  const diversity = DIVERSITY_PRESETS[diversityIdx]
  // Compact model picker surfaced on the GENERATE bar. Shares the same `model`
  // state the generate request reads (and the Settings panel writes), so the
  // two controls stay in sync — no duplicate state. The bar cycles through the
  // three headline backends; Settings still exposes the full set (sd35/fluxfm).
  const BAR_MODELS = [
    { key: 'flux',  label: 'FLUX'   },
    { key: 'fluxfm', label: 'FLUX-FM' },
    { key: 'sd35',  label: 'SD3.5'  },
    { key: 'krea2', label: 'KREA-2' },
  ]
  // Live availability from the backend so the picker never offers a broken
  // model. /api/models is HONEST per-instance (resident model first, then
  // actually-loadable, then unavailable) — we DEFAULT to its first available
  // model so a SANA-only (or FLUX-only) instance never boots pointed at a
  // model it cannot load.
  const [availModels, setAvailModels] = useState(null)  // null=unknown, else Set of keys
  useEffect(() => {
    let ok = true
    fetch('/api/models').then(r => r.json()).then(d => {
      if (!ok) return
      const models = d.models || []
      const avail = new Set(models.filter(m => m.available).map(m => m.key))
      setAvailModels(avail)
      // One-time honest default: if the current pick can't load HERE, switch
      // to the backend's declared default (resident/loadable first).
      setModel(cur => (avail.has(cur) ? cur : (d.default || cur)))
    }).catch(() => {})
    return () => { ok = false }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])
  const isAvail = (k) => availModels === null || availModels.has(k)
  // Friendly label for whatever model is currently selected (covers models that
  // only live in Settings, e.g. SD3.5 / FluxFM, so the bar never shows a blank).
  const MODEL_LABELS = { flux: 'FLUX', sana: 'SANA', krea2: 'KREA-2', sd35: 'SD3.5', fluxfm: 'FLUXFM' }
  function cycleModel() {
    const avail = BAR_MODELS.filter(m => isAvail(m.key))
    const pool = avail.length ? avail : BAR_MODELS
    const i = pool.findIndex(m => m.key === model)
    const next = i < 0 ? pool[0] : pool[(i + 1) % pool.length]
    setModel(next.key)
  }
  const [scheduleMode, setScheduleMode] = useState('aggressive')  // '' = None, 'linear', 'aggressive'
  const [treeMode, setTreeMode] = useState(false)
  const [seed, setSeed] = useState('')
  const [height, setHeight] = useState(512)
  const [width, setWidth] = useState(512)

  // ── Boot gate — REAL progress while the backend loads models ─────────────
  // Polls /api/status until ready. While booting: named phase + honest bar
  // (fraction measured server-side: GPU bytes loaded / expected). If the
  // server itself is unreachable (job restarting), show a reconnecting state
  // and keep polling — the screen is never dead.
  const [boot, setBoot] = useState({ checked: false, ready: false, phase: '', fraction: 0, detail: '', unreachable: false })
  useEffect(() => {
    let stopped = false
    async function poll() {
      while (!stopped) {
        try {
          const d = await getServerStatus()
          if (stopped) return
          setBoot({
            checked: true, ready: !!d.ready, phase: d.phase || '',
            fraction: d.fraction || 0, detail: d.detail || '', unreachable: false,
          })
          if (d.ready) return
        } catch {
          if (stopped) return
          setBoot(b => ({ ...b, checked: true, ready: false, unreachable: true }))
        }
        await new Promise(r => setTimeout(r, 1500))
      }
    }
    poll()
    return () => { stopped = true }
  }, [])
  const bootReady = !boot.checked || boot.ready  // fail-open until first check lands

  // On mount, reconnect to any in-progress job
  useEffect(() => {
    const savedJobId = localStorage.getItem('ltnt_active_job')
    if (!savedJobId) return
    setLoading(true)
    setStatusText('Reconnecting to generation in progress...')
    let reconnectHandedOff = false  // SPAN MODE: hand off on the first streamed batch
    pollJob(savedJobId, (status, data) => {
      if (!reconnectHandedOff && data?.span?.active && (data.images?.length > 0)) {
        reconnectHandedOff = true
        setLoading(false)
        onImages(data.images, savedJobId, data.interval || 1, data.total_intervals, false, data.prompt || prompt || '')
        return
      }
      if (reconnectHandedOff) return
      if (data?.stage) setStatusText(data.stage)
      if (data?.progress != null) setProgress(data.progress)
      // The reconnect poll shows the live solve strip too — found missing by
      // driving a real mid-solve reload (only handleGenerate had the capture).
      if (data?.step_previews?.urls?.length) setStepPreviews(data.step_previews)
    }).then(data => {
      if (reconnectHandedOff) return  // ClusterView owns the job
      localStorage.removeItem('ltnt_active_job')
      setLoading(false)
      const isFinal = data.status === 'done'
      const images = isFinal ? data.result.images : data.images
      onImages(images, savedJobId, data.interval, data.total_intervals, isFinal, data.prompt || prompt || '')
    }).catch(() => {
      // Stale job from previous server session — silently clear it
      localStorage.removeItem('ltnt_active_job')
      setStatusText('')
      setLoading(false)
    })
  }, [])

  // Escape hatch: clear a stuck/abandoned generation so the prompt is never
  // permanently bricked (a wedged backend job otherwise locks the UI forever,
  // even across reloads). Backend self-releases via its selection timeout.
  function cancelGeneration() {
    try { localStorage.removeItem('ltnt_active_job') } catch {}
    setLoading(false)
    setStatusText('')
    setProgress(0)
  }

  async function handleGenerate() {
    if (!prompt.trim() || loading) return
    if (!isAvail(model)) {
      setError(`${MODEL_LABELS[model] || model.toUpperCase()} is still downloading \u2014 pick FLUX or SD3.5.`)
      return
    }
    setLoading(true)
    setError('')
    setProgress(0)
    setStatusText('Submitting...')

    const params = {
      model,
      n_images: nImages,
      // Readings count: ONLY override the server default (3) when the spread
      // is big enough to need it. Sending nImages unconditionally silently
      // changed Jerry's generation config on 2026-08-26 (3 readings x 2
      // seeds -> 6 readings x 1) — the mechanistic suspect for his
      // "looks worse" sessions. At spread<=6 the 18:26 known-good config
      // is restored exactly; at spread>6 we send nImages to avoid
      // reading-wraparound.
      ...(nImages > 6 ? { num_variations: nImages } : {}),
      lookahead_steps: PREVIEW_MODES[previewMode].la,
      lookahead_steps_first: PREVIEW_MODES[previewMode].laFirst,
      // mirrors execution). Equality-oracle verified: MAE 0.00 vs lockstep
      // on identical seeds; first breed preview ~27s vs ~46s. Krea-2 only —
      // FLUX ignores this server-side (its batching is real parallelism).
      order: 'sequential',
      num_steps: numSteps,
      num_clones: numClones,
      num_resampling_steps: numResamplingSteps,
      guidance_scale: guidanceScale,
      // Diversity dial drives rho + spawn_mode. Default 'balanced' (glass, 0.4)
      // matches prior behavior. Advanced settings-panel `rho` is overridden by
      // the preset so the two controls never conflict.
      schedule_mode: scheduleMode || null,
      tree_mode: treeMode,
      seed: seed === '' ? null : Number(seed),
      height,
      width,
      // Only honored when model === 'fluxfm'. 'glass' = exact bridge,
      // 'flow_map' = WDM-amortized 1-NFE clone jump.
      // The DIVERSITY dial is authoritative for the spawn kernel + rho
      // but never sent, so the dial did nothing to the request).
      clone_mode: diversity.spawn_mode || cloneMode,
      rho: diversity.rho,
      augment: augmentOn,
      // SPAN MODE round 1 (seamless first run): stream independent fast full
      // images batch-by-batch, novelty meter + auto-stop, then pick parents.
      // The server falls back to the classic flow on samplers that don't
      // support it (currently SANA only), so this is safe to always request.
      span_round1: true,
    }

    try {
      const jobId = await startGenerate(prompt.trim(), params)
      localStorage.setItem('ltnt_active_job', jobId)
      setStepPreviews(null)   // don't flash the previous run's strip
      setGenReadings(null)
      setStatusText('Starting your session \u2014 the first images land in under a minute...')

      // SPAN MODE: the moment the first streamed batch lands, hand the canvas
      // over to ClusterView (it keeps polling and folds later batches in) —
      // zero dead air. handedOff guards the normal resolution below from
      // double-firing when the span round later reaches waiting_selection.
      let handedOff = false
      const data = await pollJob(jobId, (status, d) => {
        if (!handedOff && d?.span?.active && (d.images?.length > 0)) {
          handedOff = true
          setLoading(false)
          onImages(d.images, jobId, d.interval || 1, d.total_intervals, false, prompt.trim())
          return
        }
        if (handedOff) return
        if (d?.stage) setStatusText(d.stage)
        if (d?.progress != null) setProgress(d.progress)
        // LIVE SOLVE PREVIEWS: while the solve runs (the previously-blank
        // 50s of round 1) the backend streams each particle's current
        // denoised estimate. Render them as a resolving strip under the
        // progress bar; "seq" busts the browser cache on the fixed URLs.
        if (d?.step_previews?.urls?.length) setStepPreviews(d.step_previews)
        // The LLM's readings appear the moment they exist — fills the
        // augment window (the last text-only stretch) with a real visual
        // beat instead of hiding the readings behind a click.
        if (d?.prompt_variations?.length) setGenReadings(d.prompt_variations)
      })
      if (handedOff) return  // ClusterView owns the job (incl. the saved job id)

      setLoading(false)
      // Clear the saved job id — from here on the user interacts with the
      // cluster view. Without this, pressing NEW PROMPT later would retrigger
      // the reconnect flow on GenerateView mount and bounce back.
      localStorage.removeItem('ltnt_active_job')
      const isFinal = data.status === 'done'
      const images = isFinal ? data.result.images : data.images
      onImages(images, jobId, data.interval, data.total_intervals, isFinal, prompt.trim())
    } catch (err) {
      localStorage.removeItem('ltnt_active_job')
      setError(err.message)
      setStatusText('')
      setLoading(false)
    }
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter') handleGenerate()
  }

  return (
    <div style={{ height: '100vh', display: 'flex', flexDirection: 'column', background: 'var(--bg)' }}>

      {/* Header */}
      <header style={{
        height: '56px',
        display: 'flex',
        alignItems: 'flex-start',
        justifyContent: 'space-between',
        padding: '14px 20px 0',
        flexShrink: 0,
        position: 'relative',
        zIndex: 10,
      }}>
        <div>
          <div style={{ textTransform: 'uppercase' }}>[ x ] LTNT.APP</div>
          <div style={{ color: 'var(--muted)', marginTop: '4px', textTransform: 'uppercase' }}>/GENERATE</div>
        </div>
        <div style={{ display: 'flex', gap: '10px' }}>
          <TourToggle tourMode={tourMode} onTourModeChange={onTourModeChange} />
          <button
            onClick={onBoard}
            title="Open your saved board"
            style={{
              border: '1px solid var(--text)',
              padding: '6px 14px',
              background: 'none',
              color: 'var(--text)',
              cursor: 'pointer',
              fontSize: '11px',
              letterSpacing: '0.04em',
              textTransform: 'uppercase',
            }}
          >
            ✦ /BOARD
          </button>
          <button
            data-tour="settings-toggle"
            onClick={() => setShowSettings(s => !s)}
            style={{
              border: '1px solid var(--text)',
              padding: '6px 14px',
              background: showSettings ? 'var(--text)' : 'none',
              color: showSettings ? 'var(--bg)' : 'var(--text)',
              cursor: 'pointer',
              fontSize: '11px',
              letterSpacing: '0.04em',
              textTransform: 'uppercase',
            }}
          >
            {showSettings ? '[ x ] SETTINGS' : '[ ] SETTINGS'}
          </button>
        </div>
      </header>

      {/* Canvas */}
      <main style={{ flex: 1, position: 'relative', overflow: 'hidden' }}>

        {/* Watermark */}
        <div style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -58%)',
          textAlign: 'center',
          pointerEvents: 'none',
          userSelect: 'none',
        }}>
          <div style={{
            fontSize: '110px',
            fontWeight: 700,
            color: loading ? 'rgba(0,0,0,0.04)' : 'rgba(0,0,0,0.065)',
            letterSpacing: '-4px',
            lineHeight: 1,
            transition: 'color 0.4s',
          }}>
            ltnt
          </div>
          <div style={{
            fontSize: '12px',
            color: 'var(--muted)',
            letterSpacing: '0.08em',
            textTransform: 'uppercase',
            marginTop: '10px',
          }}>
            {loading ? '' : 'Generate \u2192 Pick favorites \u2192 Make more'}
          </div>
          {!loading && (
            <div style={{
              fontSize: '10px', color: 'var(--muted)', letterSpacing: '0.06em',
              textTransform: 'uppercase', marginTop: '8px', lineHeight: 1.7,
            }}>
              Type an idea below — we paint a whole spread of takes on it,
              and you steer by picking favorites.
            </div>
          )}
          {!loading && (
            <button
              onClick={tour.start}
              style={{
                pointerEvents: 'auto', marginTop: '14px',
                border: '1px solid var(--border)', background: 'var(--panel)',
                color: 'var(--muted)', padding: '5px 14px', cursor: 'pointer',
                fontSize: '10px', letterSpacing: '0.08em', textTransform: 'uppercase',
              }}
            >
              ✦ First time? Take the 30-second tour
            </button>
          )}
        </div>

        {/* Settings panel */}
        {showSettings && !loading && (
          <div style={{
            position: 'absolute',
            top: '20px',
            right: '20px',
            width: '300px',
            // Never clip off the viewport edge: cap to the visible area and
            // scroll internally instead (critic polish #7).
            maxWidth: 'calc(100vw - 40px)',
            maxHeight: 'calc(100% - 40px)',
            overflowY: 'auto',
            boxSizing: 'border-box',
            background: 'var(--panel)',
            border: '1px solid var(--border)',
            padding: '16px',
            zIndex: 20,
            fontSize: '11px',
            letterSpacing: '0.03em',
          }}>
            <div style={{ fontWeight: 700, textTransform: 'uppercase', marginBottom: '14px', letterSpacing: '0.08em' }}>
              /GENERATION SETTINGS
            </div>

            <SettingRow label="Model" tooltip="Which image model paints your spread">
              <div style={{ display: 'flex', gap: '4px', flexWrap: 'wrap', justifyContent: 'flex-end', maxWidth: '200px' }}>
                {[['sd35', 'SD3.5'], ['flux', 'FLUX'], ['sana', 'SANA'], ['fluxfm', 'FluxFM'], ['krea2', 'KREA-2 (512)']].map(([val, label]) => (
                  <button
                    key={val}
                    onClick={() => setModel(val)}
                    style={{
                      padding: '4px 10px',
                      fontSize: '10px',
                      letterSpacing: '0.04em',
                      border: '1px solid var(--border)',
                      background: model === val ? 'var(--text)' : 'var(--panel)',
                      color: model === val ? 'var(--bg)' : 'var(--muted)',
                      cursor: 'pointer',
                    }}
                  >
                    {label}
                  </button>
                ))}
              </div>
            </SettingRow>

            <SettingRow label="Images" tooltip="How many images per breeding round">
              <NumberInput value={nImages} onChange={setNImages} min={1} max={64} />
            </SettingRow>

            <SettingRow label="Resolution">
              <select
                value={`${width}x${height}`}
                onChange={e => {
                  const [w, h] = e.target.value.split('x').map(Number)
                  setWidth(w); setHeight(h)
                }}
                style={{ border: '1px solid var(--border)', padding: '4px 8px', background: 'var(--panel)', fontSize: '11px' }}
              >
                <option value="512x512">512 x 512</option>
                <option value="768x768">768 x 768</option>
                <option value="1024x1024">1024 x 1024</option>
              </select>
            </SettingRow>

            <SettingRow label="Seed" tooltip="Leave empty for random">
              <input
                type="text"
                value={seed}
                onChange={e => setSeed(e.target.value.replace(/[^0-9]/g, ''))}
                placeholder="random"
                style={{
                  border: '1px solid var(--border)',
                  padding: '4px 8px',
                  width: '80px',
                  fontSize: '11px',
                  background: 'var(--panel)',
                }}
              />
            </SettingRow>

            {/* ── Advanced (plain labels; raw dials off the first surface) ── */}
            <button
              onClick={() => setShowAdvanced(a => !a)}
              style={{
                display: 'block', width: '100%', textAlign: 'left',
                border: 'none', background: 'none', cursor: 'pointer',
                padding: '8px 0 2px', fontFamily: 'inherit',
                fontSize: '10px', letterSpacing: '0.08em',
                textTransform: 'uppercase', color: 'var(--muted)',
              }}
            >
              [ {showAdvanced ? '−' : '+'} ] ADVANCED
            </button>

            {showAdvanced && (<>
            {false && model === 'fluxfm' && (
              <SettingRow label="Clone mode" tooltip="GLASS = exact stochastic bridge. Flow map = Weighted-Diamond-Maps amortization (1 NFE per clone, ~9× faster, biased but invisible for human-pick UX)">
                <div style={{ display: 'flex', gap: '4px' }}>
                  {[['glass', 'GLASS'], ['flow_map', 'Flow Map']].map(([val, label]) => (
                    <button
                      key={val}
                      onClick={() => setCloneMode(val)}
                      style={{
                        padding: '4px 10px',
                        fontSize: '10px',
                        letterSpacing: '0.04em',
                        border: '1px solid var(--border)',
                        background: cloneMode === val ? 'var(--text)' : 'var(--panel)',
                        color: cloneMode === val ? 'var(--bg)' : 'var(--muted)',
                        cursor: 'pointer',
                      }}
                    >
                      {label}
                    </button>
                  ))}
                </div>
              </SettingRow>
            )}

            <SettingRow label="Variations per pick" tooltip="How many new variations each picked image breeds (1 anchor + N-1 clones)">
              <NumberInput value={numClones} onChange={setNumClones} min={2} max={8} />
            </SettingRow>

            <SettingRow label="Quality steps" tooltip="More steps = finer detail, slower (ODE integration steps)">
              <NumberInput value={numSteps} onChange={setNumSteps} min={1} max={100} />
            </SettingRow>

            <SettingRow label="Picking rounds" tooltip="How many times you get to pick favorites and breed more (selection rounds)">
              <NumberInput value={numResamplingSteps} onChange={setNumResamplingSteps} min={0} max={20} />
            </SettingRow>

            <SettingRow label="Prompt strength" tooltip="How strictly images follow your prompt (classifier-free guidance scale)">
              <NumberInput value={guidanceScale} onChange={setGuidanceScale} min={1} max={20} step={0.5} />
            </SettingRow>

            <SettingRow label="Variation looseness" tooltip="How far bred variations can stray from their parent (GLASS rho; 0 = wildest)">
              <NumberInput value={rho} onChange={setRho} min={0} max={1} step={0.1} />
            </SettingRow>

            <SettingRow label="Schedule" tooltip="When in the process the picking rounds happen">
              <select
                value={scheduleMode}
                onChange={e => setScheduleMode(e.target.value)}
                style={{ border: '1px solid var(--border)', padding: '4px 8px', background: 'var(--panel)', fontSize: '11px' }}
              >
                <option value="">Normal</option>
                <option value="linear">Linear</option>
                <option value="aggressive">Aggressive</option>
              </select>
            </SettingRow>

            <SettingRow label="Layout" tooltip="How images are arranged">
              <div style={{ display: 'flex', gap: '4px' }}>
                {[['cluster', 'Cluster'], ['tree', 'Tree']].map(([val, label]) => (
                  <button
                    key={val}
                    onClick={() => setTreeMode(val === 'tree')}
                    style={{
                      padding: '4px 10px',
                      fontSize: '10px',
                      letterSpacing: '0.04em',
                      border: '1px solid var(--border)',
                      background: (val === 'tree') === treeMode ? 'var(--text)' : 'var(--panel)',
                      color: (val === 'tree') === treeMode ? 'var(--bg)' : 'var(--muted)',
                      cursor: 'pointer',
                    }}
                  >
                    {label}
                  </button>
                ))}
              </div>
            </SettingRow>
            </>)}
          </div>
        )}

        {/* Boot overlay — the backend is loading models (or restarting).
            Bound to /api/status: named phase + REAL progress, never dead air. */}
        {boot.checked && !boot.ready && !loading && (
          <div style={{
            position: 'absolute',
            inset: 0,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            background: 'var(--bg)',
            zIndex: 6,
          }}>
            <div style={{ textAlign: 'center', width: '360px' }}>
              <div style={{
                fontSize: '11px', letterSpacing: '0.1em',
                textTransform: 'uppercase', color: 'var(--muted)', marginBottom: '6px',
              }}>
                {boot.unreachable
                  ? '[ SERVER RESTARTING — RECONNECTING... ]'
                  : `[ WARMING UP: ${boot.phase || 'starting'} ]`}
              </div>
              {!boot.unreachable && boot.detail && (
                <div style={{
                  fontSize: '10px', letterSpacing: '0.06em',
                  textTransform: 'uppercase', color: 'var(--muted)', marginBottom: '14px',
                }}>
                  {boot.detail}
                </div>
              )}
              {!boot.unreachable && (
                <>
                  <PixelCatProgress progress={Math.round((boot.fraction || 0) * 100)} />
                  <div style={{
                    marginTop: '8px', fontSize: '10px',
                    letterSpacing: '0.08em', color: 'var(--muted)',
                  }}>
                    {Math.round((boot.fraction || 0) * 100)}%
                  </div>
                </>
              )}
              <div style={{
                marginTop: '14px', fontSize: '10px', lineHeight: 1.6,
                letterSpacing: '0.06em', textTransform: 'uppercase', color: 'var(--muted)',
              }}>
                One-time model loading — you can type your prompt meanwhile
              </div>
            </div>
          </div>
        )}

        {/* Loading overlay */}
        {loading && (
          <div style={{
            position: 'absolute',
            inset: 0,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            background: 'var(--bg)',
            zIndex: 5,
          }}>
            <div style={{ textAlign: 'center', width: '340px' }}>
              <div style={{
                fontSize: '11px',
                letterSpacing: '0.1em',
                textTransform: 'uppercase',
                color: 'var(--muted)',
                marginBottom: '16px',
              }}>
                {statusText}
              </div>
              <PixelCatProgress progress={progress} />
              <div style={{
                marginTop: '8px',
                fontSize: '10px',
                letterSpacing: '0.08em',
                color: 'var(--muted)',
              }}>
                {progress}%
              </div>
              {genReadings?.length > 1 && !stepPreviews?.urls?.length && (
                <div style={{
                  marginTop: '20px', textAlign: 'left', maxWidth: '340px',
                  marginLeft: 'auto', marginRight: 'auto',
                }}>
                  <div style={{
                    fontSize: '9px', letterSpacing: '0.1em',
                    color: 'var(--muted)', textTransform: 'uppercase',
                    marginBottom: '6px',
                  }}>
                    {genReadings.length} ways of reading your prompt
                  </div>
                  {genReadings.slice(0, 6).map((p, i) => (
                    <div key={i} style={{
                      fontSize: '10px', color: 'var(--text)', opacity: 0.85,
                      padding: '2px 0', whiteSpace: 'nowrap',
                      overflow: 'hidden', textOverflow: 'ellipsis',
                    }}>
                      {i === 0 ? '◉ ' : '✦ '}{p}
                    </div>
                  ))}
                </div>
              )}
              {stepPreviews?.urls?.length > 0 && (
                <div>
                  <div style={{
                    display: 'flex', gap: '8px', justifyContent: 'center',
                    marginTop: '22px', flexWrap: 'wrap',
                  }}>
                    {stepPreviews.urls.map((u, i) => (
                      <img
                        key={i}
                        src={`${u}?s=${stepPreviews.seq}`}
                        alt=""
                        style={{
                          width: '84px', height: '84px', objectFit: 'cover',
                          // Bright edge = the tile the GPU is refining now.
                          border: stepPreviews.active === i
                            ? '2px solid var(--text)'
                            : '1px solid var(--border)',
                          opacity: stepPreviews.active == null
                            || stepPreviews.active === i ? 1 : 0.7,
                          imageRendering: 'auto',
                        }}
                      />
                    ))}
                  </div>
                  <div style={{
                    marginTop: '8px', fontSize: '9px',
                    letterSpacing: '0.1em', color: 'var(--muted)',
                    textTransform: 'uppercase',
                  }}>
                    live — the model's current guess, sharpening each step
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* RECENT sessions — unobtrusive one-click return to a previous
            exploration (each entry restores via its ?session= link). */}
        {!loading && recent.length > 0 && (
          <div style={{
            position: 'absolute', left: '20px', bottom: '14px', zIndex: 4,
            maxWidth: '300px',
          }}>
            <div style={{
              fontSize: '9px', letterSpacing: '0.1em', textTransform: 'uppercase',
              color: 'var(--muted)', marginBottom: '6px',
            }}>
              Recent — pick up where you left off
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
              {recent.map(r => (
                <button
                  key={r.id}
                  onClick={() => { window.location.href = `/?session=${r.id}` }}
                  title={r.prompt || r.id}
                  style={{
                    display: 'flex', alignItems: 'center', gap: '8px',
                    border: '1px solid var(--border)', background: 'var(--panel)',
                    padding: '4px 8px', cursor: 'pointer', textAlign: 'left',
                    fontFamily: 'inherit',
                  }}
                >
                  {r.thumb && (
                    <img
                      src={r.thumb}
                      alt=""
                      style={{ width: '26px', height: '26px', objectFit: 'cover', flexShrink: 0 }}
                    />
                  )}
                  <span style={{
                    fontSize: '10px', color: 'var(--muted)', letterSpacing: '0.03em',
                    overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                  }}>
                    {r.prompt || `session ${r.id}`}
                  </span>
                </button>
              ))}
            </div>
          </div>
        )}

        {/* One-line plain-words legend for the footer toggles (INTERPRET /
            CREATE MAP were unexplained jargon on the first screen). */}
        {!loading && (
          <div style={{
            position: 'absolute', right: '20px', bottom: '14px', zIndex: 4,
            fontSize: '9px', letterSpacing: '0.06em', textTransform: 'uppercase',
            color: 'var(--muted)', textAlign: 'right', lineHeight: 1.8, pointerEvents: 'none',
            maxWidth: '380px',
          }}>
            <div>INTERPRET varies how your prompt is read (more diverse takes)</div>
            <div>CREATE MAP builds a zoomable overview of the whole prompt</div>
          </div>
        )}

        {/* Error */}
        {error && (
          <div style={{
            position: 'absolute',
            bottom: '14px',
            left: '50%',
            transform: 'translateX(-50%)',
            fontSize: '11px',
            color: 'var(--danger)',
            letterSpacing: '0.05em',
            textTransform: 'uppercase',
            background: 'var(--panel)',
            padding: '6px 12px',
            border: '1px solid var(--danger-border)',
          }}>
            <span>ERROR: {error}</span>
            <button
              onClick={() => setError('')}
              title="Dismiss"
              style={{
                marginLeft: 10, border: 'none', background: 'none',
                color: 'var(--danger)', cursor: 'pointer', fontSize: '12px',
              }}
            >
              ✕
            </button>
          </div>
        )}

      </main>

      {/* Footer bar */}
      <footer style={{
        height: '80px',
        background: '#111',
        display: 'flex',
        alignItems: 'center',
        padding: '0 16px',
        gap: '16px',
        flexShrink: 0,
      }}>
        <input
          data-tour="prompt-input"
          value={prompt}
          onChange={e => setPrompt(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={loading}
          placeholder="[ ] Enter prompt to generate variations..."
          style={{
            flex: 1,
            color: loading ? 'rgba(255,255,255,0.4)' : 'rgba(255,255,255,0.85)',
            background: 'transparent',
            border: 'none',
            outline: 'none',
            fontSize: '13px',
            letterSpacing: '0.03em',
          }}
        />
        <button
          onClick={() => { if (!loading) setShowSettings(true) }}
          disabled={loading}
          title="Choose the image model in Settings"
          style={{
            color: 'rgba(255,255,255,0.55)',
            background: 'transparent',
            border: '1px solid rgba(255,255,255,0.18)',
            padding: '6px 12px',
            fontSize: '10px',
            letterSpacing: '0.06em',
            textTransform: 'uppercase',
            whiteSpace: 'nowrap',
            flexShrink: 0,
            cursor: loading ? 'not-allowed' : 'pointer',
          }}
        >
          MODEL: {MODEL_LABELS[model] || model.toUpperCase()}{!isAvail(model) ? ' (unavailable)' : ''} ▸
        </button>
        {/* DIVERSITY preset dial removed (Jerry 2026-08-27: staged
            contraction is now the DEFAULT sampler behavior; no named modes).
            Advanced settings still expose raw rho for engineers. */}
        <button
          onClick={() => !loading && setAugmentOn(v => !v)}
          disabled={loading}
          title="Prompt variations for a more DIVERSE spread: an on-device LLM explores different interpretations of your prompt so the first images vary in style/mood/take. Off = every image stays literal to your exact prompt (near-identical spread)."
          style={{
            background: 'transparent',
            border: '1px solid rgba(255,255,255,0.18)',
            padding: '6px 12px', fontSize: '10px', letterSpacing: '0.06em',
            textTransform: 'uppercase', whiteSpace: 'nowrap', flexShrink: 0,
            cursor: loading ? 'not-allowed' : 'pointer',
            color: augmentOn ? 'white' : 'rgba(255,255,255,0.4)',
          }}
        >
          VARIATIONS: {augmentOn ? 'ON' : 'OFF'} ▸
        </button>
        <label
          title="How many images per round — type any number 2-18. More = a wider view of the latent space. On KREA-2 each extra image costs ~12s (serial); on FLUX extra images are nearly free (batched)."
          style={{
            background: 'transparent',
            border: '1px solid rgba(255,255,255,0.18)',
            padding: '6px 12px', fontSize: '10px', letterSpacing: '0.06em',
            textTransform: 'uppercase', whiteSpace: 'nowrap', flexShrink: 0,
            color: 'rgba(255,255,255,0.55)', display: 'inline-flex',
            alignItems: 'center', gap: '4px',
          }}
        >
          SPREAD:
          <input
            type="number" min="2" max="18" value={nImages}
            disabled={loading}
            onChange={e => {
              const v = parseInt(e.target.value || '0', 10)
              if (!Number.isNaN(v)) setNImages(Math.min(18, Math.max(2, v)))
            }}
            style={{
              width: '34px', background: 'transparent', border: 'none',
              borderBottom: '1px solid rgba(255,255,255,0.3)',
              color: 'rgba(255,255,255,0.8)', fontSize: '11px',
              textAlign: 'center', outline: 'none',
            }}
          />
        </label>
        <button
          onClick={() => !loading && setPreviewMode(m =>
            m === 'SKETCH' ? 'STUDY' : m === 'STUDY' ? 'RENDER' : 'SKETCH')}
          disabled={loading}
          title="Preview depth per round. SKETCH = fastest, roughest tiles (2 refine steps) — for rapid exploration. STUDY = the balance (4). RENDER = slower, richer previews (8) — when you're judging detail. Early roughness stays legible and foreshadows the final image."
          style={{
            background: 'transparent',
            border: '1px solid rgba(255,255,255,0.18)',
            padding: '6px 12px', fontSize: '10px', letterSpacing: '0.06em',
            textTransform: 'uppercase', whiteSpace: 'nowrap', flexShrink: 0,
            cursor: loading ? 'not-allowed' : 'pointer',
            color: 'rgba(255,255,255,0.55)',
          }}
        >
          PREVIEW: {previewMode} ▸
        </button>
        {/* Model/schedule readout moved into /GENERATION SETTINGS \u2014 kept off the
            landing bar to avoid engineer-facing jargon on the primary surface.
            Selectors live in the Settings panel (Model + Schedule rows). */}
        <button
          data-tour="generate-btn"
          onClick={loading ? cancelGeneration : handleGenerate}
          disabled={!loading && (!prompt.trim() || !bootReady)}
          title={!bootReady ? 'The image model is still warming up — the bar above shows real progress' : ''}
          style={{
            // PRIMARY CTA: filled when actionable so it clearly outranks the
            // secondary CREATE MAP button beside it.
            color: loading || !prompt.trim() || !bootReady ? 'rgba(255,255,255,0.3)' : '#111',
            border: `1px solid ${loading || !prompt.trim() || !bootReady ? 'rgba(255,255,255,0.1)' : '#fff'}`,
            padding: '10px 22px',
            fontWeight: 700,
            background: loading || !prompt.trim() || !bootReady ? 'transparent' : '#fff',
            whiteSpace: 'nowrap',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            cursor: loading ? 'not-allowed' : 'pointer',
          }}
        >
          {loading ? '\u2715 CANCEL' : '\u2726 [ ] GENERATE'}
        </button>
        {/* EXPLORE MAP button stashed — uncomment the block in git history (or the tour step above) to restore. */}
        {/* === CREATE MAP button (delete this block to remove feature) === */}
        <button
          data-tour="create-map-btn"
          onClick={() => {
            if (!prompt.trim() || loading) return
            onCreateMap?.(prompt.trim(), model, augmentOn)
          }}
          disabled={loading || !prompt.trim()}
          title="Precompute a zoomable map of the prompt — takes a minute or two"
          style={{
            // SECONDARY action: visually subordinate to GENERATE (the primary
            // CTA) — smaller, dimmer, no competing border weight.
            padding: '8px 12px',
            fontSize: '10px',
            fontFamily: 'inherit',
            letterSpacing: '0.08em',
            color: loading || !prompt.trim() ? 'rgba(255,255,255,0.2)' : 'rgba(255,255,255,0.45)',
            border: '1px solid rgba(255,255,255,0.12)',
            background: 'transparent',
            whiteSpace: 'nowrap',
            cursor: loading || !prompt.trim() ? 'not-allowed' : 'pointer',
            marginLeft: 8,
          }}
        >
          {'◉ CREATE MAP'}
        </button>
        {/* === END CREATE MAP === */}
      </footer>

      <Tour
        steps={GENERATE_TOUR_STEPS}
        open={tour.open}
        onClose={() => { tour.close(); onTourModeChange(false) }}
        storageKey="ltnt.tour.generate"
      />
    </div>
  )
}

// ── Settings helpers ──────────────────────────────────────────────────────────

function SettingRow({ label, tooltip, children }) {
  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      padding: '6px 0',
      borderTop: '1px solid var(--border)',
    }}>
      <span style={{ textTransform: 'uppercase', color: 'var(--muted)' }} title={tooltip}>{label}</span>
      {children}
    </div>
  )
}

function NumberInput({ value, onChange, min, max, step = 1 }) {
  return (
    <input
      type="number"
      value={value}
      onChange={e => onChange(Number(e.target.value))}
      min={min}
      max={max}
      step={step}
      style={{
        border: '1px solid var(--border)',
        padding: '4px 8px',
        width: '70px',
        fontSize: '11px',
        textAlign: 'right',
        background: 'var(--panel)',
      }}
    />
  )
}

// Pixel cat frames (2-frame walk cycle drawn with box-shadows)
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
        height: '2px', background: 'var(--tile-empty)',
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
