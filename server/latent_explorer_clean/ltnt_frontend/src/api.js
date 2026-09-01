const BASE = '/api'

/**
 * Start a generation job.
 * @param {string} prompt
 * @param {object} params - Generation parameters
 * @returns {Promise<string>} job_id
 */
export async function startGenerate(prompt, params = {}) {
  const res = await fetch(`${BASE}/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt, ...params }),
  })
  if (!res.ok) throw new Error(`Generate failed: ${res.statusText}`)
  const data = await res.json()
  // === SESSION REHYDRATION === the durable session id is minted server-side
  // at t=0 and returned here. Put it in the URL IMMEDIATELY (not after round
  // 1) so a refresh at ANY point — even mid-model-load — can rehydrate and
  // reattach to the running job instead of orphaning it.
  if (data.session_id) {
    try {
      const u = new URL(window.location.href)
      u.searchParams.delete('map')
      u.searchParams.delete('view')
      u.searchParams.set('session', data.session_id)
      window.history.replaceState({}, '', u)
    } catch {}
    try {
      localStorage.setItem('ltnt_last_session', data.session_id)
      localStorage.setItem('ltnt.current_session', data.session_id)
    } catch {}
  }
  // === END SESSION REHYDRATION ===
  return data.job_id
}

/**
 * Poll job status once.
 * @returns {Promise<object>} job data
 */
export async function getJobStatus(jobId) {
  const res = await fetch(`${BASE}/jobs/${jobId}`)
  if (!res.ok) throw new Error(`Poll failed: ${res.statusText}`)
  return res.json()
}

/**
 * Poll a job until it reaches a terminal or actionable state.
 * Resolves when status is 'done', 'waiting_selection', or 'error'.
 * @param {string} jobId
 * @param {function} onProgress - called with (status, data) on each poll
 * @param {number} intervalMs - poll interval
 * @returns {Promise<object>} full job data at terminal/actionable state
 */
export function pollJob(jobId, onProgress, intervalMs = 2000) {
  return new Promise((resolve, reject) => {
    const check = async () => {
      try {
        const data = await getJobStatus(jobId)
        onProgress(data.status, data)

        if (data.status === 'done' || data.status === 'waiting_selection') {
          resolve(data)
        } else if (data.status === 'error') {
          reject(new Error(data.error))
        } else {
          setTimeout(check, intervalMs)
        }
      } catch (err) {
        reject(err)
      }
    }
    setTimeout(check, intervalMs)
  })
}

/**
 * Submit selected particle indices to continue generation.
 * @param {string} jobId
 * @param {number[]} indices - selected particle indices
 * @returns {Promise<object>}
 */
export async function submitSelection(jobId, indices) {
  const res = await fetch(`${BASE}/jobs/${jobId}/select`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ indices }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || 'Selection failed')
  }
  return res.json()
}

/**
 * Request full-quality renders of particle indices during waiting_selection.
 * Refined URLs then appear in job polling under data.refined ({index: url});
 * the job returns to waiting_selection afterward.
 * @param {string} jobId
 * @param {number[]} indices
 * @returns {Promise<object>}
 */
export async function requestRefine(jobId, indices) {
  const res = await fetch(`${BASE}/jobs/${jobId}/refine`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ indices }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || 'Refine failed')
  }
  return res.json()
}

/**
 * SPAN MODE early-stop: gracefully end the streaming round-1 span after the
 * in-flight batch ("spread looks good — start picking"). The job then flips
 * to waiting_selection on its own.
 * @param {string} jobId
 * @returns {Promise<object>}
 */
export async function finishRound(jobId) {
  const res = await fetch(`${BASE}/jobs/${jobId}/finish_round`, { method: 'POST' })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || 'Finish round failed')
  }
  return res.json()
}

/**
 * Adjust layout with repulsion/cohesion forces.
 * @param {string} jobId
 * @param {number} repulsion - 0-1
 * @param {number} cohesion - 0-1
 * @returns {Promise<object>} { images: [...] } with updated positions
 */
export async function adjustLayout(jobId, repulsion, cohesion, cohesionSpring = 0) {
  const res = await fetch(`${BASE}/jobs/${jobId}/adjust`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ repulsion, cohesion, cohesion_spring: cohesionSpring }),
  })
  if (!res.ok) throw new Error('Adjust failed')
  return res.json()
}

/**
 * Start pre-computing an explore map.
 */
export async function startExplore(params = {}) {
  const res = await fetch(`${BASE}/explore/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  })
  if (!res.ok) throw new Error('Explore generate failed')
  const data = await res.json()
  return data.job_id
}

/**
 * Poll explore job status.
 */
export async function getExploreStatus(jobId) {
  const res = await fetch(`${BASE}/explore/${jobId}`)
  if (!res.ok) throw new Error('Explore poll failed')
  return res.json()
}

/**
 * Poll explore job until done.
 */
export function pollExplore(jobId, onProgress, intervalMs = 3000) {
  return new Promise((resolve, reject) => {
    const check = async () => {
      try {
        const data = await getExploreStatus(jobId)
        onProgress(data.status, data)
        if (data.status === 'done') resolve(data)
        else if (data.status === 'error') reject(new Error(data.error))
        else setTimeout(check, intervalMs)
      } catch (err) { reject(err) }
    }
    setTimeout(check, intervalMs)
  })
}

/**
 * Check backend health.
 * @returns {Promise<object>} { status, device }
 */
export async function checkHealth() {
  const res = await fetch(`${BASE}/health`)
  if (!res.ok) throw new Error('Backend unreachable')
  return res.json()
}

/**
 * Boot / model-readiness status with REAL progress.
 * @returns {Promise<object>} { ready, phase, fraction, detail, device }
 * Old backends without the endpoint are treated as ready (404 -> ready).
 */
export async function getServerStatus() {
  const res = await fetch(`${BASE}/status`)
  if (res.status === 404) return { ready: true, phase: 'ready', fraction: 1 }
  if (!res.ok) throw new Error('Status unavailable')
  return res.json()
}

/**
 * Session rehydration — everything the frontend needs to rebuild a session's
 * canvas after navigation/refresh (images + coords, status, frozen history,
 * refined map, prompt, per-round URLs). See GET /api/sessions/{id}/state.
 */
export async function getSessionState(sessionId) {
  const res = await fetch(`${BASE}/sessions/${encodeURIComponent(sessionId)}/state`)
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || 'Session state failed')
  }
  return res.json()
}

// ── Image helpers (for EditView) ──────────────────────────────────────────────

function loadImage(src) {
  return new Promise((resolve, reject) => {
    if (!src) return reject(new Error('loadImage: src is undefined'))
    const img = new Image()
    if (!src.startsWith('data:')) img.crossOrigin = 'anonymous'
    img.onload = () => resolve(img)
    img.onerror = () => reject(new Error(`Failed to load image: ${src.slice(0, 80)}`))
    img.src = src
  })
}

export async function blendImages(baseSrc, editedB64, intensity) {
  const [baseImg, editedImg] = await Promise.all([
    loadImage(baseSrc),
    loadImage(`data:image/png;base64,${editedB64}`),
  ])
  const canvas = document.createElement('canvas')
  canvas.width  = editedImg.naturalWidth
  canvas.height = editedImg.naturalHeight
  const ctx = canvas.getContext('2d')
  ctx.drawImage(baseImg, 0, 0, canvas.width, canvas.height)
  ctx.globalAlpha = intensity / 100
  ctx.drawImage(editedImg, 0, 0)
  ctx.globalAlpha = 1
  return canvas.toDataURL('image/png')
}

export async function compositeEdit(baseSrc, editedB64, region) {
  const [baseImg, editedImg] = await Promise.all([
    loadImage(baseSrc),
    loadImage(`data:image/png;base64,${editedB64}`),
  ])
  const canvas = document.createElement('canvas')
  canvas.width = baseImg.naturalWidth
  canvas.height = baseImg.naturalHeight
  const ctx = canvas.getContext('2d')
  ctx.drawImage(baseImg, 0, 0)
  const rx = region.x * canvas.width
  const ry = region.y * canvas.height
  const rw = region.w * canvas.width
  const rh = region.h * canvas.height
  ctx.drawImage(editedImg, rx, ry, rw, rh)
  return canvas.toDataURL('image/png')
}

export async function getEffects() {
  const res = await fetch(`${BASE}/effects`)
  if (!res.ok) throw new Error(`Failed to load effects: ${res.statusText}`)
  const data = await res.json()
  return data.effects ?? data
}

/**
 * Composite a stack of layers into a single image.
 * Each layer: { src, intensity?, region? }
 * Base layer is drawn first, subsequent layers blended on top.
 */
// Resolve a layer's image source: prefer `src`, fall back to base64 payload
// from a Reve result (`editedB64`).
function layerSrc(layer) {
  if (!layer) return null
  if (layer.src) return layer.src
  if (layer.editedB64) return `data:image/png;base64,${layer.editedB64}`
  return null
}

export async function compositeLayerStack(layers) {
  if (!layers || layers.length === 0) return null
  const baseSrc = layerSrc(layers[0])
  if (!baseSrc) return null
  const baseImg = await loadImage(baseSrc)
  const canvas = document.createElement('canvas')
  canvas.width = baseImg.naturalWidth
  canvas.height = baseImg.naturalHeight
  const ctx = canvas.getContext('2d')
  ctx.drawImage(baseImg, 0, 0)

  for (let i = 1; i < layers.length; i++) {
    const layer = layers[i]
    if (layer.visible === false) continue
    const src = layerSrc(layer)
    if (!src) continue
    const img = await loadImage(src)
    const intensity = (layer.intensity ?? 100) / 100
    ctx.globalAlpha = intensity

    if (layer.region) {
      // Region edit: the returned image is the edited CROP of the user's
      // bounding-box. Stretch it back into those coordinates. For lasso
      // selections, additionally clip the paste to the polygon shape so
      // pixels inside the bbox-but-outside-the-lasso stay untouched.
      const rx = layer.region.x * canvas.width
      const ry = layer.region.y * canvas.height
      const rw = layer.region.w * canvas.width
      const rh = layer.region.h * canvas.height

      if (layer.selectionType === 'lasso' && Array.isArray(layer.normalizedRegions)) {
        const path = new Path2D()
        for (const region of layer.normalizedRegions) {
          const pts = region?.points
          if (!pts || pts.length < 3) continue
          path.moveTo(pts[0].x * canvas.width, pts[0].y * canvas.height)
          for (let k = 1; k < pts.length; k++) {
            path.lineTo(pts[k].x * canvas.width, pts[k].y * canvas.height)
          }
          path.closePath()
        }
        ctx.save()
        ctx.clip(path)
        ctx.drawImage(img, rx, ry, rw, rh)
        ctx.restore()
      } else {
        ctx.drawImage(img, rx, ry, rw, rh)
      }
    } else {
      // Whole-image layer (vary, full-image edit, effect, etc.).
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height)
    }
    ctx.globalAlpha = 1
  }
  return canvas.toDataURL('image/png')
}

export async function editRegion(imageSrc, options) {
  const img = await loadImage(imageSrc)
  const canvas = document.createElement('canvas')
  canvas.width = img.naturalWidth
  canvas.height = img.naturalHeight
  canvas.getContext('2d').drawImage(img, 0, 0)
  const image_b64 = canvas.toDataURL('image/png').split(',')[1]

  const body = {
    image_b64,
    operation:      options.operation ?? 'edit',
    prompt:         options.prompt ?? '',
    region:         options.region ?? null,
    effect_name:    options.effectName ?? null,
    upscale_factor: options.upscaleFactor ?? 2,
  }

  const res = await fetch(`${BASE}/edit`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || res.statusText)
  }
  return res.json()
}


// ── Board (AI-Pinterest pin store) — backend-persisted ─────────────────────
// Pins COPY the image into a durable server dir so they survive between
// sessions (generated/ may be cleaned). See server board_api.py.

/** GET the board, newest first. -> [{ id, url, prompt, savedAt }] */
export async function fetchBoard() {
  const res = await fetch(`${BASE}/board`)
  if (!res.ok) throw new Error(`Board load failed: ${res.statusText}`)
  const data = await res.json()
  return Array.isArray(data.pins) ? data.pins : []
}

/** Pin one image. body { url, prompt }. -> pin record. Deduped server-side by url. */
export async function pinToBoard(url, prompt = '') {
  const res = await fetch(`${BASE}/board/pin`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url, prompt }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || 'Pin failed')
  }
  return res.json()
}

/** Unpin by id (removes record + image file). */
export async function unpinFromBoard(id) {
  const res = await fetch(`${BASE}/board/${id}`, { method: 'DELETE' })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || 'Unpin failed')
  }
  return res.json()
}
