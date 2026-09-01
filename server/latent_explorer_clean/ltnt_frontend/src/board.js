// ── Board (pin) store ─────────────────────────────────────────────────────
// Frontend-only "save to board" persistence backed by localStorage. Matches
// the app's existing localStorage convention (see ltnt.* / ltnt_active_job).
// A pin = { url, prompt, ts }. Deduped by url.
const KEY = 'ltnt_board'

export function getPins() {
  try {
    const raw = localStorage.getItem(KEY)
    if (!raw) return []
    const arr = JSON.parse(raw)
    return Array.isArray(arr) ? arr : []
  } catch {
    return []
  }
}

function save(pins) {
  try { localStorage.setItem(KEY, JSON.stringify(pins)) } catch {}
}

export function isPinned(url) {
  return getPins().some(p => p.url === url)
}

// Add a single pin (deduped by url). Newest first. Returns the updated list.
export function addPin(pin) {
  if (!pin || !pin.url) return getPins()
  const pins = getPins()
  if (pins.some(p => p.url === pin.url)) return pins
  const next = [{ url: pin.url, prompt: pin.prompt || '', ts: pin.ts || Date.now() }, ...pins]
  save(next)
  return next
}

export function removePin(url) {
  const next = getPins().filter(p => p.url !== url)
  save(next)
  return next
}

// Convenience: pin many at once (e.g. a selection). Returns updated list.
export function addPins(list = []) {
  let pins = getPins()
  const existing = new Set(pins.map(p => p.url))
  const toAdd = list
    .filter(p => p && p.url && !existing.has(p.url))
    .map(p => ({ url: p.url, prompt: p.prompt || '', ts: p.ts || Date.now() }))
  if (toAdd.length === 0) return pins
  pins = [...toAdd, ...pins]
  save(pins)
  return pins
}
