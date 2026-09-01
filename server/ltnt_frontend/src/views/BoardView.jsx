import { useState, useEffect, useCallback, useMemo } from 'react'
import { fetchBoard, unpinFromBoard } from '../api'
import PixelCatProgress from '../components/PixelCatProgress'
import Lightbox from '../components/Lightbox'

// EXPLORE strength presets — how far "explore from this" roams from the pin.
// CLOSE = stays near the pinned image (subtle variations); FAR = diverges more.
const EXPLORE_PRESETS = [
  { label: 'CLOSE', strength: 0.3 },
  { label: 'BALANCED', strength: 0.55 },
  { label: 'FAR', strength: 0.8 },
]

// ── BOARD view ────────────────────────────────────────────────────────────
// Pinterest-style grid of pins saved from the cluster view. BACKEND-PERSISTED:
// pins are copied server-side into board_store/ and survive across sessions
// and devices (GET /api/board). Matches the monospace terminal aesthetic.
export default function BoardView({ onBack, onExploreFromPin = null }) {
  const [pins, setPins] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [hovered, setHovered] = useState(null)
  const [busy, setBusy] = useState(null)  // id currently being unpinned
  // EXPLORE FROM THIS PIN — once clicked, we show a full-view launching overlay
  // (with the SAME live progress bar as the first-generate flow) while the
  // seeded job runs (App owns the start+poll+transition machinery).
  const [exploring, setExploring] = useState(false)
  const [exploreStage, setExploreStage] = useState('')
  const [explorePct, setExplorePct] = useState(0)
  const [exploreIdx, setExploreIdx] = useState(1)  // BALANCED by default
  // Track which pin images have finished loading -> skeleton shimmer until then.
  const [loadedIds, setLoadedIds] = useState(() => new Set())
  // Full-size viewer.
  const [lightboxIdx, setLightboxIdx] = useState(null)

  // ── Session scoping ──────────────────────────────────────────────────────
  // ClusterView records the live session id in localStorage; pins carry the
  // sourceUrl ('/images/<session>/...') they were saved from. Default to
  // showing just this session's pins (with an ALL PINS toggle) so the board
  // opens on what the artist is working on right now.
  const sessionId = useMemo(() => {
    try { return localStorage.getItem('ltnt.current_session') || '' } catch { return '' }
  }, [])
  const sessionPins = useMemo(
    () => (sessionId
      ? pins.filter(p => (p.sourceUrl || '').startsWith(`/images/${sessionId}/`))
      : []),
    [pins, sessionId],
  )
  // 'session' | 'all' — start scoped to the session when it has pins.
  const [scope, setScope] = useState(null)
  useEffect(() => {
    if (scope === null && !loading) {
      setScope(sessionPins.length > 0 ? 'session' : 'all')
    }
  }, [loading, sessionPins.length, scope])
  const shownPins = scope === 'session' ? sessionPins : pins

  async function explorePin(pin) {
    if (!onExploreFromPin || exploring) return
    setExploring(true)
    setExploreStage('Starting a new exploration from this image...')
    setExplorePct(0)
    try {
      await onExploreFromPin(
        { prompt: pin.prompt || '', init_image_url: pin.url, init_strength: EXPLORE_PRESETS[exploreIdx].strength },
        (stage, pct) => {
          if (stage) setExploreStage(stage)
          if (pct != null) setExplorePct(pct)
        },
      )
      // On success App switches to the cluster view; nothing more to do here.
    } catch (e) {
      setExploring(false)
      setError(e.message || 'Failed to start exploration')
    }
  }

  const load = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      setPins(await fetchBoard())
    } catch (e) {
      setError(e.message || 'Failed to load board')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { load() }, [load])

  async function unpin(id) {
    setBusy(id)
    // Optimistic removal; restore on failure.
    const prev = pins
    setPins(pins.filter(p => p.id !== id))
    try {
      await unpinFromBoard(id)
    } catch {
      setPins(prev)
    } finally {
      setBusy(null)
    }
  }

  return (
    <div style={{
      height: '100vh', display: 'flex', flexDirection: 'column',
      background: '#e8e8e8',
    }}>
      {/* EXPLORE-FROM-PIN launching overlay — same live progress UI as the
          first-generate flow, so this wait is never a dead gray screen. */}
      {exploring && (
        <div style={{
          position: 'fixed', inset: 0, zIndex: 100,
          background: 'rgba(232,232,232,0.96)',
          display: 'flex', flexDirection: 'column',
          alignItems: 'center', justifyContent: 'center', gap: '14px',
          textTransform: 'uppercase', letterSpacing: '0.1em',
        }}>
          <div style={{ fontSize: '48px', opacity: 0.3 }}>{'⊕'}</div>
          <div style={{ fontSize: '13px', color: '#666' }}>[ EXPLORING FROM YOUR PIN ]</div>
          <div style={{ fontSize: '11px', color: '#999', maxWidth: '440px', textAlign: 'center' }}>
            {'—'} {exploreStage || 'Starting...'}
          </div>
          <div style={{ width: '340px' }}>
            <PixelCatProgress progress={explorePct} />
            <div style={{
              marginTop: '8px', fontSize: '10px', textAlign: 'center',
              letterSpacing: '0.08em', color: '#999',
            }}>
              {explorePct}%
            </div>
          </div>
          <div style={{
            fontSize: '10px', color: '#aaa', maxWidth: '440px',
            textAlign: 'center', letterSpacing: '0.06em', lineHeight: 1.6,
          }}>
            WE'RE GENERATING A FRESH ROUND OF IMAGES SEEDED FROM YOUR PIN —
            YOU'LL PICK YOUR FAVORITES WHEN THEY'RE READY
          </div>
        </div>
      )}
      {/* Top bar */}
      <div style={{
        height: '56px', display: 'flex', alignItems: 'center',
        justifyContent: 'space-between', padding: '0 20px',
        borderBottom: '1px solid #d0d0d0', flexShrink: 0, background: '#fff',
      }}>
        <button onClick={onBack} style={{
          border: '1px solid #1a1a1a', padding: '6px 14px',
          background: 'none', cursor: 'pointer',
          textTransform: 'uppercase', fontSize: '11px', letterSpacing: '0.04em',
        }}>
          {'←'} [ ] BACK
        </button>
        <div style={{
          textTransform: 'uppercase', letterSpacing: '0.1em',
          fontSize: '12px', fontWeight: 700,
        }}>
          {'✦'} /BOARD
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          {/* Scope toggle: this session's pins vs everything ever saved. */}
          {sessionId && sessionPins.length > 0 && (
            <button
              onClick={() => setScope(s => (s === 'session' ? 'all' : 'session'))}
              title="Show only images saved in this session, or everything ever saved"
              style={{
                border: '1px solid #1a1a1a', padding: '6px 12px',
                background: 'none', cursor: 'pointer',
                textTransform: 'uppercase', fontSize: '11px', letterSpacing: '0.04em',
                whiteSpace: 'nowrap',
              }}
            >
              VIEW: {scope === 'session' ? `THIS SESSION (${sessionPins.length})` : `ALL PINS (${pins.length})`} {'▸'}
            </button>
          )}
          {onExploreFromPin && (
            <button
              onClick={() => setExploreIdx((i) => (i + 1) % EXPLORE_PRESETS.length)}
              title="How far ‘explore from this’ roams from the pin"
              style={{
                border: '1px solid #1a1a1a', padding: '6px 12px',
                background: 'none', cursor: 'pointer',
                textTransform: 'uppercase', fontSize: '11px', letterSpacing: '0.04em',
                whiteSpace: 'nowrap',
              }}
            >
              {'⊕'} EXPLORE: {EXPLORE_PRESETS[exploreIdx].label} {'▸'}
            </button>
          )}
          <div style={{
            textTransform: 'uppercase', color: '#888',
            letterSpacing: '0.08em', fontSize: '11px',
          }}>
            {loading ? '⟳ LOADING' : `${shownPins.length} PIN${shownPins.length === 1 ? '' : 'S'}`}
          </div>
        </div>
      </div>

      {/* Grid / empty / error states */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '24px' }}>
        {error ? (
          <div style={{
            height: '100%', display: 'flex', flexDirection: 'column',
            alignItems: 'center', justifyContent: 'center', gap: '14px',
            color: '#999', textTransform: 'uppercase', letterSpacing: '0.1em',
          }}>
            <div style={{ fontSize: '13px', color: '#a33' }}>[ ERROR ]</div>
            <div style={{ fontSize: '11px', color: '#999' }}>{'—'} {error}</div>
            <button onClick={load} style={{
              border: '1px solid #1a1a1a', padding: '6px 14px', background: 'none',
              cursor: 'pointer', textTransform: 'uppercase', fontSize: '11px',
              letterSpacing: '0.04em', marginTop: '6px',
            }}>[ ] RETRY</button>
          </div>
        ) : loading ? (
          /* Skeleton grid while the board index loads — no blank white screen. */
          <div style={{ columnGap: '16px', columnWidth: '220px' }}>
            {[180, 240, 200, 160, 220, 190, 210, 170].map((hgt, i) => (
              <div
                key={i}
                className="ltnt-tile-loading"
                style={{
                  breakInside: 'avoid', marginBottom: '16px',
                  height: `${hgt}px`, border: '1px solid #d0d0d0',
                }}
              />
            ))}
          </div>
        ) : shownPins.length === 0 ? (
          <div style={{
            height: '100%', display: 'flex', flexDirection: 'column',
            alignItems: 'center', justifyContent: 'center', gap: '14px',
            color: '#999', textTransform: 'uppercase', letterSpacing: '0.1em',
          }}>
            <div style={{ fontSize: '48px', opacity: 0.3 }}>{'☆'}</div>
            <div style={{ fontSize: '13px', color: '#666' }}>[ EMPTY ]</div>
            <div style={{ fontSize: '11px', color: '#999' }}>
              {'—'} SAVE IMAGES YOU LOVE FROM THE CANVAS ("+ SAVE TO BOARD")
            </div>
            {scope === 'session' && pins.length > 0 && (
              <button onClick={() => setScope('all')} style={{
                border: '1px solid #1a1a1a', padding: '6px 14px', background: 'none',
                cursor: 'pointer', textTransform: 'uppercase', fontSize: '11px',
                letterSpacing: '0.04em', marginTop: '6px',
              }}>
                SHOW ALL PINS ({pins.length})
              </button>
            )}
          </div>
        ) : (
          <div style={{
            columnGap: '16px',
            columnWidth: '220px',
          }}>
            {shownPins.map((pin, idx) => (
              <div
                key={pin.id}
                className="ltnt-pin"
                tabIndex={0}
                onMouseEnter={() => setHovered(pin.id)}
                onMouseLeave={() => setHovered(null)}
                onFocus={() => setHovered(pin.id)}
                onBlur={(e) => { if (!e.currentTarget.contains(e.relatedTarget)) setHovered(null) }}
                style={{
                  breakInside: 'avoid',
                  marginBottom: '16px',
                  position: 'relative',
                  border: '1px solid #d0d0d0',
                  background: '#fff',
                  opacity: busy === pin.id ? 0.4 : 1,
                }}
              >
                {/* Skeleton shimmer sits behind the image until it loads. */}
                <div
                  className={loadedIds.has(pin.id) ? undefined : 'ltnt-tile-loading'}
                  style={{
                    minHeight: loadedIds.has(pin.id) ? 0 : '180px',
                    cursor: 'zoom-in',
                  }}
                  onClick={() => setLightboxIdx(idx)}
                  title="Click to view full size"
                >
                  <img
                    src={pin.url}
                    alt={pin.prompt || 'pin'}
                    style={{ width: '100%', display: 'block' }}
                    draggable={false}
                    onLoad={() => setLoadedIds(prev => {
                      const next = new Set(prev); next.add(pin.id); return next
                    })}
                  />
                </div>
                {hovered === pin.id && (
                  <div style={{
                    position: 'absolute', top: '8px', right: '8px',
                    display: 'flex', gap: '6px', alignItems: 'flex-start',
                  }}>
                    {onExploreFromPin && (
                      <button
                        onClick={() => explorePin(pin)}
                        disabled={exploring}
                        title="Start a new exploration seeded from this image"
                        style={{
                          border: '1px solid #1a1a1a',
                          background: 'rgba(26,26,26,0.92)',
                          color: '#fff',
                          cursor: exploring ? 'not-allowed' : 'pointer',
                          padding: '4px 8px', fontSize: '10px',
                          textTransform: 'uppercase', letterSpacing: '0.06em',
                          whiteSpace: 'nowrap',
                        }}
                      >
                        {'⊕'} EXPLORE FROM THIS
                      </button>
                    )}
                    <button
                      onClick={() => unpin(pin.id)}
                      disabled={busy === pin.id}
                      style={{
                        border: '1px solid #1a1a1a',
                        background: 'rgba(255,255,255,0.92)',
                        color: '#1a1a1a',
                        cursor: busy === pin.id ? 'not-allowed' : 'pointer',
                        padding: '4px 8px', fontSize: '10px',
                        textTransform: 'uppercase', letterSpacing: '0.06em',
                      }}
                    >
                      [ {'×'} ] UNPIN
                    </button>
                  </div>
                )}
                {pin.prompt && (
                  <div style={{
                    padding: '8px 10px', borderTop: '1px solid #eee',
                    fontSize: '10px', letterSpacing: '0.04em',
                    textTransform: 'uppercase', color: '#999',
                    lineHeight: 1.4,
                    display: '-webkit-box', WebkitLineClamp: 2,
                    WebkitBoxOrient: 'vertical', overflow: 'hidden',
                  }}>
                    {pin.prompt}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Full-size viewer */}
      {lightboxIdx != null && (
        <Lightbox
          items={shownPins.map(p => ({ url: p.url, label: p.prompt || '' }))}
          index={Math.min(lightboxIdx, shownPins.length - 1)}
          onNavigate={setLightboxIdx}
          onClose={() => setLightboxIdx(null)}
        />
      )}
    </div>
  )
}
