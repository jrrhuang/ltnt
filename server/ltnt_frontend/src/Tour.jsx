import { useEffect, useLayoutEffect, useState } from 'react'

/**
 * Guided-tour overlay.
 *
 * Props:
 *   steps: Array<{
 *     target?: string,       // CSS selector for the element to highlight
 *     title: string,
 *     body: string | JSX,
 *     placement?: 'top' | 'bottom' | 'left' | 'right' | 'center',
 *   }>
 *   open: boolean
 *   onClose: () => void
 *   storageKey?: string      // localStorage key; marks tour as seen on close
 *
 * The overlay spotlights the target element via a CSS clip-path cutout and
 * parks a tooltip card next to it. Keyboard: Esc closes, ← / → navigate.
 */
export default function Tour({ steps, open, onClose, storageKey }) {
  const [idx, setIdx] = useState(0)
  const [rect, setRect] = useState(null)
  const [viewport, setViewport] = useState({ w: window.innerWidth, h: window.innerHeight })

  // Reset to first step whenever the tour is opened.
  useEffect(() => { if (open) setIdx(0) }, [open])

  // Track the highlighted element's rect.
  useLayoutEffect(() => {
    if (!open) return
    const step = steps[idx]
    const update = () => {
      setViewport({ w: window.innerWidth, h: window.innerHeight })
      if (!step?.target) { setRect(null); return }
      const el = document.querySelector(step.target)
      if (!el) { setRect(null); return }
      setRect(el.getBoundingClientRect())
    }
    update()
    // Re-measure on every frame for a few hundred ms so we catch layout shifts,
    // lazy mounts, etc. Also re-measure on resize/scroll.
    const interval = setInterval(update, 120)
    window.addEventListener('resize', update)
    window.addEventListener('scroll', update, true)
    return () => {
      clearInterval(interval)
      window.removeEventListener('resize', update)
      window.removeEventListener('scroll', update, true)
    }
  }, [open, idx, steps])

  // Keyboard navigation.
  useEffect(() => {
    if (!open) return
    const onKey = (e) => {
      if (e.key === 'Escape') close()
      else if (e.key === 'ArrowRight') next()
      else if (e.key === 'ArrowLeft') prev()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, idx, steps])

  if (!open || !steps || steps.length === 0) return null
  const step = steps[idx]

  function close() {
    if (storageKey) {
      try { localStorage.setItem(storageKey, '1') } catch {}
    }
    onClose?.()
  }
  function next() { idx < steps.length - 1 ? setIdx(idx + 1) : close() }
  function prev() { if (idx > 0) setIdx(idx - 1) }

  // Spotlight clip-path: full rect with a hole cut around the target.
  const pad = 8
  const hole = rect ? {
    left:   Math.max(0, rect.left - pad),
    top:    Math.max(0, rect.top - pad),
    right:  Math.min(viewport.w, rect.right + pad),
    bottom: Math.min(viewport.h, rect.bottom + pad),
  } : null

  const clipPath = hole
    ? `polygon(
        0 0, 100% 0, 100% 100%, 0 100%, 0 0,
        ${hole.left}px ${hole.top}px,
        ${hole.left}px ${hole.bottom}px,
        ${hole.right}px ${hole.bottom}px,
        ${hole.right}px ${hole.top}px,
        ${hole.left}px ${hole.top}px
      )`
    : undefined

  // Tooltip position: below the target by default, flip to above if no room.
  const placement = step.placement ?? (rect ? 'bottom' : 'center')
  const tipW = 340
  let tipX, tipY
  if (rect && placement !== 'center') {
    const margin = 16
    if (placement === 'bottom') {
      tipY = Math.min(hole.bottom + margin, viewport.h - 200)
      tipX = Math.min(Math.max(rect.left + rect.width / 2 - tipW / 2, 16), viewport.w - tipW - 16)
    } else if (placement === 'top') {
      tipY = Math.max(hole.top - margin - 170, 16)
      tipX = Math.min(Math.max(rect.left + rect.width / 2 - tipW / 2, 16), viewport.w - tipW - 16)
    } else if (placement === 'right') {
      tipX = Math.min(hole.right + margin, viewport.w - tipW - 16)
      tipY = Math.min(Math.max(rect.top + rect.height / 2 - 80, 16), viewport.h - 200)
    } else if (placement === 'left') {
      tipX = Math.max(hole.left - margin - tipW, 16)
      tipY = Math.min(Math.max(rect.top + rect.height / 2 - 80, 16), viewport.h - 200)
    }
  } else {
    tipX = viewport.w / 2 - tipW / 2
    tipY = viewport.h / 2 - 100
  }

  return (
    <>
      {/* Spotlight backdrop — pointerEvents NONE on purpose: the tour must
          never swallow clicks/typing meant for the app (prompt bar included),
          and a stray backdrop click must not kill the tour mid-read. Dismiss
          via Skip / Done / Esc only. */}
      <div style={{
        position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.45)',
        clipPath, WebkitClipPath: clipPath,
        zIndex: 9998, pointerEvents: 'none',
      }} />

      {/* Highlight ring around the target */}
      {rect && (
        <div style={{
          position: 'fixed',
          left: hole.left, top: hole.top,
          width:  hole.right - hole.left,
          height: hole.bottom - hole.top,
          border: '2px solid #1a1a1a',
          boxShadow: '0 0 0 2px rgba(255,255,255,0.7), 0 0 24px rgba(0,0,0,0.3)',
          pointerEvents: 'none',
          zIndex: 9999,
          borderRadius: 4,
          animation: 'tourPulse 1.4s ease-in-out infinite',
        }} />
      )}

      {/* Tooltip card */}
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          position: 'fixed',
          left: tipX, top: tipY,
          width: tipW,
          background: '#fff',
          border: '1px solid #1a1a1a',
          boxShadow: '0 8px 32px rgba(0,0,0,0.25)',
          padding: '18px 20px',
          zIndex: 10000,
          fontSize: '12px',
          lineHeight: 1.5,
          color: '#1a1a1a',
        }}
      >
        <div style={{
          fontSize: '10px', letterSpacing: '0.1em',
          textTransform: 'uppercase', color: '#999',
          marginBottom: 6,
        }}>
          Step {idx + 1} of {steps.length}
        </div>
        <div style={{
          fontSize: '14px', fontWeight: 700, marginBottom: 8,
          letterSpacing: '0.02em',
        }}>
          {step.title}
        </div>
        <div style={{ color: '#333', marginBottom: 16 }}>
          {step.body}
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 8 }}>
          <button onClick={close} style={btnStyle('ghost')}>Skip</button>
          <div style={{ display: 'flex', gap: 8 }}>
            {idx > 0 && (
              <button onClick={prev} style={btnStyle('ghost')}>← Back</button>
            )}
            <button onClick={next} style={btnStyle('primary')}>
              {idx === steps.length - 1 ? 'Done' : 'Next →'}
            </button>
          </div>
        </div>
      </div>

      <style>{`
        @keyframes tourPulse {
          0%, 100% { box-shadow: 0 0 0 2px rgba(255,255,255,0.7), 0 0 24px rgba(0,0,0,0.3); }
          50%      { box-shadow: 0 0 0 4px rgba(255,255,255,0.9), 0 0 32px rgba(0,0,0,0.45); }
        }
      `}</style>
    </>
  )
}

function btnStyle(variant) {
  const base = {
    border: '1px solid #1a1a1a',
    padding: '6px 14px',
    fontSize: '11px',
    letterSpacing: '0.06em',
    textTransform: 'uppercase',
    cursor: 'pointer',
    background: 'none',
  }
  if (variant === 'primary') {
    return { ...base, background: '#1a1a1a', color: '#fff' }
  }
  return { ...base, color: '#1a1a1a' }
}

/**
 * Hook: manages tour `open` state.
 *
 * Usage: `const tour = useTour('ltnt.tour.generate', { autoOpen: tourMode })`
 *
 * - When `autoOpen` becomes truthy, the tour opens (e.g. Tour Mode flipped
 *   ON, or the user navigated to a new view while Tour Mode is already ON).
 * - When `autoOpen` becomes falsy, the tour closes.
 * - The user can still manually open/close via `start` / `close` without
 *   changing `autoOpen` — closing mid-view leaves Tour Mode intact, and
 *   navigating to the next view will auto-open again.
 */
/**
 * Shared button used in every view's top bar. Toggles the global Tour Mode
 * flag (owned by App) on click; renders differently depending on state so
 * the user can see at a glance whether auto-tour is on.
 *
 * When ON  → solid dark background, label "⊙ TOUR ON".
 * When OFF → outlined,            label "? TOUR".
 */
export function TourToggle({ tourMode, onTourModeChange }) {
  return (
    <button
      onClick={() => onTourModeChange(!tourMode)}
      title={tourMode
        ? "Tour Mode is ON — auto-shows hints on every step. Click to turn off."
        : "Turn on Tour Mode to see guided hints automatically."}
      style={{
        border: '1px solid #1a1a1a',
        padding: '6px 12px',
        background: tourMode ? '#1a1a1a' : 'none',
        color: tourMode ? '#fff' : '#1a1a1a',
        cursor: 'pointer',
        fontSize: '11px',
        letterSpacing: '0.04em',
        textTransform: 'uppercase',
      }}
    >
      {tourMode ? '⊙ Tour ON' : '? Tour'}
    </button>
  )
}

export function useTour(storageKey, { autoOpen = false } = {}) {
  // DISMISS-ONCE: a tour the user has already closed (storageKey flag, set in
  // Tour.close()) never auto-opens again on mount — only an EXPLICIT Tour
  // toggle (an autoOpen false->true transition after mount) or start() can
  // reopen it.
  const seen = (() => {
    try { return !!storageKey && localStorage.getItem(storageKey) === '1' } catch { return false }
  })()
  const [open, setOpen] = useState(!!autoOpen && !seen)
  const mounted = useState(() => ({ current: false }))[0]

  useEffect(() => {
    if (!autoOpen) { setOpen(false); mounted.current = true; return }
    // First evaluation (mount): respect the seen flag. Later transitions to
    // true are user-driven (Tour toggle) — always honor them.
    if (!mounted.current && seen) { mounted.current = true; return }
    mounted.current = true
    setOpen(true)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autoOpen])

  return {
    open,
    start: () => setOpen(true),
    close: () => setOpen(false),
  }
}
