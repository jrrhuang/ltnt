import { useEffect, useState, useCallback, useRef } from 'react'

/**
 * Lightbox — full-size overlay viewer for canvas tiles / board pins.
 *
 * items: [{ url, refinedUrl?, label? }]
 * index: which item is open
 * onNavigate(nextIndex): arrow-key / button / click-through navigation
 * onClose(): ESC / backdrop / [x]
 *
 * Waits are visible, never dead: a skeleton + "LOADING FULL SIZE" shows until
 * the image's onLoad fires, and neighbors (plus refined variants) are
 * preloaded so browsing is instant after the first view.
 *
 * When an item has a refinedUrl, both images are stacked and crossfaded so
 * the SKETCH ⇄ REFINED delta is FELT: bold active pill, animated crossfade,
 * and a HOLD-TO-COMPARE button (press = sketch, release = refined).
 */
export default function Lightbox({ items = [], index = 0, onNavigate = () => {}, onClose = () => {},
                                   // Optional: request a full-quality render of the
                                   // current item — (index) => void. Renders a
                                   // "RENDER FULL QUALITY" button when the item has
                                   // no refinedUrl yet (deliberate refine affordance
                                   // for dense canvases whose tiles are too small
                                   // for per-tile buttons). refiningIndex marks the
                                   // in-flight one.
                                   onRefine = null, refiningIndex = null }) {
  const item = items[index]
  // 'refined' | 'sketch' — only meaningful when item.refinedUrl exists.
  const [variant, setVariant] = useState('refined')
  // Track which srcs have finished loading (skeleton until then).
  const [loadedSrcs, setLoadedSrcs] = useState(() => new Set())
  const markLoaded = useCallback((src) => {
    setLoadedSrcs(prev => {
      if (prev.has(src)) return prev
      const next = new Set(prev); next.add(src); return next
    })
  }, [])
  useEffect(() => { setVariant('refined') }, [index])

  const hasRefined = !!(item && item.refinedUrl)
  const baseSrc = item ? item.url : null
  const shownSrc = hasRefined && variant === 'refined' ? item.refinedUrl : baseSrc
  const shownLoaded = shownSrc != null && loadedSrcs.has(shownSrc)

  const go = useCallback((delta) => {
    if (items.length < 2) return
    onNavigate((index + delta + items.length) % items.length)
  }, [items.length, index, onNavigate])

  // Preload neighbors (and every variant of the current + neighbor items) so
  // arrow-key browsing swaps instantly instead of counter-leading-the-image.
  const preloadCache = useRef({})
  useEffect(() => {
    const wanted = []
    for (const off of [0, 1, -1]) {
      const it = items[(index + off + items.length) % items.length]
      if (!it) continue
      wanted.push(it.url)
      if (it.refinedUrl) wanted.push(it.refinedUrl)
    }
    for (const src of wanted) {
      if (!src || preloadCache.current[src]) continue
      const im = new Image()
      im.onload = () => markLoaded(src)
      im.src = src
      preloadCache.current[src] = im
    }
  }, [items, index, markLoaded])

  useEffect(() => {
    function onKey(e) {
      if (e.key === 'Escape') { e.stopPropagation(); onClose() }
      else if (e.key === 'ArrowRight') { e.preventDefault(); go(1) }
      else if (e.key === 'ArrowLeft') { e.preventDefault(); go(-1) }
    }
    window.addEventListener('keydown', onKey, true)
    return () => window.removeEventListener('keydown', onKey, true)
  }, [go, onClose])

  if (!item) return null

  return (
    <div
      onClick={onClose}
      style={{
        position: 'fixed', inset: 0, zIndex: 200,
        background: 'rgba(20,20,20,0.88)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        cursor: 'zoom-out',
      }}
    >
      {/* Image stage (click-through navigation: clicking advances) */}
      <div
        onClick={(e) => { e.stopPropagation(); go(1) }}
        style={{
          position: 'relative',
          maxWidth: 'calc(100vw - 160px)', maxHeight: 'calc(100vh - 120px)',
          cursor: items.length > 1 ? 'pointer' : 'default',
        }}
      >
        {/* Base image (sketch / plain preview) defines the layout box. */}
        <img
          src={baseSrc}
          alt={item.label || ''}
          onLoad={() => markLoaded(baseSrc)}
          onError={() => markLoaded(baseSrc)}
          style={{
            maxWidth: 'calc(100vw - 160px)', maxHeight: 'calc(100vh - 120px)',
            objectFit: 'contain', display: 'block',
            boxShadow: '0 8px 48px rgba(0,0,0,0.6)',
            opacity: shownLoaded ? 1 : 0.15,
            transition: 'opacity 0.2s',
          }}
          draggable={false}
        />
        {/* Refined overlay — crossfades in/out so the A/B delta is FELT. */}
        {hasRefined && (
          <img
            src={item.refinedUrl}
            alt=""
            onLoad={() => markLoaded(item.refinedUrl)}
            onError={() => markLoaded(item.refinedUrl)}
            style={{
              position: 'absolute', inset: 0,
              width: '100%', height: '100%',
              objectFit: 'contain', display: 'block',
              opacity: variant === 'refined' && loadedSrcs.has(item.refinedUrl) ? 1 : 0,
              transition: 'opacity 0.28s ease',
              pointerEvents: 'none',
            }}
            draggable={false}
          />
        )}
        {/* Load skeleton — the fetch of a full-size image is a visible wait. */}
        {!shownLoaded && (
          <div style={{
            position: 'absolute', inset: 0,
            display: 'flex', flexDirection: 'column',
            alignItems: 'center', justifyContent: 'center', gap: '10px',
            minWidth: '340px', minHeight: '280px',
            color: 'rgba(255,255,255,0.75)',
            fontSize: '11px', letterSpacing: '0.1em', textTransform: 'uppercase',
          }}>
            <div className="ltnt-refine-shimmer" style={{ position: 'absolute', inset: 0 }} />
            <div style={{
              width: '22px', height: '22px', borderRadius: '50%',
              border: '2px solid rgba(255,255,255,0.25)',
              borderTopColor: 'rgba(255,255,255,0.9)',
              animation: 'ltnt-spin 0.8s linear infinite',
            }} />
            <div>LOADING FULL SIZE...</div>
          </div>
        )}
        {/* Variant badge — always know which version you're looking at. */}
        {hasRefined && shownLoaded && (
          <div style={{
            position: 'absolute', top: '10px', left: '10px',
            background: variant === 'refined' ? '#1a1a1a' : 'rgba(255,251,232,0.95)',
            color: variant === 'refined' ? '#fff' : '#7a6a2f',
            border: variant === 'refined' ? '1px solid #000' : '1px solid #e8dcae',
            padding: '3px 8px', fontSize: '10px', fontWeight: 700,
            letterSpacing: '0.1em', textTransform: 'uppercase',
            pointerEvents: 'none',
          }}>
            {variant === 'refined' ? '✦ REFINED' : '✎ SKETCH'}
          </div>
        )}
      </div>

      {/* Close */}
      <button
        onClick={(e) => { e.stopPropagation(); onClose() }}
        title="Close (Esc)"
        style={{
          position: 'absolute', top: '16px', right: '20px',
          border: '1px solid rgba(255,255,255,0.4)', background: 'rgba(0,0,0,0.4)',
          color: '#fff', padding: '8px 14px', fontSize: '12px',
          letterSpacing: '0.08em', cursor: 'pointer',
        }}
      >
        [ × ] CLOSE
      </button>

      {/* Prev / Next */}
      {items.length > 1 && (
        <>
          <button
            onClick={(e) => { e.stopPropagation(); go(-1) }}
            title="Previous (←)"
            style={navBtnStyle('left')}
          >
            ‹
          </button>
          <button
            onClick={(e) => { e.stopPropagation(); go(1) }}
            title="Next (→)"
            style={navBtnStyle('right')}
          >
            ›
          </button>
        </>
      )}

      {/* Footer: counter + label + A/B controls */}
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          position: 'absolute', bottom: '18px', left: '50%',
          transform: 'translateX(-50%)',
          display: 'flex', alignItems: 'center', gap: '14px',
          color: 'rgba(255,255,255,0.8)', fontSize: '11px',
          letterSpacing: '0.08em', textTransform: 'uppercase',
          background: 'rgba(0,0,0,0.45)', padding: '8px 16px',
          cursor: 'default',
        }}
      >
        <span>{index + 1} / {items.length}</span>
        {item.label && <span style={{ color: 'rgba(255,255,255,0.55)' }}>{item.label}</span>}
        {onRefine && !hasRefined && (
          <button
            onClick={() => onRefine(index)}
            disabled={refiningIndex === index}
            title="Render this image at full quality"
            style={{
              border: '1px solid rgba(255,255,255,0.45)',
              background: 'transparent',
              color: 'rgba(255,255,255,0.85)',
              padding: '3px 10px', fontSize: '10px',
              letterSpacing: '0.06em', textTransform: 'uppercase',
              cursor: refiningIndex === index ? 'wait' : 'pointer',
            }}
          >
            {refiningIndex === index ? '⟳ REFINING…' : '✦ RENDER FULL QUALITY'}
          </button>
        )}
        {hasRefined && (
          <span style={{ display: 'flex', gap: '4px', alignItems: 'center' }}>
            {['sketch', 'refined'].map(v => {
              const active = variant === v
              return (
                <button
                  key={v}
                  onClick={() => setVariant(v)}
                  style={{
                    border: active ? '2px solid #fff' : '1px solid rgba(255,255,255,0.35)',
                    background: active ? '#fff' : 'transparent',
                    color: active ? '#111' : 'rgba(255,255,255,0.55)',
                    fontWeight: active ? 700 : 400,
                    padding: active ? '4px 12px' : '3px 10px',
                    fontSize: '10px',
                    letterSpacing: '0.06em', textTransform: 'uppercase',
                    cursor: 'pointer',
                  }}
                >
                  {active ? '● ' : ''}{v === 'sketch' ? '✎ SKETCH' : '✦ REFINED'}
                </button>
              )
            })}
            {/* Press-and-hold A/B flick: press = sketch, release = refined. */}
            <button
              onPointerDown={(e) => { e.preventDefault(); setVariant('sketch') }}
              onPointerUp={() => setVariant('refined')}
              onPointerLeave={() => setVariant('refined')}
              title="Press and hold to see the sketch; release for the refined image"
              style={{
                border: '1px dashed rgba(255,255,255,0.45)',
                background: 'transparent',
                color: 'rgba(255,255,255,0.75)',
                padding: '3px 10px', fontSize: '10px',
                letterSpacing: '0.06em', textTransform: 'uppercase',
                cursor: 'pointer', userSelect: 'none',
              }}
            >
              ⇄ HOLD TO COMPARE
            </button>
          </span>
        )}
        {items.length > 1 && (
          <span style={{ color: 'rgba(255,255,255,0.4)', fontSize: '10px' }}>
            ← → TO BROWSE · ESC TO CLOSE
          </span>
        )}
      </div>
    </div>
  )
}

function navBtnStyle(side) {
  return {
    position: 'absolute', [side]: '18px', top: '50%',
    transform: 'translateY(-50%)',
    border: '1px solid rgba(255,255,255,0.4)', background: 'rgba(0,0,0,0.4)',
    color: '#fff', width: '48px', height: '64px',
    fontSize: '26px', lineHeight: 1, cursor: 'pointer',
  }
}
