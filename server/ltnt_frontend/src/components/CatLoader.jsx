/**
 * CatLoader — pixel-art black cat + progress bar + French word cycling.
 *
 * Props:
 *   progress   number|null   0–100 for real progress, null for indeterminate
 *   statusText string        label shown beneath the bar
 */

import { useState, useEffect, useRef } from 'react'

// ── French vocabulary ─────────────────────────────────────────────────────────

const WORDS = [
  { fr: 'lumière',        en: 'light'                          },
  { fr: 'douceur',        en: 'softness'                       },
  { fr: 'éphémère',       en: 'ephemeral'                      },
  { fr: 'sérénité',       en: 'serenity'                       },
  { fr: 'flâner',         en: 'to stroll without purpose'      },
  { fr: 'dépaysement',    en: 'the feeling of being elsewhere'  },
  { fr: 'retrouvailles',  en: 'the joy of reunion'             },
  { fr: 'insouciance',    en: 'carefree lightness'             },
  { fr: 'frisson',        en: 'a sudden chill of excitement'   },
  { fr: 'rêverie',        en: 'daydream'                       },
  { fr: 'mélancolie',     en: 'melancholy'                     },
  { fr: 'fugace',         en: 'fleeting'                       },
  { fr: 'tendresse',      en: 'tenderness'                     },
  { fr: 'enchantement',   en: 'enchantment'                    },
  { fr: 'nostalgie',      en: 'nostalgia'                      },
  { fr: 'vertige',        en: 'vertigo'                        },
  { fr: 'féerie',         en: 'fairyland'                      },
  { fr: 'crépuscule',     en: 'twilight'                       },
  { fr: 'aurore',         en: 'dawn'                           },
  { fr: 'émerveillement', en: 'sense of wonder'                },
  { fr: 'langueur',       en: 'languid weariness'              },
  { fr: 'nonchalant',     en: 'casually unconcerned'           },
  { fr: 'quiétude',       en: 'quietude'                       },
  { fr: 'velours',        en: 'velvet'                         },
  { fr: 'délicatesse',    en: 'delicacy'                       },
  { fr: 'papillon',       en: 'butterfly'                      },
  { fr: 'mystère',        en: 'mystery'                        },
  { fr: 'lumineux',       en: 'radiant, glowing'               },
  { fr: 'nuit',           en: 'night'                          },
  { fr: 'ivresse',        en: 'intoxication, bliss'            },
]

// ── Pixel art cat frames ──────────────────────────────────────────────────────
// 16 cols × 14 rows, P=4px per cell → 64×56 SVG
// 0 = transparent, 1 = black (#111), 2 = white (eye pixel)
// Cat faces RIGHT: tail on left, head+ears on right

const P = 4

// Walk frame A — legs at cols 4 and 9
const WALK_A = [
  [0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0],
  [0,1,0,0,0,0,0,0,0,1,1,1,1,1,0,0],
  [0,1,1,0,0,0,0,0,0,1,2,1,2,1,0,0],
  [0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
  [0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0],
  [0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0],
  [0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0],
  [0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0],
  [0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0],
  [0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0],
  [0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
]

// Walk frame B — legs at cols 5 and 8 (trot alternation)
const WALK_B = [
  [0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0],
  [0,1,0,0,0,0,0,0,0,1,1,1,1,1,0,0],
  [0,1,1,0,0,0,0,0,0,1,2,1,2,1,0,0],
  [0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
  [0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0],
  [0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0],
  [0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0],
  [0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0],
  [0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0],
  [0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0],
  [0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
]

// Loaf — sitting, paws tucked, tail wrapping left side of body
const LOAF = [
  [0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0],
  [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0],
  [0,0,0,0,0,0,0,0,0,0,1,2,1,2,1,0],
  [0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0],
  [0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0],
  [0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0],
  [0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
  [0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0],
  [0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0],
  [0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
]

// ── Pixel cat renderer ────────────────────────────────────────────────────────

const COLS = 16
const ROWS = 14
const CAT_W = COLS * P  // 64px
const CAT_H = ROWS * P  // 56px

// Pre-flatten all cells so React always keeps the same DOM nodes (no mount/unmount flicker)
function PixelCat({ grid }) {
  const cells = []
  for (let y = 0; y < ROWS; y++) {
    for (let x = 0; x < COLS; x++) {
      const v = grid[y][x]
      cells.push(
        <rect
          key={`${x}-${y}`}
          x={x * P} y={y * P}
          width={P} height={P}
          fill={v === 2 ? 'var(--bg)' : v === 1 ? 'var(--text)' : 'none'}
        />
      )
    }
  }
  return (
    <svg
      width={CAT_W} height={CAT_H}
      style={{ display: 'block', imageRendering: 'pixelated' }}
    >
      {cells}
    </svg>
  )
}

// ── Pose sequence ─────────────────────────────────────────────────────────────

const POSES = [
  { grid: WALK_A, dur: 140 },
  { grid: WALK_B, dur: 140 },
  { grid: WALK_A, dur: 140 },
  { grid: WALK_B, dur: 140 },
  { grid: WALK_A, dur: 140 },
  { grid: WALK_B, dur: 140 },
  { grid: WALK_A, dur: 140 },
  { grid: WALK_B, dur: 140 },
  { grid: LOAF,   dur: 900 },
]

// ── Main component ────────────────────────────────────────────────────────────

const BAR_W  = 340
const TRAVEL = BAR_W - CAT_W   // 276px — how far the cat can slide

export default function CatLoader({ progress = null, statusText = '' }) {
  const indeterminate = progress === null

  // ── Pose cycling ─────────────────────────────────────────────────────────
  const [poseIdx, setPoseIdx] = useState(0)
  const poseTimer = useRef(null)

  useEffect(() => {
    function advance() {
      setPoseIdx(i => {
        const next = (i + 1) % POSES.length
        poseTimer.current = setTimeout(advance, POSES[next].dur)
        return next
      })
    }
    poseTimer.current = setTimeout(advance, POSES[0].dur)
    return () => clearTimeout(poseTimer.current)
  }, [])

  const { grid } = POSES[poseIdx]

  // ── Indeterminate bounce — rAF-driven, no CSS animation ──────────────────
  // Updates the cat div's transform directly to avoid React re-renders at 60fps
  // and to get an instant (no-squish) flip at each boundary.
  const catDivRef  = useRef(null)
  const bounceRef  = useRef({ x: 0, dir: 1, lastTime: null, raf: null })

  useEffect(() => {
    if (!indeterminate) return
    const state = bounceRef.current
    state.x        = 0
    state.dir      = 1
    state.lastTime = null

    const SPEED = 90 // px/s

    function tick(now) {
      if (state.lastTime !== null) {
        const dt = Math.min((now - state.lastTime) / 1000, 0.05)
        state.x += state.dir * SPEED * dt
        if (state.x >= TRAVEL) { state.x = TRAVEL; state.dir = -1 }
        if (state.x <= 0)      { state.x = 0;      state.dir =  1 }
      }
      state.lastTime = now

      if (catDivRef.current) {
        // scaleX(dir) flips the sprite instantly when direction reverses
        catDivRef.current.style.transform =
          `translateX(${state.x}px) scaleX(${state.dir})`
      }
      state.raf = requestAnimationFrame(tick)
    }

    state.raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(state.raf)
  }, [indeterminate])

  // ── French word cycling ───────────────────────────────────────────────────
  const [wordIdx,     setWordIdx]     = useState(() => Math.floor(Math.random() * WORDS.length))
  const [wordVisible, setWordVisible] = useState(true)

  useEffect(() => {
    let t
    function scheduleNext() {
      const delay = 10_000 + Math.random() * 20_000
      t = setTimeout(() => {
        setWordVisible(false)
        setTimeout(() => {
          setWordIdx(i => (i + 1) % WORDS.length)
          setWordVisible(true)
          scheduleNext()
        }, 500)
      }, delay)
    }
    scheduleNext()
    return () => clearTimeout(t)
  }, [])

  const word = WORDS[wordIdx]

  // ── Progress-mode cat position (JS calc, transform-based) ────────────────
  const catX = indeterminate
    ? 0
    : Math.round(Math.max(0, Math.min((progress / 100) * TRAVEL, TRAVEL)))

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div style={{ width: `${BAR_W}px`, textAlign: 'center' }}>

      {/* Cat stage */}
      <div style={{ position: 'relative', height: `${CAT_H + 4}px`, marginBottom: '4px' }}>

        {indeterminate ? (
          /* rAF-driven bounce — ref updated imperatively */
          <div
            ref={catDivRef}
            style={{
              position: 'absolute',
              bottom: 0,
              left: 0,
              transformOrigin: 'center bottom',
              willChange: 'transform',
            }}
          >
            <PixelCat grid={grid} />
          </div>
        ) : (
          /* Progress-based — CSS transition on transform */
          <div style={{
            position: 'absolute',
            bottom: 0,
            left: 0,
            transform: `translateX(${catX}px)`,
            transition: 'transform 0.8s cubic-bezier(0.25, 0.46, 0.45, 0.94)',
            willChange: 'transform',
          }}>
            <PixelCat grid={grid} />
          </div>
        )}
      </div>

      {/* Progress bar */}
      <div style={{
        position: 'relative',
        height: '2px',
        background: 'var(--border)',
        borderRadius: '1px',
        overflow: 'hidden',
      }}>
        {indeterminate ? (
          <div style={{
            position: 'absolute', inset: 0,
            background:
              'linear-gradient(90deg, transparent 0%, #333 35%, #333 65%, transparent 100%)',
            backgroundSize: '200% 100%',
            animation: 'catBarShimmer 1.8s linear infinite',
          }} />
        ) : (
          <div style={{
            position: 'absolute', top: 0, left: 0, bottom: 0,
            width: `${progress}%`,
            background: 'var(--accent)',
            transition: 'width 0.8s cubic-bezier(0.25, 0.46, 0.45, 0.94)',
            borderRadius: '1px',
          }} />
        )}
      </div>

      {/* Status text + % */}
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        marginTop: '10px',
        fontSize: '10px',
        letterSpacing: '0.08em',
        textTransform: 'uppercase',
        color: 'var(--muted)',
      }}>
        <span>{statusText}</span>
        {!indeterminate && <span>{Math.round(progress)}%</span>}
      </div>

      {/* French word */}
      <div style={{
        marginTop: '18px',
        opacity: wordVisible ? 1 : 0,
        transition: 'opacity 0.5s ease',
        fontSize: '11px',
        letterSpacing: '0.06em',
        color: 'var(--muted)',
        fontStyle: 'italic',
      }}>
        <span style={{ color: 'var(--muted)', fontStyle: 'normal', letterSpacing: '0.04em' }}>
          {word.fr}
        </span>
        {' — '}
        <span>{word.en}</span>
      </div>

      <style>{`
        @keyframes catBarShimmer {
          from { background-position: -200% 0; }
          to   { background-position:  200% 0; }
        }
      `}</style>
    </div>
  )
}
