/**
 * Cat meow synthesizer using Web Audio API.
 * Generates 20 distinct meow variations — no audio files needed.
 *
 * Each meow is a frequency-swept oscillator shaped through a vocal filter,
 * with a short gain envelope and optional vibrato.
 */

// 20 meow profiles: [startHz, peakHz, endHz, durationMs, vibratoRate, vibratoDepth, filterQ]
const MEOW_PROFILES = [
  [380, 780, 420,  340, 5.2, 18, 6],   // classic short meow
  [420, 900, 380,  480, 4.8, 22, 7],   // questioning meow ↑
  [500, 750, 300,  260, 6.0, 14, 5],   // quick chirp
  [340, 860, 500,  560, 3.8, 26, 8],   // long drawn meow
  [460, 680, 460,  300, 7.2, 10, 5],   // flat little mew
  [380, 1020,340,  520, 4.2, 30, 9],   // high excited meow
  [280, 620, 280,  400, 5.5, 16, 6],   // low grumbly meow
  [440, 840, 560,  380, 4.0, 20, 7],   // upward lilt
  [600, 920, 400,  290, 8.0, 12, 5],   // tiny high squeak
  [360, 740, 260,  440, 3.5, 24, 8],   // sleepy meow
  [480, 800, 480,  200, 9.0,  8, 4],   // very quick mew
  [320, 1100,380,  600, 3.2, 32, 10],  // big dramatic meow
  [420, 720, 380,  360, 6.4, 18, 6],   // normal everyday meow
  [500, 880, 440,  450, 5.0, 22, 7],   // eager meow
  [260, 580, 320,  520, 4.6, 20, 7],   // low rumble meow
  [440, 960, 320,  400, 5.8, 28, 9],   // chirpy high meow
  [380, 700, 500,  280, 7.0, 14, 5],   // short inquisitive
  [320, 840, 280,  640, 3.0, 36, 10],  // long sad meow
  [560, 860, 560,  240, 8.4, 10, 4],   // tiny satisfied mew
  [400, 780, 400,  360, 5.6, 18, 6],   // classic balanced meow
]

let audioCtx = null

function getCtx() {
  if (!audioCtx || audioCtx.state === 'closed') {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)()
  }
  if (audioCtx.state === 'suspended') {
    audioCtx.resume()
  }
  return audioCtx
}

/**
 * Play a single meow sound.
 * @param {number} [variant] - 0-19, or random if omitted
 */
export function playMeow(variant) {
  try {
    const ctx = getCtx()
    const idx = variant !== undefined
      ? (variant % MEOW_PROFILES.length)
      : Math.floor(Math.random() * MEOW_PROFILES.length)

    const [startHz, peakHz, endHz, durMs, vibratoRate, vibratoDepth, filterQ] = MEOW_PROFILES[idx]
    const dur = durMs / 1000

    const now = ctx.currentTime

    // ── Oscillator (carrier) ─────────────────────────────────────────────
    const osc = ctx.createOscillator()
    osc.type = 'sawtooth'

    // Frequency sweep: start → peak (60% through) → end
    osc.frequency.setValueAtTime(startHz, now)
    osc.frequency.linearRampToValueAtTime(peakHz, now + dur * 0.55)
    osc.frequency.linearRampToValueAtTime(endHz,  now + dur)

    // ── Vibrato (LFO) ────────────────────────────────────────────────────
    const lfo = ctx.createOscillator()
    const lfoGain = ctx.createGain()
    lfo.frequency.value = vibratoRate
    lfoGain.gain.setValueAtTime(0, now)
    lfoGain.gain.linearRampToValueAtTime(vibratoDepth, now + dur * 0.3)
    lfoGain.gain.linearRampToValueAtTime(vibratoDepth * 0.5, now + dur)
    lfo.connect(lfoGain)
    lfoGain.connect(osc.frequency)
    lfo.start(now)
    lfo.stop(now + dur + 0.05)

    // ── Vocal filter (bandpass) ──────────────────────────────────────────
    const filter = ctx.createBiquadFilter()
    filter.type = 'bandpass'
    filter.frequency.setValueAtTime(startHz * 1.4, now)
    filter.frequency.linearRampToValueAtTime(peakHz * 1.3, now + dur * 0.5)
    filter.frequency.linearRampToValueAtTime(endHz * 1.4, now + dur)
    filter.Q.value = filterQ

    // Second harmonic overtone for richness
    const osc2 = ctx.createOscillator()
    osc2.type = 'sine'
    osc2.frequency.setValueAtTime(startHz * 2, now)
    osc2.frequency.linearRampToValueAtTime(peakHz * 2, now + dur * 0.55)
    osc2.frequency.linearRampToValueAtTime(endHz * 2,  now + dur)

    const osc2Gain = ctx.createGain()
    osc2Gain.gain.value = 0.25
    osc2.connect(osc2Gain)

    // ── Master gain envelope ─────────────────────────────────────────────
    const gainNode = ctx.createGain()
    gainNode.gain.setValueAtTime(0, now)
    gainNode.gain.linearRampToValueAtTime(0.18, now + dur * 0.08)   // attack
    gainNode.gain.setValueAtTime(0.18, now + dur * 0.75)             // sustain
    gainNode.gain.linearRampToValueAtTime(0, now + dur)              // release

    // ── Connect graph ────────────────────────────────────────────────────
    osc.connect(filter)
    osc2Gain.connect(filter)
    filter.connect(gainNode)
    gainNode.connect(ctx.destination)

    osc.start(now)
    osc.stop(now + dur + 0.05)
    osc2.start(now)
    osc2.stop(now + dur + 0.05)

  } catch (e) {
    // Audio not available (e.g. SSR, user blocked audio) — silently ignore
  }
}

/**
 * Play a random meow.
 */
export function playRandomMeow() {
  playMeow()
}
