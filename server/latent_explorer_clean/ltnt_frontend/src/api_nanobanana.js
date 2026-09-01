/**
 * Google Gemini / Nano Banana image-edit client. Same shape as api_openai.js
 * — returns { images_b64 } — but Nano Banana doesn't accept masks, so we
 * always send the full image and leave any rect/lasso clipping to the
 * compositor on the client side.
 */

async function loadImg(src) {
  return new Promise((resolve, reject) => {
    if (!src) return reject(new Error('loadImg: src is undefined'))
    const img = new Image()
    if (!src.startsWith('data:')) img.crossOrigin = 'anonymous'
    img.onload = () => resolve(img)
    img.onerror = () => reject(new Error(`Failed to load image: ${src.slice(0, 80)}`))
    img.src = src
  })
}

/**
 * Call the server's Nano Banana edit proxy.
 *
 * @param {string} imageSrc
 * @param {string} prompt
 * @param {object} options  { n?, model?, region?: {x,y,w,h} normalized 0..1 }
 *                          If `region` is given, the server crops to it
 *                          before sending to Gemini, and the frontend
 *                          should paste the returned crop back at the
 *                          same coordinates.
 * @returns {Promise<{ images_b64: string[], model?: string }>}
 */
export async function editImageNanoBanana(imageSrc, prompt, options = {}) {
  const { n = 1, model, region = null } = options

  const img = await loadImg(imageSrc)
  const canvas = document.createElement('canvas')
  canvas.width = img.naturalWidth
  canvas.height = img.naturalHeight
  canvas.getContext('2d').drawImage(img, 0, 0)
  const image_b64 = canvas.toDataURL('image/png').split(',')[1]

  const body = { image_b64, prompt, n }
  if (model) body.model = model
  if (region) body.region = region

  const res = await fetch('/api/nanobanana/edit', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || `Nano Banana edit failed: ${res.statusText}`)
  }
  return res.json()
}
