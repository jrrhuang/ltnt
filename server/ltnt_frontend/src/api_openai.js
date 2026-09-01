/**
 * OpenAI image-edit provider client. Isolated from api.js so it's easy to
 * disconnect or delete — just remove this file and the EditViewOpenAI
 * imports.
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
 * Rasterise an image src to a canvas and return its raw base64 (no data-URL
 * prefix). Ensures PNG output regardless of input type.
 */
export async function imageSrcToB64(src) {
  const img = await loadImg(src)
  const canvas = document.createElement('canvas')
  canvas.width = img.naturalWidth
  canvas.height = img.naturalHeight
  canvas.getContext('2d').drawImage(img, 0, 0)
  return canvas.toDataURL('image/png').split(',')[1]
}

/**
 * Build a mask PNG from the caller's selection shape(s). OpenAI wants
 * TRANSPARENT pixels = edit area, OPAQUE = keep. Returns { width, height,
 * b64 } — raw base64 with no data-URL prefix. Returns null if no shapes.
 *
 * shapes: Array of either
 *   { type: 'rect', region: { x, y, w, h } }   // normalized 0..1
 *   { type: 'lasso', polygons: [[{x,y}, ...], ...] }  // normalized 0..1
 */
export async function buildMaskB64(width, height, shapes) {
  if (!shapes || shapes.length === 0) return null
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  const ctx = canvas.getContext('2d')

  // Fill fully opaque black first — everything is "keep" by default.
  ctx.fillStyle = 'rgba(0, 0, 0, 1)'
  ctx.fillRect(0, 0, width, height)

  // Erase the selected regions so they become transparent = "edit here".
  ctx.globalCompositeOperation = 'destination-out'
  for (const shape of shapes) {
    if (shape.type === 'rect' && shape.region) {
      ctx.fillRect(
        shape.region.x * width, shape.region.y * height,
        shape.region.w * width, shape.region.h * height,
      )
    } else if (shape.type === 'lasso' && Array.isArray(shape.polygons)) {
      for (const pts of shape.polygons) {
        if (!pts || pts.length < 3) continue
        ctx.beginPath()
        ctx.moveTo(pts[0].x * width, pts[0].y * height)
        for (let k = 1; k < pts.length; k++) ctx.lineTo(pts[k].x * width, pts[k].y * height)
        ctx.closePath()
        ctx.fill()
      }
    }
  }
  ctx.globalCompositeOperation = 'source-over'

  return canvas.toDataURL('image/png').split(',')[1]
}

/**
 * Call the server's OpenAI edit proxy.
 *
 * @param {string} imageSrc   Source URL or data-URL of the image to edit.
 * @param {string} prompt     Edit instruction.
 * @param {object} options    { shapes?, n?, size?, quality? }
 * @returns {Promise<{ images_b64: string[], model?: string }>}
 */
export async function editImageOpenAI(imageSrc, prompt, options = {}) {
  const { shapes = null, n = 1, size = 'auto', quality = 'auto', model } = options

  // Prepare the image.
  const img = await loadImg(imageSrc)
  const canvas = document.createElement('canvas')
  canvas.width = img.naturalWidth
  canvas.height = img.naturalHeight
  canvas.getContext('2d').drawImage(img, 0, 0)
  const image_b64 = canvas.toDataURL('image/png').split(',')[1]

  // Prepare the mask (optional).
  let mask_b64 = null
  if (shapes && shapes.length > 0) {
    mask_b64 = await buildMaskB64(canvas.width, canvas.height, shapes)
  }

  const body = { image_b64, prompt, mask_b64, n, size, quality }
  if (model) body.model = model

  const res = await fetch('/api/openai/edit', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || `OpenAI edit failed: ${res.statusText}`)
  }
  return res.json()
}
