// CREATE MAP — isolated API. Deleting this directory removes the feature.
const BASE = '/api/create_map'

export async function startCreateMap(prompt, params = {}) {
  const res = await fetch(`${BASE}/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt, ...params }),
  })
  if (!res.ok) throw new Error(`Create Map failed: ${res.statusText}`)
  const data = await res.json()
  return data.job_id
}

export async function getCreateMapStatus(jobId) {
  const res = await fetch(`${BASE}/${jobId}`)
  if (!res.ok) throw new Error(`Poll failed: ${res.statusText}`)
  return res.json()
}

export function pollCreateMap(jobId, onProgress, intervalMs = 1500) {
  return new Promise((resolve, reject) => {
    let stopped = false
    const tick = async () => {
      if (stopped) return
      try {
        const data = await getCreateMapStatus(jobId)
        onProgress?.(data)
        if (data.status === 'done') return resolve(data)
        if (data.status === 'error') return reject(new Error(data.error || 'Error'))
        setTimeout(tick, intervalMs)
      } catch (e) {
        reject(e)
      }
    }
    tick()
    return () => { stopped = true }
  })
}
