export async function fetchJSON(url) {
  const res = await fetch(url, { cache: 'no-store' })
  if (!res.ok) throw new Error(`Request failed: ${res.status}`)
  return res.json()
}

export const getStatus = () => fetchJSON('/api/status')
export const getClients = () => fetchJSON('/api/clients')
export const getHistory = () => fetchJSON('/api/metrics/history')
