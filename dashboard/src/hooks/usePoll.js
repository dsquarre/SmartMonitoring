import { useEffect, useRef, useState } from 'react'

// Polls fetchFn every intervalMs. Keeps the last good `data` on screen even
// if a single poll fails, and only flips `error` when the *current* fetch
// attempt fails (so the UI can immediately show the server as unreachable).
export function usePoll(fetchFn, intervalMs) {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const savedFetch = useRef(fetchFn)
  savedFetch.current = fetchFn

  useEffect(() => {
    let mounted = true
    let timer = null

    async function tick() {
      try {
        const result = await savedFetch.current()
        if (mounted) {
          setData(result)
          setError(null)
        }
      } catch (e) {
        if (mounted) setError(e)
      } finally {
        if (mounted) timer = setTimeout(tick, intervalMs)
      }
    }

    tick()
    return () => {
      mounted = false
      if (timer) clearTimeout(timer)
    }
  }, [intervalMs])

  return { data, error }
}
