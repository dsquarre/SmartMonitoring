import { useEffect, useRef, useState } from 'react'
import { ChipIcon, LoaderIcon, CheckIcon, ArrowDownIcon, ArrowUpIcon } from './Icons.jsx'

function deriveState(client, aggregating, running) {
  if (client.evaluated) return 'evaluated'
  if (client.uploaded) return aggregating ? 'aggregating' : 'uploaded'
  if (client.selected) return 'training'
  if (running) return 'waiting'
  return 'idle'
}

const LABEL = {
  evaluated: 'Evaluated',
  aggregating: 'Uploaded · aggregating',
  uploaded: 'Uploaded',
  training: 'Training…',
  waiting: 'Waiting (not selected)',
  idle: 'Connected',
}

export default function ClientCard({ client, aggregating, running }) {
  const prev = useRef({ selected: false, uploaded: false })
  const [flashDown, setFlashDown] = useState(false)
  const [flashUp, setFlashUp] = useState(false)

  useEffect(() => {
    if (client.selected && !prev.current.selected) {
      setFlashDown(true)
      const t = setTimeout(() => setFlashDown(false), 1300)
      return () => clearTimeout(t)
    }
  }, [client.selected])

  useEffect(() => {
    if (client.uploaded && !prev.current.uploaded) {
      setFlashUp(true)
      const t = setTimeout(() => setFlashUp(false), 1300)
      return () => clearTimeout(t)
    }
  }, [client.uploaded])

  useEffect(() => {
    prev.current = { selected: client.selected, uploaded: client.uploaded }
  })

  const state = deriveState(client, aggregating, running)
  const cpuGHz = client.cpu_frequency ? (client.cpu_frequency / 1e9).toFixed(1) : null

  return (
    <div className="client-card">
      <div className="client-top">
        <ChipIcon className="client-chip" />
        <div className="client-id">{client.client_id}</div>
        {flashDown && <ArrowDownIcon className="flash flash-down" />}
        {flashUp && <ArrowUpIcon className="flash flash-up" />}
      </div>

      <span className={`badge state-${state}`}>
        {state === 'training' && <LoaderIcon className="spin" />}
        {state === 'aggregating' && <LoaderIcon className="spin" />}
        {(state === 'uploaded' || state === 'evaluated') && <CheckIcon />}
        {LABEL[state]}
      </span>

      {cpuGHz && <div className="client-meta">{cpuGHz} GHz</div>}
    </div>
  )
}
