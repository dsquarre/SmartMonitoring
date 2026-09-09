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
    <article className={`client-card client-state-${state}`}>
      <div className="client-top">
        <div className="client-chip-wrap">
          <ChipIcon className="client-chip" />
          <span className={`client-online-dot ${state === 'idle' || state === 'waiting' ? '' : 'active'}`} />
        </div>
        <div className="client-heading">
          <div className="client-id">{client.client_id}</div>
          <div className="client-meta">{cpuGHz ? `${cpuGHz} GHz` : 'Edge device'}</div>
        </div>
        {flashDown && <ArrowDownIcon className="flash flash-down" />}
        {flashUp && <ArrowUpIcon className="flash flash-up" />}
      </div>

      <div className="client-status-block">
        <div className="client-status-main">
          <span className={`status-glyph status-${state}`}>
            {state === 'training' && <LoaderIcon className="spin" />}
            {state === 'aggregating' && <LoaderIcon className="spin" />}
            {(state === 'uploaded' || state === 'evaluated') && <CheckIcon />}
            {state === 'waiting' && <span className="waiting-glyph">◌</span>}
            {state === 'idle' && <span className="idle-glyph">●</span>}
          </span>
          <div>
            <b>{LABEL[state]}</b>
            <span>
              {state === 'training' && 'Local training active'}
              {state === 'waiting' && 'Waiting for round assignment'}
              {state === 'uploaded' && 'Local weights uploaded'}
              {state === 'aggregating' && 'Server is aggregating weights'}
              {state === 'evaluated' && 'Round evaluation complete'}
              {state === 'idle' && 'Connected and ready'}
            </span>
          </div>
        </div>
        <div className={`activity-rail rail-${state}`} aria-hidden="true">
          <i /><i /><i /><i /><i />
        </div>
      </div>

      <div className="client-details">
        <div><span>Selection</span><b>{client.selected ? 'Participating' : 'Standby'}</b></div>
        <div><span>Upload</span><b>{client.uploaded ? 'Received' : 'Pending'}</b></div>
      </div>

      <div className="client-state-line">
        <span className={`activity-dot ${state}`} />
        <span>{state === 'training' ? 'Training' : state === 'aggregating' ? 'Aggregation' : state === 'uploaded' ? 'Weights uploaded' : state === 'evaluated' ? 'Evaluated' : state === 'waiting' ? 'Waiting' : 'Connected'}</span>
        {client.selected && <em>Selected</em>}
      </div>
    </article>
  )
}