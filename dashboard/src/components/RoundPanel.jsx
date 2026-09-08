import { LoaderIcon, UsersIcon } from './Icons.jsx'

export default function RoundPanel({ status, selectedCount }) {
  if (!status) return <div className="card round-panel"><h2>Federated Round</h2><p className="muted-line">Waiting for server…</p></div>

  const current = status.current_round ?? 0
  const total = status.total_rounds ?? 0
  const pct = total ? Math.min(100, Math.round((current / total) * 100)) : 0
  const waiting = !status.fl_running && current === 0

  return (
    <div className="card round-panel">
      <div className="section-head">
        <div><h2>Federated Round</h2><p>Round aggregation status</p></div>
        <span className="method-pill">{status.aggregator || 'FedAvg'}</span>
      </div>
      {waiting ? (
        <div className="round-waiting"><UsersIcon /><div><b>Waiting for clients</b><span>{status.connected_clients ?? 0} / {status.n_required ?? '—'} connected</span></div></div>
      ) : (
        <>
          <div className="round-big"><b>Round {current}</b><span>/ {total} max</span></div>
          <div className="round-progress"><i style={{width: `${pct}%`}}></i></div>
          <div className="round-progress-meta"><span>{pct}% complete</span><span>{status.rounds_left ?? '—'} rounds left</span></div>
          <div className="round-stat-grid">
            <div><span>Selected</span><b>{selectedCount} / {status.k_selected ?? '—'}</b></div>
            <div><span>Connected</span><b>{status.connected_clients ?? 0} / {status.n_required ?? '—'}</b></div>
            <div><span>Aggregator</span><b>{status.aggregator || '—'}</b></div>
            <div><span>Selector</span><b>{status.selector || '—'}</b></div>
          </div>
        </>
      )}
      {status.fl_running && <div className="running-pill"><LoaderIcon className="spin" /> Training in progress</div>}
      {status.aggregating && <div className="agg-banner"><LoaderIcon className="spin" /> Aggregating client weights with <b>{status.aggregator}</b>…</div>}
    </div>
  )
}