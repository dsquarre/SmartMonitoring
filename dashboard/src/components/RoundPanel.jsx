import { LoaderIcon, UsersIcon } from './Icons.jsx'

function Stat({ label, value }) {
  return (
    <div className="stat">
      <div className="value">{value}</div>
      <div className="label">{label}</div>
    </div>
  )
}

export default function RoundPanel({ status, selectedCount }) {
  if (!status) {
    return (
      <div className="card">
        <h2>Federated Round</h2>
        <p className="muted-line">Waiting for server…</p>
      </div>
    )
  }

  const {
    fl_running,
    current_round,
    total_rounds,
    rounds_left,
    n_required,
    k_selected,
    connected_clients,
    aggregating,
    aggregator,
    selector,
  } = status

  const complete = !fl_running && current_round > 0 && rounds_left === 0
  const waitingForClients = !fl_running && current_round === 0
  const progressPct = total_rounds > 0 ? Math.min(100, Math.round((current_round / total_rounds) * 100)) : 0

  return (
    <div className="card">
      <div className="round-head">
        <h2 style={{ margin: 0 }}>Federated Round</h2>
        {fl_running && (
          <span className="badge state-training">
            <LoaderIcon className="spin" /> running
          </span>
        )}
        {complete && (
          <span className="badge state-evaluated">complete</span>
        )}
      </div>

      {waitingForClients ? (
        <div className="waiting-block">
          <span className="dot gray pulse" />
          <UsersIcon />
          <span>
            Waiting for clients to connect — <b>{connected_clients}</b> / {n_required}
          </span>
        </div>
      ) : (
        <>
          <div className="progress-track">
            <div className="progress-fill" style={{ width: `${progressPct}%` }} />
          </div>
          <div className="stat-row">
            <Stat label="Round" value={`${current_round} / ${total_rounds}`} />
            <Stat label="Rounds left" value={rounds_left} />
            <Stat label="Selected" value={`${selectedCount} / ${k_selected}`} />
            <Stat label="Connected" value={`${connected_clients} / ${n_required}`} />
            <Stat label="Aggregator" value={aggregator} />
            <Stat label="Selector" value={selector} />
          </div>
        </>
      )}

      {aggregating && (
        <div className="agg-banner">
          <LoaderIcon className="spin" />
          <span>
            Aggregating client weights with <b>{aggregator}</b>…
          </span>
        </div>
      )}
    </div>
  )
}
