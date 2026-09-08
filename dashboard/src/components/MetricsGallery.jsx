import { useState } from 'react'

const PLOTS = [
  { file: 'loss_vs_round.png', label: 'Loss vs Round', short: 'Loss' },
  { file: 'accuracy_vs_round.png', label: 'Accuracy vs Round', short: 'Accuracy' },
  { file: 'f1_vs_round.png', label: 'Performance Metrics vs Round', short: 'F1 / ROC AUC' },
  { file: 'system_resources_vs_round.png', label: 'System Resources vs Round', short: 'Resources' },
  { file: 'confusion_matrix_latest.png', label: 'Confusion matrix', short: 'Confusion matrix' }
]

function fmt(v, digits = 3) {
  return typeof v === 'number' && !isNaN(v) ? v.toFixed(digits) : '—'
}

export default function MetricsGallery({ history, roundKey }) {
  const [active, setActive] = useState(0)
  const [missing, setMissing] = useState(false)
  const latest = history?.length ? history[history.length - 1] : null
  const plot = PLOTS[active]

  return (
    <section className="metrics-card">
      <div className="metrics-head">
        <div>
          <div className="eyebrow">MODEL TELEMETRY</div>
          <h2>Training Metrics</h2>
          <p>One graph at a time, with the four saved training views.</p>
        </div>
        <span className="metric-live"><i></i>LIVE</span>
      </div>

      <div className="metric-stat-row">
        <div><span>Global Loss</span><b>{latest ? fmt(latest.loss) : '—'}</b></div>
        <div><span>Accuracy</span><b>{latest ? `${fmt((latest.accuracy ?? 0) * 100, 1)}%` : '—'}</b></div>
        <div><span>F1 Score</span><b>{latest ? fmt(latest.f1) : '—'}</b></div>
        <div><span>ROC AUC</span><b>{latest ? fmt(latest.roc_auc) : '—'}</b></div>
        <div><span>Avg Latency</span><b>{latest ? `${fmt(latest.avg_comp_latency, 1)}s` : '—'}</b></div>
        <div><span>Round Energy</span><b>{latest ? `${fmt(latest.total_round_energy, 1)}J` : '—'}</b></div>
      </div>

      <div className="plot-tabs" role="tablist" aria-label="Training graphs">
        {PLOTS.map((p, i) => (
          <button
            key={p.file}
            className={active === i ? 'active' : ''}
            onClick={() => { setActive(i); setMissing(false) }}
            role="tab"
            aria-selected={active === i}
          >
            <span>{String(i + 1).padStart(2, '0')}</span>{p.short}
          </button>
        ))}
      </div>

      <div className="selected-plot">
        <div className="selected-plot-title">
          <div><b>{plot.label}</b><span>Federated round {roundKey || '—'}</span></div>
          <span>{active + 1} / {PLOTS.length}</span>
        </div>
        {missing ? (
          <div className="plot-missing">
            <div>⌁</div>
            <b>{plot.file}</b>
            <span>Plot not generated yet.</span>
          </div>
        ) : (
          <img
            key={`${plot.file}-${roundKey}`}
            src={`/plots/${plot.file}?r=${roundKey}`}
            alt={plot.label}
            onError={() => setMissing(true)}
          />
        )}
      </div>
    </section>
  )
}