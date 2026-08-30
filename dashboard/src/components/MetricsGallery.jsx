import { useState } from 'react'

const PLOTS = [
  { file: 'loss_vs_round.png', label: 'Loss vs Round' },
  { file: 'accuracy_vs_round.png', label: 'Accuracy vs Round' },
  { file: 'f1_vs_round.png', label: 'Disease F1 vs Round' },
  { file: 'system_resources_vs_round.png', label: 'System Resources vs Round' },
]

function PlotCard({ file, label, roundKey }) {
  const [missing, setMissing] = useState(false)
  return (
    <div className="plot-card">
      <div className="plot-label">{label}</div>
      {missing ? (
        <div className="placeholder-plot">Not generated yet</div>
      ) : (
        <img
          src={`/plots/${file}?r=${roundKey}`}
          alt={label}
          onError={() => setMissing(true)}
          onLoad={() => setMissing(false)}
        />
      )}
    </div>
  )
}

function fmt(v, digits = 3) {
  return typeof v === 'number' ? v.toFixed(digits) : '—'
}

export default function MetricsGallery({ history, roundKey }) {
  const latest = history && history.length > 0 ? history[history.length - 1] : null

  return (
    <div className="card">
      <h2>Training Metrics</h2>

      {latest && (
        <div className="stat-row" style={{ marginBottom: 20 }}>
          <div className="stat">
            <div className="value">{fmt(latest.total_loss)}</div>
            <div className="label">Global loss</div>
          </div>
          <div className="stat">
            <div className="value">{fmt((latest.anomaly_accuracy ?? 0) * 100, 1)}%</div>
            <div className="label">Anomaly acc.</div>
          </div>
          <div className="stat">
            <div className="value">{fmt((latest.disease_accuracy ?? 0) * 100, 1)}%</div>
            <div className="label">Disease acc.</div>
          </div>
          <div className="stat">
            <div className="value">{fmt(latest.disease_f1)}</div>
            <div className="label">Disease F1</div>
          </div>
          <div className="stat">
            <div className="value">{fmt(latest.avg_comp_latency, 1)}s</div>
            <div className="label">Avg latency</div>
          </div>
          <div className="stat">
            <div className="value">{fmt(latest.total_round_energy, 1)}J</div>
            <div className="label">Round energy</div>
          </div>
        </div>
      )}

      <div className="metrics-grid">
        {PLOTS.map((p) => (
          <PlotCard key={p.file} file={p.file} label={p.label} roundKey={roundKey} />
        ))}
      </div>
    </div>
  )
}
