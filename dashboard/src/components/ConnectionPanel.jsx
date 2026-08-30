import { ServerIcon, DbIcon, CloudIcon } from './Icons.jsx'

function Row({ icon, label, sub, ok }) {
  return (
    <div className="conn-row">
      <div className="conn-icon">{icon}</div>
      <div className="conn-text">
        <div className="conn-label">{label}</div>
        {sub && <div className="conn-sub">{sub}</div>}
      </div>
      <span className={`dot ${ok ? 'green pulse' : 'red'}`} title={ok ? 'Connected' : 'Disconnected'} />
    </div>
  )
}

export default function ConnectionPanel({ serverUp, redisUp, s3Up, s3Mode, s3Bucket }) {
  return (
    <div className="card">
      <h2>System Health</h2>
      <div className="conn-list">
        <Row icon={<ServerIcon />} label="FL Server" sub="FastAPI / uvicorn" ok={serverUp} />
        <Row icon={<DbIcon />} label="Redis" sub="broker & live state" ok={redisUp} />
        <Row
          icon={<CloudIcon />}
          label="AWS S3 Storage"
          sub={s3Bucket ? `${s3Mode === 'aws' ? 'AWS S3' : 'Mock S3 (local)'} · ${s3Bucket}` : '—'}
          ok={s3Up}
        />
      </div>
    </div>
  )
}
