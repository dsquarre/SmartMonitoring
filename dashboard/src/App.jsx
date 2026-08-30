import { usePoll } from './hooks/usePoll.js'
import { getStatus, getClients, getHistory } from './api.js'
import ConnectionPanel from './components/ConnectionPanel.jsx'
import RoundPanel from './components/RoundPanel.jsx'
import ClientsPanel from './components/ClientsPanel.jsx'
import MetricsGallery from './components/MetricsGallery.jsx'

export default function App() {
  const { data: status, error: statusError } = usePoll(getStatus, 2000)
  const { data: clientsData } = usePoll(getClients, 2000)
  const { data: historyData } = usePoll(getHistory, 4000)

  const serverUp = !statusError && !!status
  const redisUp = serverUp && !!status.redis_connected
  const s3Up = serverUp && !!status.s3_connected
  const clients = clientsData?.clients ?? []
  const selectedCount = clients.filter((c) => c.selected).length
  const history = historyData?.history ?? []

  return (
    <div className="app">
      <header className="top">
        <div>
          <h1>SmartMonitoring</h1>
          <div className="subtitle">Federated Learning Live Dashboard</div>
        </div>
      </header>

      {!serverUp && (
        <div className="banner-error">
          Can&apos;t reach the FL server. Make sure uvicorn is running and refresh this page.
        </div>
      )}

      <div className="grid-top">
        <ConnectionPanel
          serverUp={serverUp}
          redisUp={redisUp}
          s3Up={s3Up}
          s3Mode={status?.s3_mode}
          s3Bucket={status?.s3_bucket}
        />
        <RoundPanel status={status} selectedCount={selectedCount} />
      </div>

      <div style={{ marginTop: 20 }}>
        <ClientsPanel clients={clients} aggregating={!!status?.aggregating} running={!!status?.fl_running} />
      </div>

      <div style={{ marginTop: 20 }}>
        <MetricsGallery history={history} roundKey={status?.current_round ?? 0} />
      </div>
    </div>
  )
}
