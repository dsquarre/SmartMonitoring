import { useEffect, useMemo, useState } from 'react'
import { usePoll } from './hooks/usePoll.js'
import { getStatus, getClients, getHistory } from './api.js'
import ConnectionPanel from './components/ConnectionPanel.jsx'
import RoundPanel from './components/RoundPanel.jsx'
import MetricsGallery from './components/MetricsGallery.jsx'
import ClientCard from './components/ClientCard.jsx'

const NAV = [
  { id: 'dashboard', label: 'Dashboard', icon: '▦' },
  { id: 'clients', label: 'Clients', icon: '♧' },
  { id: 'models', label: 'Models', icon: '◇' },
  { id: 'logs', label: 'Activity Logs', icon: '≡' },
  { id: 'settings', label: 'Settings', icon: '⚙' },
]

function Login({ onLogin }) {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [error, setError] = useState('')

  function submit(e) {
    e.preventDefault()

    if (!username.trim() || !password) {
      setError('Enter your username and password.')
      return
    }

    const user = {
      username: username.trim(),
      name: username.trim(),
    }

    onLogin(user)
  }

  return (
    <div className="login-page">
      <div className="login-art">
        <div className="login-grid"></div>
        <div className="login-orbit orbit-a"></div>
        <div className="login-orbit orbit-b"></div>
        <div className="login-brand-mark">✣</div>
        <div className="login-art-copy">
          <div className="eyebrow">FEDERATED LEARNING CONTROL</div>
          <h1>Train globally.<br />Monitor locally.</h1>
          <p>Real-time telemetry for distributed model training, edge clients and federated rounds.</p>
          <div className="login-art-stats">
            <span><b>LIVE</b><small>Telemetry</small></span>
            <span><b>FL</b><small>Federated</small></span>
            <span><b>24/7</b><small>Monitoring</small></span>
          </div>
        </div>
      </div>

      <div className="login-panel">
        <div className="login-form-wrap">
          <div className="mobile-login-brand">✣ SmartMonitoring</div>
          <div className="login-kicker">WELCOME BACK</div>
          <h2>Sign in to your workspace</h2>
          <p className="login-description">Access your federated learning control center.</p>

          <form onSubmit={submit} className="login-form">
            <label>
              Username
              <input
                value={username}
                onChange={(e) => { setUsername(e.target.value); setError('') }}
                placeholder="Enter username"
                autoComplete="username"
                autoFocus
              />
            </label>

            <label>
              Password
              <div className="password-wrap">
                <input
                  type={showPassword ? 'text' : 'password'}
                  value={password}
                  onChange={(e) => { setPassword(e.target.value); setError('') }}
                  placeholder="Enter password"
                  autoComplete="current-password"
                />
                <button type="button" onClick={() => setShowPassword((v) => !v)}>
                  {showPassword ? 'Hide' : 'Show'}
                </button>
              </div>
            </label>

            {error && <div className="login-error">{error}</div>}

            <div className="login-options">
              <label className="remember"><input type="checkbox" defaultChecked /> <span>Remember me</span></label>
              <button type="button" className="text-button" onClick={() => setError('Password reset requires a backend auth service.')}>Forgot password?</button>
            </div>

            <button className="login-submit" type="submit">Sign in <span>→</span></button>
          </form>

          <div className="login-footer">SmartMonitoring · Federated Learning Telemetry</div>
        </div>
      </div>
    </div>
  )
}

function ClientsPage({ clients, aggregating, running, query }) {
  const filtered = clients.filter((c) => {
    const q = query.trim().toLowerCase()
    return !q || `${c.client_id} ${c.status} ${c.state} ${c.device} ${c.cpu} ${c.gpu}`.toLowerCase().includes(q)
  })

  return (
    <div className="page-shell">
      <div className="page-heading">
        <div>
          <div className="eyebrow">EDGE FLEET</div>
          <h2>Connected Clients</h2>
          <p>Monitor participating devices without crowding the main training overview.</p>
        </div>
        <div className="page-count"><b>{clients.length}</b><span>online</span></div>
      </div>

      <div className="client-toolbar">
        <div className="toolbar-title">Federated participants <span>{filtered.length} shown</span></div>
        <div className="toolbar-status"><i></i>{running ? 'Training in progress' : aggregating ? 'Aggregating' : 'Standing by'}</div>
      </div>

      {filtered.length === 0 ? (
        <div className="empty-state">
          <div>⌕</div>
          <h3>No clients found</h3>
          <p>Try a different client ID or search term.</p>
        </div>
      ) : (
        <div className="full-client-grid animated-client-grid">
          {filtered.map((c, index) => (
            <div className="client-card-wrap" key={c.client_id} style={{ '--client-delay': `${Math.min(index, 12) * 45}ms` }}>
              <ClientCard client={c} aggregating={aggregating} running={running} />
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function GenericPage({ title, eyebrow, text }) {
  return (
    <div className="page-shell">
      <div className="page-heading">
        <div>
          <div className="eyebrow">{eyebrow}</div>
          <h2>{title}</h2>
          <p>{text}</p>
        </div>
      </div>
      <div className="coming-card">
        <div className="coming-icon">◇</div>
        <h3>{title}</h3>
        <p>This page is ready for the corresponding operational data. The dashboard navigation is now functional without changing the federated-learning API.</p>
      </div>
    </div>
  )
}

export default function App() {
  const [user, setUser] = useState(null)
  const [page, setPage] = useState('dashboard')
  const [query, setQuery] = useState('')

  const { data: status, error: statusError } = usePoll(getStatus, 2000)
  const { data: clientsData } = usePoll(getClients, 2000)
  const { data: historyData } = usePoll(getHistory, 4000)

  const clients = clientsData?.clients ?? []
  const history = historyData?.history ?? []
  const selectedCount = clients.filter((c) => c.selected).length
  const serverUp = !statusError && !!status
  const redisUp = serverUp && !!status.redis_connected
  const s3Up = serverUp && !!status.s3_connected
  const latest = history.length ? history[history.length - 1] : null
  const round = status?.current_round ?? 0
  const totalRounds = status?.total_rounds ?? 0
  const convergence = latest?.accuracy != null ? `${(latest.accuracy * 100).toFixed(1)}%` : '—'
  const loss = typeof latest?.loss === 'number' ? latest.loss.toFixed(3) : '—'

  const searchResults = useMemo(() => {
    const q = query.trim().toLowerCase()
    if (!q) return []
    const results = []
    NAV.forEach((n) => {
      if (`${n.label} ${n.id}`.toLowerCase().includes(q)) results.push({ type: 'page', label: n.label, page: n.id })
    })
    clients.filter((c) => `${c.client_id}`.toLowerCase().includes(q)).slice(0, 5)
      .forEach((c) => results.push({ type: 'client', label: c.client_id, page: 'clients' }))
    ;[
      ['Loss vs Round', 'metrics'],
      ['Accuracy vs Round', 'metrics'],
      ['Performance Metrics', 'metrics'],
      ['System Resources', 'metrics'],
      ['Federated Round', 'dashboard'],
      ['Model Checkpoints', 'dashboard'],
    ].filter(([label]) => label.toLowerCase().includes(q)).forEach(([label, p]) => results.push({ type: 'metric', label, page: p }))
    return results.slice(0, 8)
  }, [query, clients])

  function logout() {
    setUser(null)
    setPage('dashboard')
  }

  if (!user) return <Login onLogin={setUser} />

  return (
    <div className="dashboard-app">
      <aside className="sidebar">
        <div className="side-brand">
          <div className="side-logo">✣</div>
          <div><b>SmartMonitoring</b><span>Federated Learning</span></div>
        </div>

        <nav className="side-nav">
          {NAV.map((item) => (
            <button key={item.id} className={`nav-item ${page === item.id ? 'active' : ''}`} onClick={() => { setPage(item.id); setQuery('') }}>
              <span className="nav-icon">{item.icon}</span>
              <span>{item.label}</span>
              {item.id === 'clients' && <small>{clients.length}</small>}
            </button>
          ))}
        </nav>

        <div className="side-bottom">
          <div className="side-health"><i className={serverUp ? 'up' : 'down'}></i>{serverUp ? 'System online' : 'Server offline'}</div>
          <button className="user-card" onClick={logout} title="Sign out">
            <div className="user-avatar">{(user.name || user.username || 'U').slice(0, 2).toUpperCase()}</div>
            <div><b>{user.name || user.username}</b><span>Sign out</span></div>
            <span>↗</span>
          </button>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div className="top-title">
            <div className="top-logo">✣</div>
            <div>
              <h1>{page === 'dashboard' ? 'SmartMonitoring' : NAV.find((n) => n.id === page)?.label || 'SmartMonitoring'}</h1>
              <span>{page === 'dashboard' ? `Federated Learning Telemetry · Round ${round} of ${totalRounds || '—'}` : 'SmartMonitoring Control Center'}</span>
            </div>
          </div>

          <div className="top-right">
            <div className="global-search">
              <span>⌕</span>
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => { if (e.key === 'Escape') setQuery('') }}
                placeholder="Search clients, metrics, pages..."
                aria-label="Search"
              />
              {query && <button onClick={() => setQuery('')}>×</button>}
              {searchResults.length > 0 && (
                <div className="search-results">
                  {searchResults.map((r, i) => (
                    <button key={`${r.label}-${i}`} onClick={() => { setPage(r.page); setQuery('') }}>
                      <span className="result-icon">{r.type === 'client' ? '♧' : r.type === 'metric' ? '⌁' : '▦'}</span>
                      <span><b>{r.label}</b><small>{r.type === 'client' ? 'Client' : r.type === 'metric' ? 'Metric' : 'Page'}</small></span>
                      <em>→</em>
                    </button>
                  ))}
                </div>
              )}
            </div>
            <div className="live-indicator"><i></i>Live</div>
          </div>
        </header>

        {!serverUp && (
          <div className="banner-error">Can&apos;t reach the FL server. Live data will appear automatically when the server is available.</div>
        )}

        {page === 'dashboard' && (
          <>
            <section className="summary-grid">
              <div className="summary-card">
                <div><span>Active Nodes</span><b>{status?.connected_clients ?? clients.length} / {status?.n_required ?? '—'}</b><small>{selectedCount} selected for current round</small></div>
                <div className="mini-bars"><i></i><i></i><i></i><i></i><i></i><i></i></div>
              </div>
              <div className="summary-card">
                <div><span>Current Step</span><b>Round {round || '—'}</b><small>{status?.rounds_left ?? '—'} rounds left</small></div>
                <div className="summary-symbol">↻</div>
              </div>
              <div className="summary-card">
                <div><span>Global Loss</span><b>{loss}</b><small>Latest aggregated model</small></div>
                <div className="summary-symbol warm">⌁</div>
              </div>
              <div className="summary-card dark">
                <div><span>Convergence</span><b>{convergence}</b><small>Latest validation accuracy</small></div>
                <div className="summary-spark"><i></i><i></i><i></i><i></i><i></i></div>
              </div>
            </section>

            <section className="primary-grid">
              <ConnectionPanel serverUp={serverUp} redisUp={redisUp} s3Up={s3Up} s3Mode={status?.s3_mode} s3Bucket={status?.s3_bucket} />
              <RoundPanel status={status} selectedCount={selectedCount} />
            </section>

            <section className="bottom-grid">
              <div className="compact-card checkpoint-card">
                <div className="section-head"><div><h2>Model Checkpoints</h2><p>Latest federated weights</p></div><span>SYNC</span></div>
                <div className="checkpoint-row"><i></i><div><b>global_weights_r{round || '—'}.keras</b><small>Latest aggregation</small></div><em>Ready</em></div>
                <div className="checkpoint-row"><i></i><div><b>global_weights_r{Math.max(0, round - 1)}.keras</b><small>Previous checkpoint</small></div><em>Stored</em></div>
                <div className="checkpoint-row"><i className="yellow"></i><div><b>client_drop_eval.keras</b><small>Fallback evaluation</small></div><em>Fallback</em></div>
                <button className="outline-button" onClick={() => setPage('models')}>View model store →</button>
              </div>

              <div className="compact-card health-card">
                <div className="health-icon">♢</div>
                <h2>Cluster Shield</h2>
                <p>Infrastructure health and storage status.</p>
                <div className="health-list">
                  <span>FastAPI Server <b className="ok">● 99.9%</b></span>
                  <span>Redis <b className={redisUp ? 'ok' : 'bad'}>● {redisUp ? 'Healthy' : 'Offline'}</b></span>
                  <span>S3 Weights <b className={s3Up ? 'ok' : 'bad'}>● {s3Up ? 'Synced' : 'Offline'}</b></span>
                </div>
              </div>
            </section>

            <MetricsGallery history={history} roundKey={status?.current_round ?? 0} />
          </>
        )}

        {page === 'clients' && <ClientsPage clients={clients} aggregating={!!status?.aggregating} running={!!status?.fl_running} query={query} />}
        {page === 'models' && <GenericPage eyebrow="MODEL STORE" title="Models & Checkpoints" text="Review the federated weights generated by each training round." />}
        {page === 'logs' && <GenericPage eyebrow="ACTIVITY" title="Activity Logs" text="Review the operational events emitted by the federated learning system." />}
        {page === 'settings' && <GenericPage eyebrow="CONTROL CENTER" title="Settings" text="Configure dashboard preferences and operational controls." />}
      </main>
    </div>
  )
}