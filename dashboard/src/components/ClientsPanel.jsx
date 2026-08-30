import ClientCard from './ClientCard.jsx'
import { UsersIcon } from './Icons.jsx'

export default function ClientsPanel({ clients, aggregating, running }) {
  return (
    <div className="card">
      <div className="round-head">
        <h2 style={{ margin: 0 }}>
          <UsersIcon className="inline-icon" /> Connected Clients
        </h2>
        <span className="muted-line">{clients.length} online</span>
      </div>

      {clients.length === 0 ? (
        <p className="muted-line">No clients connected yet.</p>
      ) : (
        <div className="client-grid">
          {clients.map((c) => (
            <ClientCard key={c.client_id} client={c} aggregating={aggregating} running={running} />
          ))}
        </div>
      )}
    </div>
  )
}
