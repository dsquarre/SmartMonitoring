const base = {
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: 1.7,
  strokeLinecap: 'round',
  strokeLinejoin: 'round',
}

export function ServerIcon({ size = 18, className }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" className={className} {...base}>
      <rect x="3" y="3.5" width="18" height="6.5" rx="1.4" />
      <rect x="3" y="14" width="18" height="6.5" rx="1.4" />
      <circle cx="7" cy="6.75" r="0.9" fill="currentColor" stroke="none" />
      <circle cx="7" cy="17.25" r="0.9" fill="currentColor" stroke="none" />
      <line x1="11" y1="6.75" x2="17" y2="6.75" />
      <line x1="11" y1="17.25" x2="17" y2="17.25" />
    </svg>
  )
}

export function DbIcon({ size = 18, className }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" className={className} {...base}>
      <ellipse cx="12" cy="5.5" rx="8" ry="3" />
      <path d="M4 5.5v6c0 1.66 3.58 3 8 3s8-1.34 8-3v-6" />
      <path d="M4 11.5v6c0 1.66 3.58 3 8 3s8-1.34 8-3v-6" />
    </svg>
  )
}

export function CloudIcon({ size = 18, className }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" className={className} {...base}>
      <path d="M7 18h10.5a3.5 3.5 0 0 0 0-7 5.5 5.5 0 0 0-10.66-1.9A4 4 0 0 0 7 18Z" />
    </svg>
  )
}

export function ChipIcon({ size = 18, className }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" className={className} {...base}>
      <rect x="6" y="6" width="12" height="12" rx="2" />
      <rect x="9.5" y="9.5" width="5" height="5" rx="1" />
      <line x1="12" y1="1.5" x2="12" y2="4" />
      <line x1="12" y1="20" x2="12" y2="22.5" />
      <line x1="1.5" y1="12" x2="4" y2="12" />
      <line x1="20" y1="12" x2="22.5" y2="12" />
    </svg>
  )
}

export function ArrowDownIcon({ size = 14, className }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" className={className} {...base}>
      <line x1="12" y1="4" x2="12" y2="18" />
      <polyline points="6,12 12,18 18,12" />
    </svg>
  )
}

export function ArrowUpIcon({ size = 14, className }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" className={className} {...base}>
      <line x1="12" y1="20" x2="12" y2="6" />
      <polyline points="6,12 12,6 18,12" />
    </svg>
  )
}

export function LoaderIcon({ size = 14, className }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" className={className} {...base}>
      <path d="M12 3a9 9 0 1 0 9 9" strokeOpacity="1" />
    </svg>
  )
}

export function CheckIcon({ size = 14, className }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" className={className} {...base}>
      <polyline points="4,13 9.5,18.5 20,6" />
    </svg>
  )
}

export function UsersIcon({ size = 16, className }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" className={className} {...base}>
      <circle cx="9" cy="8" r="3.2" />
      <path d="M2.5 20c0-3.6 2.9-6 6.5-6s6.5 2.4 6.5 6" />
      <circle cx="17" cy="8.5" r="2.6" />
      <path d="M15.7 14.2c2.9.4 4.8 2.5 4.8 5.8" />
    </svg>
  )
}
