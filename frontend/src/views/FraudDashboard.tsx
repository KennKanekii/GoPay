import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { Button } from '../ui/Button'

const API = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8080'

function getToken() { return localStorage.getItem('gopay_token') ?? '' }
function authHeaders() {
  return { Authorization: `Bearer ${getToken()}`, 'Content-Type': 'application/json' }
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type VelocitySummary = {
  txns1h: number; txns24h: number
  amountSent1h: number; amountSent24h: number
  uniqueRecips24h: number
  limitTxns1h: number; limitAmount1h: number
  limitTxns24h: number; limitAmount24h: number
  limitRecips24h: number
}

type FraudEvent = {
  id: string
  fromUserId: string
  fromName: string
  fromIdentifier: string
  toIdentifier: string
  amount: number
  fraudScore: number
  riskLevel: string
  recommendation: string
  signals: string[]
  model: string
  createdAt: string
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const RISK_STYLES: Record<string, { bg: string; text: string; border: string; dot: string }> = {
  LOW:      { bg: 'bg-green-50',  text: 'text-green-700',  border: 'border-green-200', dot: 'bg-green-500'  },
  MEDIUM:   { bg: 'bg-yellow-50', text: 'text-yellow-700', border: 'border-yellow-200',dot: 'bg-yellow-500' },
  HIGH:     { bg: 'bg-orange-50', text: 'text-orange-700', border: 'border-orange-200',dot: 'bg-orange-500' },
  CRITICAL: { bg: 'bg-red-50',    text: 'text-red-700',    border: 'border-red-200',   dot: 'bg-red-500'    },
}

function riskStyle(level: string) {
  return RISK_STYLES[level] ?? RISK_STYLES.LOW
}

function formatINR(n: number) {
  return new Intl.NumberFormat('en-IN', { style: 'currency', currency: 'INR', maximumFractionDigits: 0 }).format(n)
}

function formatDate(iso: string) {
  return new Date(iso).toLocaleString('en-IN', { dateStyle: 'medium', timeStyle: 'short' })
}

function scoreColour(score: number) {
  if (score >= 80) return '#dc2626'
  if (score >= 60) return '#ea580c'
  if (score >= 35) return '#d97706'
  return '#16a34a'
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function VelocityGauge({
  label, current, limit, formatValue,
}: Readonly<{ label: string; current: number; limit: number; formatValue: (n: number) => string }>) {
  const pct = Math.min(100, (current / limit) * 100)
  let colour = '#16a34a'
  if (pct >= 90) colour = '#dc2626'
  else if (pct >= 70) colour = '#d97706'
  return (
    <div className="grid gap-1.5">
      <div className="flex items-center justify-between text-sm">
        <span className="font-medium text-(--gopay-fg)">{label}</span>
        <span className="text-(--gopay-muted)">
          {formatValue(current)} <span className="text-xs">/ {formatValue(limit)}</span>
        </span>
      </div>
      <div className="h-2 w-full rounded-full bg-gray-100">
        <div
          className="h-2 rounded-full transition-all duration-700"
          style={{ width: `${pct}%`, background: colour }}
        />
      </div>
    </div>
  )
}

function ScoreBadge({ score, riskLevel }: Readonly<{ score: number; riskLevel: string }>) {
  const s = riskStyle(riskLevel)
  return (
    <div className={`flex items-center gap-2 rounded-xl border ${s.border} ${s.bg} px-2.5 py-1`}>
      <span className={`h-2 w-2 rounded-full ${s.dot}`} />
      <span className={`text-xs font-bold ${s.text}`}>{score}</span>
    </div>
  )
}

function SignalPill({ signal }: Readonly<{ signal: string }>) {
  return (
    <span className="inline-block rounded-full bg-gray-100 px-2 py-0.5 text-xs font-medium text-gray-600">
      {signal.replaceAll('_', ' ')}
    </span>
  )
}

function StatCard({
  label, value, sub, accent,
}: Readonly<{ label: string; value: string; sub?: string; accent?: string }>) {
  return (
    <div className="rounded-2xl border border-(--gopay-border) bg-white/60 p-4">
      <div className="text-xs font-semibold text-(--gopay-muted)">{label}</div>
      <div className="mt-1 text-2xl font-extrabold" style={accent ? { color: accent } : {}}>
        {value}
      </div>
      {sub ? <div className="mt-0.5 text-xs text-(--gopay-muted)">{sub}</div> : null}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

export function FraudDashboard() {
  const token = useMemo(() => getToken(), [])
  const [velocity, setVelocity]   = useState<VelocitySummary | null>(null)
  const [events,   setEvents]     = useState<FraudEvent[]>([])
  const [loading,  setLoading]    = useState(true)

  useEffect(() => {
    if (!token) { setLoading(false); return }

    Promise.all([
      fetch(`${API}/api/v1/fraud/velocity`, { headers: authHeaders() }).then((r) => r.json() as Promise<VelocitySummary>),
      fetch(`${API}/api/v1/fraud/events`,   { headers: authHeaders() }).then((r) => r.json() as Promise<FraudEvent[]>),
    ])
      .then(([v, e]) => {
        setVelocity(v)
        setEvents(Array.isArray(e) ? e.slice(0, 50) : [])
      })
      .catch(() => null)
      .finally(() => setLoading(false))
  }, [token])

  if (!token) {
    return (
      <div className="mx-auto max-w-md">
        <div className="rounded-3xl border border-(--gopay-border) bg-white/75 p-8 text-center shadow-(--gopay-shadow)">
          <div className="text-4xl">🔒</div>
          <h2 className="mt-4 text-xl font-bold">Sign in to view your fraud dashboard</h2>
          <Link to="/login" className="mt-6 block"><Button className="w-full">Log in</Button></Link>
        </div>
      </div>
    )
  }

  // Derived stats
  const blocked   = events.filter((e) => e.recommendation === 'BLOCK').length
  const flagged   = events.filter((e) => e.riskLevel === 'HIGH' || e.riskLevel === 'CRITICAL').length
  const avgScore  = events.length ? Math.round(events.reduce((s, e) => s + e.fraudScore, 0) / events.length) : 0
  const highestRisk = events.reduce((max, e) => Math.max(max, e.fraudScore), 0)

  return (
    <div className="mx-auto grid max-w-3xl gap-5">

      {/* ── Header ── */}
      <div>
        <h1 className="text-3xl font-extrabold tracking-tight">Fraud Shield</h1>
        <p className="mt-1 text-sm text-(--gopay-muted)">
          Real-time fraud detection — velocity rules, blacklist check, and ML anomaly scoring.
        </p>
      </div>

      {/* ── Summary stats ── */}
      {loading ? (
        <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
          {[1,2,3,4].map((i) => <div key={i} className="h-20 animate-pulse rounded-2xl bg-gray-100" />)}
        </div>
      ) : (
        <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
          <StatCard label="Transactions scored"  value={String(events.length)} sub="All time" />
          <StatCard label="Blocked"              value={String(blocked)}  sub="Prevented" accent="#dc2626" />
          <StatCard label="Flagged (High+)"      value={String(flagged)}  sub="Reviewed"  accent="#ea580c" />
          <StatCard label="Peak risk score"      value={String(highestRisk)} sub={`Avg: ${avgScore}`}
                    accent={scoreColour(highestRisk)} />
        </div>
      )}

      {/* ── Velocity limits ── */}
      <section className="rounded-3xl border border-(--gopay-border) bg-white/75 p-6 shadow-(--gopay-shadow)">
        <h2 className="text-base font-bold">Velocity Limits</h2>
        <p className="mb-4 text-sm text-(--gopay-muted)">
          Aligned with RBI transaction monitoring guidelines. Limits reset every 1h / 24h.
        </p>
        {loading || !velocity ? (
          <div className="space-y-3">
            {[1,2,3,4,5].map((i) => <div key={i} className="h-8 animate-pulse rounded-xl bg-gray-100" />)}
          </div>
        ) : (
          <div className="grid gap-4">
            <VelocityGauge label="Transactions / hour"
              current={velocity.txns1h} limit={velocity.limitTxns1h}
              formatValue={String} />
            <VelocityGauge label="Amount sent / hour"
              current={velocity.amountSent1h} limit={velocity.limitAmount1h}
              formatValue={formatINR} />
            <VelocityGauge label="Transactions / 24 hours"
              current={velocity.txns24h} limit={velocity.limitTxns24h}
              formatValue={String} />
            <VelocityGauge label="Amount sent / 24 hours"
              current={velocity.amountSent24h} limit={velocity.limitAmount24h}
              formatValue={formatINR} />
            <VelocityGauge label="Unique recipients / 24 hours"
              current={velocity.uniqueRecips24h} limit={velocity.limitRecips24h}
              formatValue={String} />
          </div>
        )}
      </section>

      {/* ── Fraud event log ── */}
      <section className="rounded-3xl border border-(--gopay-border) bg-white/75 p-6 shadow-(--gopay-shadow)">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-base font-bold">Risk Assessment Log</h2>
            <p className="text-sm text-(--gopay-muted)">Every transaction you initiate is scored in real time.</p>
          </div>
          <span className="rounded-full bg-gray-100 px-2.5 py-0.5 text-xs font-semibold text-gray-600">
            {events.length} events
          </span>
        </div>

        {loading && (
          <div className="mt-4 space-y-3">
            {[1,2,3].map((i) => <div key={i} className="h-20 animate-pulse rounded-2xl bg-gray-100" />)}
          </div>
        )}
        {!loading && events.length === 0 && (
          <div className="mt-6 rounded-2xl border border-(--gopay-border) bg-white p-8 text-center">
            <div className="text-3xl">✅</div>
            <div className="mt-2 font-semibold">No fraud events yet</div>
            <div className="mt-1 text-sm text-(--gopay-muted)">Events appear here when you initiate transactions.</div>
          </div>
        )}
        {!loading && events.length > 0 && (
          <div className="mt-4 grid gap-3">
            {events.map((ev) => {
              const s = riskStyle(ev.riskLevel)
              const blocked = ev.recommendation === 'BLOCK'
              return (
                <div
                  key={ev.id}
                  className={`rounded-2xl border ${s.border} ${blocked ? s.bg : 'bg-white'} p-4`}
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 flex-wrap">
                        <ScoreBadge score={ev.fraudScore} riskLevel={ev.riskLevel} />
                        {blocked && (
                          <span className="rounded-full bg-red-100 px-2 py-0.5 text-xs font-bold text-red-700">
                            BLOCKED
                          </span>
                        )}
                        <span className="text-sm font-semibold truncate">
                          {formatINR(ev.amount)} → {ev.toIdentifier}
                        </span>
                      </div>
                      {ev.signals && ev.signals.length > 0 && (
                        <div className="mt-2 flex flex-wrap gap-1">
                          {ev.signals.slice(0, 4).map((sig) => <SignalPill key={sig} signal={sig} />)}
                        </div>
                      )}
                    </div>
                    <div className="text-right shrink-0">
                      <div className="text-xs text-(--gopay-muted)">{formatDate(ev.createdAt)}</div>
                      <div className="mt-1 text-xs text-gray-400">{ev.model.replaceAll('_', ' ')}</div>
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </section>

      {/* ── How it works ── */}
      <section className="rounded-3xl border border-(--gopay-border) bg-white/75 p-6 shadow-(--gopay-shadow)">
        <h2 className="mb-4 text-base font-bold">How GoPay Fraud Shield Works</h2>
        <div className="grid gap-3">
          {[
            { title: 'Blacklist Check',      desc: 'Instantly blocks transactions to known fraudulent emails, disposable domains, and watchlisted recipients.' },
            { title: 'Velocity Rules',       desc: 'Hard limits on transaction count and amount per hour and per day, aligned with RBI monitoring guidelines.' },
            { title: 'Behavioural Analysis', desc: 'Flags deviations from your own patterns — unusual amounts, new recipients, odd hours, balance drain.' },
            { title: 'ML Anomaly Scoring',   desc: 'RandomForest model trained on 5 real fraud archetypes (ATO, velocity fraud, structuring, money mule, blacklist) scores every transaction 0–100.' },
            { title: 'Audit Trail',          desc: 'Every assessment is logged with signals and model used, enabling compliance reporting and dispute resolution.' },
          ].map((item) => (
            <div key={item.title} className="flex gap-3">
              <div className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full border border-gray-200 bg-gray-100">
                <div className="h-1.5 w-1.5 rounded-full bg-gray-400" />
              </div>
              <div>
                <div className="text-sm font-semibold">{item.title}</div>
                <div className="text-sm text-(--gopay-muted)">{item.desc}</div>
              </div>
            </div>
          ))}
        </div>
      </section>

      <Link to="/dashboard" className="text-center text-sm text-(--gopay-muted) hover:text-(--gopay-fg)">
        ← Back to Dashboard
      </Link>
    </div>
  )
}
