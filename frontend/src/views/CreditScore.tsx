import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { Button } from '../ui/Button'

const API = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8080'

function getToken() {
  return localStorage.getItem('gopay_token') ?? ''
}

function authHeaders() {
  return { Authorization: `Bearer ${getToken()}`, 'Content-Type': 'application/json' }
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type Breakdown = {
  balanceFactor: number
  activityFactor: number
  netFlowFactor: number
  recencyFactor: number
  accountAgeFactor: number
}

type Features = {
  wallet_balance: number
  total_transactions: number
  total_sent: number
  total_received: number
  avg_transaction_amount: number
  account_age_days: number
  days_since_last_txn: number
  txn_frequency_per_week: number
}

type CreditScoreResponse = {
  userId: string
  name: string
  score: number
  riskBand: 'EXCELLENT' | 'VERY_GOOD' | 'GOOD' | 'FAIR' | 'POOR'
  colour: string
  tip: string
  model: string
  breakdown: Breakdown
  features: Features
}

// ---------------------------------------------------------------------------
// Gauge helpers
// ---------------------------------------------------------------------------

const BAND_LABEL: Record<string, string> = {
  EXCELLENT: 'Excellent',
  VERY_GOOD: 'Very Good',
  GOOD: 'Good',
  FAIR: 'Fair',
  POOR: 'Poor',
}

function scorePercent(score: number) {
  return ((score - 300) / 600) * 100
}

function formatINR(n: number) {
  return new Intl.NumberFormat('en-IN', { style: 'currency', currency: 'INR', maximumFractionDigits: 0 }).format(n)
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function polar(cx: number, cy: number, r: number, deg: number) {
  const rad = (deg * Math.PI) / 180
  return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) }
}

function describeArc(cx: number, cy: number, r: number, startDeg: number, endDeg: number) {
  const s = polar(cx, cy, r, startDeg)
  const e = polar(cx, cy, r, endDeg)
  const large = endDeg - startDeg > 180 ? 1 : 0
  return `M ${s.x} ${s.y} A ${r} ${r} 0 ${large} 1 ${e.x} ${e.y}`
}

/** SVG arc gauge: 0-100 fill mapped to a 220° arc */
function ScoreGauge({ score, colour }: Readonly<{ score: number; colour: string }>) {
  const R = 80
  const cx = 100
  const cy = 100
  const startAngle = -200   // degrees, measured from 3-o'clock
  const totalArc   = 220    // degrees of visible arc

  const pct   = scorePercent(score) / 100
  const bgEnd = startAngle + totalArc
  const fgEnd = startAngle + totalArc * pct

  return (
    <svg viewBox="0 0 200 155" className="w-full max-w-[260px] mx-auto" aria-label={`Credit score ${score}`}>
      {/* track */}
      <path
        d={describeArc(cx, cy, R, startAngle, bgEnd)}
        fill="none"
        stroke="#e5e7eb"
        strokeWidth="16"
        strokeLinecap="round"
      />
      {/* fill */}
      <path
        d={describeArc(cx, cy, R, startAngle, fgEnd)}
        fill="none"
        stroke={colour}
        strokeWidth="16"
        strokeLinecap="round"
        style={{ transition: 'stroke-dashoffset 0.6s ease' }}
      />
      {/* score text */}
      <text x={cx} y={cy - 4} textAnchor="middle" fontSize="32" fontWeight="800" fill="#111827">
        {score}
      </text>
      <text x={cx} y={cy + 18} textAnchor="middle" fontSize="11" fill="#6b7280">
        out of 900
      </text>
      {/* range labels — positioned below the arc ends */}
      <text x="18" y="148" fontSize="9" fill="#9ca3af">300</text>
      <text x="168" y="148" fontSize="9" fill="#9ca3af">900</text>
    </svg>
  )
}

/** Single factor bar */
function FactorBar({
  label,
  value,
  colour,
  detail,
}: Readonly<{ label: string; value: number; colour: string; detail: string }>) {
  return (
    <div className="grid gap-1">
      <div className="flex items-center justify-between text-sm">
        <span className="font-medium text-(--gopay-fg)">{label}</span>
        <span className="text-(--gopay-muted)">{detail}</span>
      </div>
      <div className="h-2 w-full rounded-full bg-gray-100">
        <div
          className="h-2 rounded-full transition-all duration-700"
          style={{ width: `${Math.min(100, Math.max(0, value))}%`, background: colour }}
        />
      </div>
    </div>
  )
}

/** Metric chip for features section */
function Metric({ label, value }: Readonly<{ label: string; value: string }>) {
  return (
    <div className="rounded-2xl border border-(--gopay-border) bg-white/60 px-4 py-3">
      <div className="text-xs text-(--gopay-muted)">{label}</div>
      <div className="mt-0.5 text-base font-bold text-(--gopay-fg)">{value}</div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

export function CreditScore() {
  const token = useMemo(() => getToken(), [])
  const [data, setData] = useState<CreditScoreResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!token) {
      setLoading(false)
      return
    }
    fetch(`${API}/api/v1/credit/score`, { headers: authHeaders() })
      .then(async (r) => {
        const json: unknown = await r.json().catch(() => null)
        if (!r.ok) {
          const msg = (json as { error?: unknown } | null)?.error
          throw new Error(typeof msg === 'string' ? msg : 'Failed to load credit score.')
        }
        return json as CreditScoreResponse
      })
      .then(setData)
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false))
  }, [token])

  if (!token) {
    return (
      <div className="mx-auto max-w-md">
        <div className="rounded-3xl border border-(--gopay-border) bg-white/75 p-8 text-center shadow-(--gopay-shadow)">
          <div className="text-4xl">🔒</div>
          <h2 className="mt-4 text-xl font-bold">Sign in to view your credit score</h2>
          <Link to="/login" className="mt-6 block">
            <Button className="w-full">Log in</Button>
          </Link>
        </div>
      </div>
    )
  }

  if (loading) {
    return (
      <div className="mx-auto max-w-lg space-y-4">
        <div className="h-64 animate-pulse rounded-3xl bg-gray-100" />
        <div className="h-32 animate-pulse rounded-3xl bg-gray-100" />
      </div>
    )
  }

  if (error || !data) {
    return (
      <div className="mx-auto max-w-md">
        <div className="rounded-3xl border border-red-200 bg-red-50 p-6 text-center">
          <p className="font-semibold text-red-700">{error ?? 'Unable to load credit score.'}</p>
          <p className="mt-2 text-sm text-red-600">Make sure the Spring Boot server is running.</p>
          <Link to="/dashboard" className="mt-4 block">
            <Button variant="ghost">← Back to Dashboard</Button>
          </Link>
        </div>
      </div>
    )
  }

  const bd = data.breakdown
  const ft = data.features
  const bandLabel = BAND_LABEL[data.riskBand] ?? data.riskBand

  const factors = [
    { label: 'Wallet Balance',    value: bd.balanceFactor,    detail: `${bd.balanceFactor.toFixed(0)}/100` },
    { label: 'Transaction Activity', value: bd.activityFactor, detail: `${bd.activityFactor.toFixed(0)}/100` },
    { label: 'Net Cash Flow',     value: bd.netFlowFactor,    detail: `${bd.netFlowFactor.toFixed(0)}/100` },
    { label: 'Recency',          value: bd.recencyFactor,    detail: `${bd.recencyFactor.toFixed(0)}/100` },
    { label: 'Account Maturity', value: bd.accountAgeFactor, detail: `${bd.accountAgeFactor.toFixed(0)}/100` },
  ]

  return (
    <div className="mx-auto grid max-w-xl gap-5">

      {/* ── Score card ── */}
      <section className="rounded-3xl border border-(--gopay-border) bg-white/75 p-6 shadow-(--gopay-shadow)">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm font-semibold text-(--gopay-muted)">Credit Score</p>
            <h1 className="text-2xl font-extrabold tracking-tight">{data.name}</h1>
          </div>
          {/* Band badge */}
          <span
            className="rounded-full px-3 py-1 text-sm font-bold text-white"
            style={{ background: data.colour }}
          >
            {bandLabel}
          </span>
        </div>

        <ScoreGauge score={data.score} colour={data.colour} />

        {/* Tip */}
        <p className="mt-3 rounded-2xl bg-gray-50 px-4 py-3 text-sm text-(--gopay-muted)">{data.tip}</p>
      </section>

      {/* ── Factor breakdown ── */}
      <section className="rounded-3xl border border-(--gopay-border) bg-white/75 p-6 shadow-(--gopay-shadow)">
        <h2 className="text-base font-bold">Score Breakdown</h2>
        <p className="mb-4 text-sm text-(--gopay-muted)">How each factor contributes to your score</p>
        <div className="grid gap-4">
          {factors.map((f) => (
            <FactorBar key={f.label} label={f.label} value={f.value} colour={data.colour} detail={f.detail} />
          ))}
        </div>
      </section>

      {/* ── Raw features ── */}
      <section className="rounded-3xl border border-(--gopay-border) bg-white/75 p-6 shadow-(--gopay-shadow)">
        <h2 className="text-base font-bold">Platform Data Used</h2>
        <p className="mb-4 text-sm text-(--gopay-muted)">Inputs derived from your real account activity</p>
        <div className="grid grid-cols-2 gap-3">
          <Metric label="Wallet Balance"          value={formatINR(ft.wallet_balance)} />
          <Metric label="Total Transactions"       value={String(ft.total_transactions)} />
          <Metric label="Total Sent"               value={formatINR(ft.total_sent)} />
          <Metric label="Total Received"           value={formatINR(ft.total_received)} />
          <Metric label="Avg Transaction"          value={formatINR(ft.avg_transaction_amount)} />
          <Metric label="Account Age"              value={`${ft.account_age_days} days`} />
          <Metric label="Days Since Last Txn"      value={`${ft.days_since_last_txn} days`} />
          <Metric label="Txn Frequency"            value={`${ft.txn_frequency_per_week.toFixed(2)}/week`} />
        </div>
      </section>

      {/* ── Navigation ── */}
      <Link to="/dashboard" className="text-center text-sm text-(--gopay-muted) hover:text-(--gopay-fg)">
        ← Back to Dashboard
      </Link>
    </div>
  )
}
