import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import type { ReactNode } from 'react'
import { Button } from '../ui/Button'

const API = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8080'

function getToken() {
  return localStorage.getItem('gopay_token') ?? ''
}

function authHeaders() {
  return { Authorization: `Bearer ${getToken()}`, 'Content-Type': 'application/json' }
}

type BalanceResponse = { balance: number; currency: string }

type TxnEntry = {
  id: string
  direction: 'SENT' | 'RECEIVED'
  amount: number
  counterpartyName: string
  counterpartyIdentifier: string
  note: string
  status: string
  createdAt: string
}

function formatINR(amount: number) {
  return new Intl.NumberFormat('en-IN', {
    style: 'currency',
    currency: 'INR',
    maximumFractionDigits: 2,
  }).format(amount)
}

function formatDate(iso: string) {
  return new Date(iso).toLocaleString('en-IN', { dateStyle: 'short', timeStyle: 'short' })
}

function Card({
  title,
  subtitle,
  loading,
  children,
}: Readonly<{ title: string; subtitle: string; loading?: boolean; children?: ReactNode }>) {
  return (
    <section className="rounded-3xl border border-(--gopay-border) bg-white/70 p-6 shadow-(--gopay-shadow)">
      <div>
        <div className="text-sm font-semibold text-(--gopay-muted)">{subtitle}</div>
        <div className="mt-1 text-xl font-extrabold tracking-tight">
          {loading ? <span className="animate-pulse text-(--gopay-muted)">Loading…</span> : title}
        </div>
      </div>
      {children ? <div className="mt-5">{children}</div> : null}
    </section>
  )
}

export function Dashboard() {
  const [balance, setBalance] = useState<number | null>(null)
  const [balanceLoading, setBalanceLoading] = useState(true)
  const [txns, setTxns] = useState<TxnEntry[]>([])
  const [txnsLoading, setTxnsLoading] = useState(true)

  const token = useMemo(() => getToken(), [])

  useEffect(() => {
    if (!token) {
      Promise.resolve().then(() => { setBalanceLoading(false) }).catch(() => null)
      return
    }
    fetch(`${API}/api/v1/wallet/balance`, { headers: authHeaders() })
      .then((r) => r.json() as Promise<BalanceResponse>)
      .then((d) => { setBalance(d.balance) })
      .catch(() => { setBalance(null) })
      .finally(() => { setBalanceLoading(false) })
  }, [token])

  useEffect(() => {
    if (!token) {
      Promise.resolve().then(() => { setTxnsLoading(false) }).catch(() => null)
      return
    }
    fetch(`${API}/api/v1/transactions`, { headers: authHeaders() })
      .then((r) => r.json() as Promise<TxnEntry[]>)
      .then((d) => { setTxns(Array.isArray(d) ? d : []) })
      .catch(() => { setTxns([]) })
      .finally(() => { setTxnsLoading(false) })
  }, [token])

  const isLoggedIn = Boolean(token)

  return (
    <div className="grid gap-6">
      {/* Header */}
      <div className="flex flex-col gap-2 md:flex-row md:items-end md:justify-between">
        <div>
          <h1 className="text-3xl font-extrabold tracking-tight">Dashboard</h1>
          <p className="text-sm text-(--gopay-muted)">
            {isLoggedIn ? 'Your GoPay wallet — live data.' : 'Log in to see your balance and transactions.'}
          </p>
        </div>
        {isLoggedIn ? (
          <Link to="/send">
            <Button className="gap-2">
              <span>↗</span> Send Money
            </Button>
          </Link>
        ) : null}
      </div>

      {/* Top stats */}
      <div className="grid gap-6 md:grid-cols-3">
        <Card
          subtitle="Wallet balance"
          title={balance !== null ? formatINR(balance) : '—'}
          loading={balanceLoading}
        >
          {!isLoggedIn ? (
            <div className="text-sm text-(--gopay-muted)">
              <Link to="/login" className="font-semibold text-(--gopay-primary)">Log in</Link> to see your balance.
            </div>
          ) : (
            <div className="text-sm text-(--gopay-muted)">Updated just now</div>
          )}
        </Card>

        <Card subtitle="Quick action" title="Send Money">
          <Link to="/send">
            <Button className="w-full mt-1">↗ Send now</Button>
          </Link>
        </Card>

        <Card subtitle="Transactions" title={txnsLoading ? '…' : `${txns.length} total`}>
          <div className="text-sm text-(--gopay-muted)">
            {txns.filter((t) => t.direction === 'SENT').length} sent ·{' '}
            {txns.filter((t) => t.direction === 'RECEIVED').length} received
          </div>
        </Card>
      </div>

      {/* Transaction history */}
      <section className="rounded-3xl border border-(--gopay-border) bg-white/70 p-6 shadow-(--gopay-shadow)">
        <div className="mb-4 flex items-center justify-between">
          <div>
            <div className="text-sm font-semibold text-(--gopay-muted)">Payments</div>
            <div className="mt-1 text-xl font-extrabold tracking-tight">Transaction history</div>
          </div>
          <Link to="/send">
            <Button variant="ghost" className="text-sm">+ New</Button>
          </Link>
        </div>

        {!isLoggedIn ? (
          <div className="rounded-2xl border border-(--gopay-border) bg-white p-6 text-center text-sm text-(--gopay-muted)">
            <Link to="/login" className="font-semibold text-(--gopay-primary)">Log in</Link> to see your transactions.
          </div>
        ) : txnsLoading ? (
          <div className="space-y-3">
            {[1, 2, 3].map((i) => (
              <div key={i} className="h-16 animate-pulse rounded-2xl bg-gray-100" />
            ))}
          </div>
        ) : txns.length === 0 ? (
          <div className="rounded-2xl border border-(--gopay-border) bg-white p-8 text-center">
            <div className="text-2xl">💸</div>
            <div className="mt-2 text-sm font-semibold">No transactions yet</div>
            <div className="mt-1 text-sm text-(--gopay-muted)">Send money to get started.</div>
            <Link to="/send">
              <Button className="mt-4">↗ Send Money</Button>
            </Link>
          </div>
        ) : (
          <div className="grid gap-3">
            {txns.map((t) => (
              <div
                key={t.id}
                className="flex items-center justify-between rounded-2xl border border-(--gopay-border) bg-white px-4 py-3"
              >
                <div className="flex items-center gap-3">
                  <div
                    className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-full text-sm font-bold ${
                      t.direction === 'SENT'
                        ? 'bg-red-50 text-red-600'
                        : 'bg-green-50 text-green-600'
                    }`}
                  >
                    {t.direction === 'SENT' ? '↗' : '↙'}
                  </div>
                  <div>
                    <div className="text-sm font-semibold">{t.counterpartyName}</div>
                    <div className="text-xs text-(--gopay-muted)">
                      {t.counterpartyIdentifier}
                      {t.note ? ` · ${t.note}` : ''}
                    </div>
                    <div className="text-xs text-(--gopay-muted)">{formatDate(t.createdAt)}</div>
                  </div>
                </div>
                <div className="text-right">
                  <div
                    className={`text-sm font-extrabold ${
                      t.direction === 'SENT' ? 'text-red-600' : 'text-green-600'
                    }`}
                  >
                    {t.direction === 'SENT' ? '−' : '+'} {formatINR(t.amount)}
                  </div>
                  <div className="text-xs text-(--gopay-muted)">{t.status}</div>
                </div>
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  )
}
