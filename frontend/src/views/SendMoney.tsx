import { useState } from 'react'
import { Link } from 'react-router-dom'
import { Button } from '../ui/Button'

const API = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8080'

type SendResponse = {
  ok: boolean
  transactionId: string
  amount: number
  recipientName: string
  recipientIdentifier: string
  newBalance: number
  status: string
  createdAt: string
  fraudScore: number
  fraudRiskLevel: string
}

function formatINR(amount: number) {
  return new Intl.NumberFormat('en-IN', { style: 'currency', currency: 'INR', maximumFractionDigits: 2 }).format(amount)
}

function formatDate(iso: string) {
  return new Date(iso).toLocaleString('en-IN', { dateStyle: 'medium', timeStyle: 'short' })
}

function RiskCheckRow({ level, score }: Readonly<{ level: string; score: number }>) {
  let cls = 'bg-yellow-100 text-yellow-700'
  if (level === 'CRITICAL') cls = 'bg-red-100 text-red-700'
  else if (level === 'HIGH') cls = 'bg-orange-100 text-orange-700'
  return (
    <div className="flex justify-between items-center border-t border-(--gopay-border) pt-2 mt-1">
      <span className="text-(--gopay-muted)">Risk check</span>
      <span className={`rounded-full px-2 py-0.5 text-xs font-bold ${cls}`}>
        {level} · score {score}
      </span>
    </div>
  )
}

export function SendMoney() {
  const [recipient, setRecipient] = useState('')
  const [amount, setAmount] = useState('')
  const [note, setNote] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<SendResponse | null>(null)

  const token = localStorage.getItem('gopay_token')

  const inputClass =
    'h-11 w-full rounded-xl border border-(--gopay-border) bg-white px-3 text-sm outline-none focus:ring-4 focus:ring-[rgba(18,179,165,0.20)]'

  const reset = () => {
    setRecipient('')
    setAmount('')
    setNote('')
    setError(null)
    setSuccess(null)
  }

  const handleSubmit = async (e: React.SyntheticEvent) => {
    e.preventDefault()
    setError(null)

    const parsedAmount = Number.parseFloat(amount)
    if (!recipient.trim()) { setError('Recipient email/phone is required.'); return }
    if (Number.isNaN(parsedAmount) || parsedAmount <= 0) { setError('Enter a valid amount greater than Rs.0.'); return }
    if (parsedAmount > 100_000) { setError('Amount cannot exceed ₹1,00,000 per transaction.'); return }

    setSubmitting(true)
    try {
      const res = await fetch(`${API}/api/v1/transactions/send`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token ?? ''}`,
        },
        body: JSON.stringify({ recipientIdentifier: recipient.trim(), amount: parsedAmount, note: note.trim() }),
      })
      const data: unknown = await res.json().catch(() => null)
      if (!res.ok) {
        const msg = (data as { error?: unknown } | null)?.error
        setError(typeof msg === 'string' ? msg : 'Something went wrong.')
        return
      }
      setSuccess(data as SendResponse)
    } catch {
      setError('Unable to reach the server. Please try again.')
    } finally {
      setSubmitting(false)
    }
  }

  // ---- Success state ----
  if (success) {
    return (
      <div className="mx-auto max-w-md">
        <div className="rounded-3xl border border-(--gopay-border) bg-white/75 p-8 shadow-(--gopay-shadow) text-center">
          <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-full bg-[rgba(18,179,165,0.15)] text-4xl">
            ✓
          </div>
          <h1 className="mt-4 text-2xl font-extrabold tracking-tight text-(--gopay-fg)">
            Money Sent!
          </h1>
          <p className="mt-1 text-sm text-(--gopay-muted)">Transaction ID: {success.transactionId}</p>

          <div className="mt-6 rounded-2xl border border-(--gopay-border) bg-white p-4 text-left grid gap-2 text-sm">
            <div className="flex justify-between">
              <span className="text-(--gopay-muted)">To</span>
              <span className="font-semibold">{success.recipientName}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-(--gopay-muted)">Email/Phone</span>
              <span className="font-semibold">{success.recipientIdentifier}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-(--gopay-muted)">Amount</span>
              <span className="font-extrabold text-red-600">− {formatINR(success.amount)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-(--gopay-muted)">New Balance</span>
              <span className="font-extrabold text-(--gopay-primary)">{formatINR(success.newBalance)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-(--gopay-muted)">Status</span>
              <span className="font-semibold text-green-600">{success.status}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-(--gopay-muted)">Time</span>
              <span className="font-semibold">{formatDate(success.createdAt)}</span>
            </div>
            {success.fraudRiskLevel && success.fraudRiskLevel !== 'LOW' && (
              <RiskCheckRow level={success.fraudRiskLevel} score={success.fraudScore} />
            )}
          </div>

          <div className="mt-6 flex gap-3">
            <Button className="flex-1" onClick={reset}>Send again</Button>
            <Link to="/dashboard" className="flex-1">
              <Button variant="ghost" className="w-full">Dashboard</Button>
            </Link>
          </div>
        </div>
      </div>
    )
  }

  // ---- Form state ----
  return (
    <div className="mx-auto max-w-md">
      <div className="rounded-3xl border border-(--gopay-border) bg-white/75 p-6 shadow-(--gopay-shadow)">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-[rgba(18,179,165,0.12)] text-xl">
            ↗
          </div>
          <div>
            <h1 className="text-2xl font-extrabold tracking-tight">Send Money</h1>
            <p className="text-sm text-(--gopay-muted)">Transfer instantly to any GoPay user.</p>
          </div>
        </div>

        {token && (
          <form className="mt-6 grid gap-4" onSubmit={(e) => { void handleSubmit(e) }}>
            <label className="grid gap-2 text-sm font-semibold">
              <span>Recipient Email / Phone</span>
              <input
                className={inputClass}
                placeholder="friend@example.com"
                value={recipient}
                onChange={(e) => setRecipient(e.target.value)}
                required
                autoComplete="off"
              />
            </label>

            <label className="grid gap-2 text-sm font-semibold">
              <span>Amount (₹)</span>
              <input
                type="number"
                min="1"
                max="100000"
                step="1"
                className={inputClass}
                placeholder="500"
                value={amount}
                onChange={(e) => setAmount(e.target.value)}
                required
              />
            </label>

            <label className="grid gap-2 text-sm font-semibold">
              <span>Note <span className="font-normal text-(--gopay-muted)">(optional)</span></span>
              <input
                className={inputClass}
                placeholder="For dinner, rent, etc."
                value={note}
                onChange={(e) => setNote(e.target.value)}
                maxLength={120}
              />
            </label>

            {error ? (
              <div className={`rounded-xl border px-3 py-3 text-sm ${
                error.startsWith('Transaction blocked')
                  ? 'border-red-300 bg-red-50 text-red-800'
                  : 'border-red-200 bg-red-50 text-red-800'
              }`}>
                {error.startsWith('Transaction blocked') && (
                  <div className="mb-1 font-bold flex items-center gap-1.5">
                    🚫 Transaction Blocked by Fraud Shield
                  </div>
                )}
                {error}
                {error.startsWith('Transaction blocked') && (
                  <div className="mt-2 text-xs text-red-600">
                    This decision is based on velocity rules and ML analysis.{' '}
                    <Link to="/fraud" className="font-semibold underline">View your risk report →</Link>
                  </div>
                )}
              </div>
            ) : null}

            <Button type="submit" className="h-11" disabled={submitting}>
              {submitting ? 'Sending…' : 'Send Money'}
            </Button>

            <Link to="/dashboard" className="text-center text-sm text-(--gopay-muted) hover:text-(--gopay-fg)">
              ← Back to Dashboard
            </Link>
          </form>
        )}
        {!token && (
          <div className="mt-6 rounded-xl border border-yellow-200 bg-yellow-50 px-4 py-3 text-sm text-yellow-800">
            You need to{' '}
            <Link className="font-semibold underline" to="/login">log in</Link>{' '}
            before sending money.
          </div>
        )}
      </div>
    </div>
  )
}
