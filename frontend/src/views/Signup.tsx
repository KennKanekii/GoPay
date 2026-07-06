import { useMemo, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { Button } from '../ui/Button'

const INPUT_CLS =
  'h-11 rounded-xl border border-(--gopay-border) bg-white px-3 text-sm outline-none focus:ring-4 focus:ring-[rgba(18,179,165,0.20)]'

export function Signup() {
  const navigate = useNavigate()

  const [name, setName]               = useState('')
  const [identifier, setIdentifier]   = useState('')
  const [mobile, setMobile]           = useState('')
  const [vpa, setVpa]                 = useState('')
  const [bankAccount, setBankAccount] = useState('')
  const [ifscCode, setIfscCode]       = useState('')
  const [password, setPassword]       = useState('')
  const [confirmPw, setConfirmPw]     = useState('')
  const [showBanking, setShowBanking] = useState(false)
  const [error, setError]             = useState<string | null>(null)
  const [submitting, setSubmitting]   = useState(false)

  const apiBaseUrl = useMemo(() => import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8080', [])

  const handleSubmit = async (e: React.SyntheticEvent) => {
    e.preventDefault()
    setError(null)

    const trimName   = name.trim()
    const trimId     = identifier.trim()
    const trimMobile = mobile.replaceAll(/\D/g, '')

    if (trimName.length < 2)   { setError('Please enter your name.'); return }
    if (trimId.length < 3)     { setError('Please enter a valid email.'); return }
    if (trimMobile.length !== 10) { setError('Mobile number must be 10 digits.'); return }
    if (password.length < 6)   { setError('Password must be at least 6 characters.'); return }
    if (password !== confirmPw) { setError('Passwords do not match.'); return }

    if (vpa.trim() && !vpa.includes('@')) {
      setError('UPI ID must be in format username@handle (e.g. name@ybl).')
      return
    }
    if (ifscCode.trim() && !/^[A-Z]{4}0[A-Z0-9]{6}$/.test(ifscCode.trim().toUpperCase())) {
      setError('IFSC code must be 11 characters: 4 letters + 0 + 6 alphanumeric (e.g. HDFC0001234).')
      return
    }

    setSubmitting(true)
    try {
      const body: Record<string, string> = {
        name: trimName, identifier: trimId, password,
        mobileNumber: trimMobile,
      }
      if (vpa.trim())         body.vpa         = vpa.trim().toLowerCase()
      if (bankAccount.trim()) body.bankAccount  = bankAccount.trim()
      if (ifscCode.trim())    body.ifscCode     = ifscCode.trim().toUpperCase()

      const res = await fetch(`${apiBaseUrl}/api/v1/auth/signup`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })

      const data: unknown = await res.json().catch(() => null)
      if (!res.ok) {
        const message = (data as { error?: unknown } | null)?.error
        setError(typeof message === 'string' ? message : 'Unable to create account.')
        return
      }
      navigate('/login')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="mx-auto max-w-md">
      <div className="rounded-3xl border border-(--gopay-border) bg-white/75 p-6 shadow-(--gopay-shadow)">
        <h1 className="text-3xl font-extrabold tracking-tight">Create your account</h1>
        <p className="mt-2 text-sm text-(--gopay-muted)">
          Fill in your details to get started with GoPay.
        </p>

        {error && (
          <div className="mt-4 rounded-xl border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-800">
            {error}
          </div>
        )}

        <form className="mt-6 grid gap-4" onSubmit={handleSubmit}>
          {/* Name */}
          <label className="grid gap-2 text-sm font-semibold">
            <span>Full name</span>
            <input className={INPUT_CLS} value={name} onChange={(e) => setName(e.target.value)}
              placeholder="Harsh Sharma" required autoComplete="name" />
          </label>

          {/* Email */}
          <label className="grid gap-2 text-sm font-semibold">
            <span>Email</span>
            <input className={INPUT_CLS} value={identifier}
              onChange={(e) => setIdentifier(e.target.value)}
              placeholder="you@example.com" required autoComplete="email" type="email" />
          </label>

          {/* Mobile */}
          <label className="grid gap-2 text-sm font-semibold">
            <span>Mobile number <span className="font-normal text-(--gopay-muted)">(required for UPI)</span></span>
            <div className="flex items-center gap-2">
              <span className="flex h-11 items-center rounded-xl border border-(--gopay-border) bg-white/60 px-3 text-sm font-semibold text-(--gopay-muted)">
                +91
              </span>
              <input className={`${INPUT_CLS} flex-1`} value={mobile}
                onChange={(e) => setMobile(e.target.value.replaceAll(/\D/g, '').slice(0, 10))}
                placeholder="9876543210" required inputMode="numeric" maxLength={10} />
            </div>
          </label>

          {/* Password */}
          <label className="grid gap-2 text-sm font-semibold">
            <span>Password</span>
            <input type="password" className={INPUT_CLS} value={password}
              onChange={(e) => setPassword(e.target.value)} placeholder="••••••••"
              required autoComplete="new-password" />
          </label>

          <label className="grid gap-2 text-sm font-semibold">
            <span>Confirm password</span>
            <input type="password" className={INPUT_CLS} value={confirmPw}
              onChange={(e) => setConfirmPw(e.target.value)} placeholder="••••••••"
              required autoComplete="new-password" />
          </label>

          {/* Optional banking section toggle */}
          <button
            type="button"
            className="flex items-center gap-2 text-sm font-semibold text-(--gopay-primary)"
            onClick={() => setShowBanking((v) => !v)}
          >
            <span className="text-lg leading-none">{showBanking ? '−' : '+'}</span>
            {showBanking ? 'Hide' : 'Add'} banking details
            <span className="font-normal text-(--gopay-muted)">(optional — for IFSC & UPI validation)</span>
          </button>

          {showBanking && (
            <div className="grid gap-4 rounded-2xl border border-(--gopay-border) bg-white/50 p-4">
              {/* UPI ID */}
              <label className="grid gap-2 text-sm font-semibold">
                <span>UPI ID / VPA</span>
                <input className={INPUT_CLS} value={vpa}
                  onChange={(e) => setVpa(e.target.value)}
                  placeholder="yourname@ybl" autoComplete="off"
                  spellCheck={false} />
                <p className="text-xs font-normal text-(--gopay-muted)">
                  e.g. name@ybl, mobile@paytm, email@oksbi
                </p>
              </label>

              {/* Bank account */}
              <label className="grid gap-2 text-sm font-semibold">
                <span>Bank account number</span>
                <input className={INPUT_CLS} value={bankAccount}
                  onChange={(e) => setBankAccount(e.target.value)}
                  placeholder="00112233445566" inputMode="numeric" />
              </label>

              {/* IFSC */}
              <label className="grid gap-2 text-sm font-semibold">
                <span>IFSC code</span>
                <input className={INPUT_CLS} value={ifscCode}
                  onChange={(e) => setIfscCode(e.target.value.toUpperCase())}
                  placeholder="HDFC0001234" maxLength={11} spellCheck={false} />
                <p className="text-xs font-normal text-(--gopay-muted)">
                  11 characters: bank code + 0 + branch code
                </p>
              </label>
            </div>
          )}

          <Button type="submit" className="h-11" disabled={submitting}>
            {submitting ? 'Creating account…' : 'Create account'}
          </Button>

          <div className="text-center text-sm text-(--gopay-muted)">
            Already have an account?{' '}
            <Link className="font-semibold text-(--gopay-primary)" to="/login">Log in</Link>
          </div>
        </form>
      </div>
    </div>
  )
}
