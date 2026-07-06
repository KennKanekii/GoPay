import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { Button } from '../ui/Button'

const INPUT_CLS =
  'h-11 w-full rounded-xl border border-(--gopay-border) bg-white px-3 text-sm outline-none focus:ring-4 focus:ring-[rgba(18,179,165,0.20)] disabled:bg-gray-50 disabled:text-(--gopay-muted)'

type Profile = {
  id: string
  name: string
  identifier: string
  balance: number
  mobileNumber?: string
  vpa?: string
  bankAccount?: string
  ifscCode?: string
}

type IfscResult = { is_valid: boolean; bank_name?: string; risk: string }

const API = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8080'

function getToken() {
  return localStorage.getItem('gopay_token') ?? ''
}

function authHeaders(extra?: Record<string, string>) {
  return { Authorization: `Bearer ${getToken()}`, 'Content-Type': 'application/json', ...extra }
}

export function Profile() {
  const [profile, setProfile]     = useState<Profile | null>(null)
  const [loading, setLoading]     = useState(true)
  const [editing, setEditing]     = useState(false)
  const [saving, setSaving]       = useState(false)
  const [success, setSuccess]     = useState(false)
  const [error, setError]         = useState<string | null>(null)

  const [editName, setEditName]           = useState('')
  const [editMobile, setEditMobile]       = useState('')
  const [editVpa, setEditVpa]             = useState('')
  const [editBank, setEditBank]           = useState('')
  const [editIfsc, setEditIfsc]           = useState('')
  const [ifscResult, setIfscResult]       = useState<IfscResult | null>(null)
  const [vpaResult, setVpaResult]         = useState<{ is_spoof: boolean; risk_score: number } | null>(null)

  const token = useMemo(() => getToken(), [])

  const load = async () => {
    setLoading(true)
    try {
      const res = await fetch(`${API}/api/v1/me`, { headers: authHeaders() })
      if (res.ok) {
        const p: Profile = await res.json()
        setProfile(p)
        setEditName(p.name ?? '')
        setEditMobile(p.mobileNumber ?? '')
        setEditVpa(p.vpa ?? '')
        setEditBank(p.bankAccount ?? '')
        setEditIfsc(p.ifscCode ?? '')
      }
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [])

  const validateIfsc = async () => {
    setIfscResult(null)
    if (!editIfsc.trim()) return
    const res = await fetch(`${API.replace('8080', '5002')}/ifsc-validate`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ifsc: editIfsc.trim().toUpperCase() }),
    }).catch(() => null)
    if (res?.ok) setIfscResult(await res.json())
  }

  const validateVpa = async () => {
    setVpaResult(null)
    if (!editVpa.trim()) return
    const res = await fetch(`${API.replace('8080', '5002')}/vpa-check`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ vpa: editVpa.trim().toLowerCase() }),
    }).catch(() => null)
    if (res?.ok) setVpaResult(await res.json())
  }

  const validateFields = (mobile: string): string | null => {
    if (mobile && mobile.length !== 10) return 'Mobile number must be 10 digits.'
    if (editVpa.trim() && !editVpa.includes('@')) return 'UPI ID must be username@handle.'
    if (editIfsc.trim() && !/^[A-Z]{4}0[A-Z0-9]{6}$/.test(editIfsc.trim().toUpperCase())) {
      return 'IFSC format invalid. Example: HDFC0001234'
    }
    return null
  }

  const handleSave = async (e: React.SyntheticEvent) => {
    e.preventDefault()
    setError(null)
    const mobile = editMobile.replaceAll(/\D/g, '')
    const validationError = validateFields(mobile)
    if (validationError) { setError(validationError); return }

    setSaving(true)
    try {
      const body: Record<string, string> = { name: editName }
      if (mobile) body.mobileNumber = mobile
      if (editVpa.trim()) body.vpa = editVpa.trim().toLowerCase()
      if (editBank.trim()) body.bankAccount = editBank.trim()
      if (editIfsc.trim()) body.ifscCode = editIfsc.trim().toUpperCase()

      const res = await fetch(`${API}/api/v1/me`, {
        method: 'PATCH', headers: authHeaders(), body: JSON.stringify(body),
      })
      const data: unknown = await res.json().catch(() => null)
      if (!res.ok) {
        const msg = (data as { error?: unknown } | null)?.error
        setError(typeof msg === 'string' ? msg : 'Failed to save.'); return
      }
      setProfile(data as Profile)
      setEditing(false)
      setSuccess(true)
      setTimeout(() => setSuccess(false), 3000)
    } finally {
      setSaving(false)
    }
  }

  if (token && loading) {
    return (
      <div className="mx-auto max-w-lg animate-pulse rounded-3xl border border-(--gopay-border) bg-white/75 p-6">
        <div className="h-6 w-32 rounded bg-gray-200" />
        <div className="mt-4 h-4 w-48 rounded bg-gray-200" />
      </div>
    )
  }

  if (token && profile === null && !loading) {
    return (
      <div className="mx-auto max-w-lg rounded-3xl border border-(--gopay-border) bg-white/75 p-6 text-center text-sm text-red-600">
        Could not load profile. Please try again.
      </div>
    )
  }

  if (token.length === 0) {
    return (
      <div className="mx-auto max-w-lg rounded-3xl border border-(--gopay-border) bg-white/75 p-6 text-center">
        <p className="text-sm text-(--gopay-muted)">
          <Link className="font-semibold text-(--gopay-primary) underline" to="/login">Log in</Link>{' '}
          to view your profile.
        </p>
      </div>
    )
  }

  if (profile === null) {
    return (
      <div className="mx-auto max-w-lg rounded-3xl border border-(--gopay-border) bg-white/75 p-6 text-center text-sm text-red-600">
        Could not load profile. Please try again.
      </div>
    )
  }

  return (
    <div className="mx-auto max-w-lg space-y-5">
      <div className="rounded-3xl border border-(--gopay-border) bg-white/75 p-6 shadow-(--gopay-shadow)">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-extrabold tracking-tight">{profile.name}</h1>
            <p className="mt-1 text-sm text-(--gopay-muted)">{profile.identifier}</p>
          </div>
          <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-(--gopay-primary) text-xl font-extrabold text-white">
            {profile.name.charAt(0).toUpperCase()}
          </div>
        </div>

        {success && (
          <div className="mt-4 rounded-xl border border-green-200 bg-green-50 px-3 py-2 text-sm font-medium text-green-800">
            Profile saved successfully.
          </div>
        )}

        {editing ? null : (
          <div className="mt-6 space-y-4">
            <Row label="Mobile" value={profile.mobileNumber ? `+91 ${profile.mobileNumber}` : '—'} />
            <Row label="UPI ID / VPA" value={profile.vpa ?? '—'} />
            <Row label="Bank account" value={profile.bankAccount ?? '—'} mono />
            <Row label="IFSC code" value={profile.ifscCode ?? '—'} mono />

            <Button className="mt-4 h-10 w-full" onClick={() => setEditing(true)}>
              Edit profile
            </Button>
          </div>
        )}
        {editing && (
          <form className="mt-6 grid gap-4" onSubmit={handleSave}>
            {error && (
              <div className="rounded-xl border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-800">
                {error}
              </div>
            )}

            <label className="grid gap-2 text-sm font-semibold">
              <span>Full name</span>
              <input className={INPUT_CLS} value={editName} onChange={(e) => setEditName(e.target.value)} required />
            </label>

            <label className="grid gap-2 text-sm font-semibold">
              <span>Mobile number</span>
              <div className="flex items-center gap-2">
                <span className="flex h-11 items-center rounded-xl border border-(--gopay-border) bg-white/60 px-3 text-sm font-semibold text-(--gopay-muted)">+91</span>
                <input className={`${INPUT_CLS} flex-1`} value={editMobile}
                  onChange={(e) => setEditMobile(e.target.value.replaceAll(/\D/g, '').slice(0, 10))}
                  inputMode="numeric" maxLength={10} placeholder="9876543210" />
              </div>
            </label>

            <label className="grid gap-2 text-sm font-semibold">
              <span>UPI ID / VPA</span>
              <div className="flex gap-2">
                <input className={INPUT_CLS} value={editVpa}
                  onChange={(e) => { setEditVpa(e.target.value); setVpaResult(null) }}
                  placeholder="yourname@ybl" spellCheck={false} />
                <button type="button"
                  className="shrink-0 rounded-xl border border-(--gopay-border) px-3 text-xs font-semibold hover:bg-gray-50"
                  onClick={validateVpa}>Check</button>
              </div>
              {vpaResult !== null && (
                <p className={`text-xs ${vpaResult.is_spoof ? 'text-red-600 font-semibold' : 'text-green-700'}`}>
                  {vpaResult.is_spoof
                    ? `Suspicious VPA — spoofing risk ${vpaResult.risk_score}/100. Please verify.`
                    : `VPA looks legitimate (risk ${vpaResult.risk_score}/100).`}
                </p>
              )}
            </label>

            <label className="grid gap-2 text-sm font-semibold">
              <span>Bank account number</span>
              <input className={INPUT_CLS} value={editBank}
                onChange={(e) => setEditBank(e.target.value)} inputMode="numeric" placeholder="00112233445566" />
            </label>

            <label className="grid gap-2 text-sm font-semibold">
              <span>IFSC code</span>
              <div className="flex gap-2">
                <input className={INPUT_CLS} value={editIfsc}
                  onChange={(e) => { setEditIfsc(e.target.value.toUpperCase()); setIfscResult(null) }}
                  placeholder="HDFC0001234" maxLength={11} spellCheck={false} />
                <button type="button"
                  className="shrink-0 rounded-xl border border-(--gopay-border) px-3 text-xs font-semibold hover:bg-gray-50"
                  onClick={validateIfsc}>Validate</button>
              </div>
              {ifscResult !== null && (
                <p className={`text-xs ${ifscResult.is_valid ? 'text-green-700' : 'text-red-600 font-semibold'}`}>
                  {ifscResult.is_valid
                    ? `Valid IFSC — ${ifscResult.bank_name ?? 'registered bank'}`
                    : `Invalid IFSC (risk: ${ifscResult.risk}). Please check the code.`}
                </p>
              )}
            </label>

            <div className="flex gap-3 pt-2">
              <Button type="submit" className="h-10 flex-1" disabled={saving}>
                {saving ? 'Saving…' : 'Save changes'}
              </Button>
              <button type="button"
                className="h-10 rounded-xl border border-(--gopay-border) px-4 text-sm font-semibold hover:bg-gray-50"
                onClick={() => { setEditing(false); setError(null) }}>
                Cancel
              </button>
            </div>
          </form>
        )}
      </div>

      <div className="rounded-3xl border border-(--gopay-border) bg-white/70 p-5">
        <p className="text-xs text-(--gopay-muted) text-center">
          Your banking details are stored locally and used only for fraud detection validation.
          GoPay never transmits them to third parties.
        </p>
      </div>
    </div>
  )
}

function Row({ label, value, mono }: Readonly<{ label: string; value: string; mono?: boolean }>) {
  return (
    <div className="flex items-center justify-between border-b border-(--gopay-border) pb-3 last:border-0 last:pb-0">
      <span className="text-sm text-(--gopay-muted)">{label}</span>
      <span className={`text-sm font-semibold ${mono ? 'font-mono' : ''}`}>{value}</span>
    </div>
  )
}
