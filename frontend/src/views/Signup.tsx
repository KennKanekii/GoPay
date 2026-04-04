import { useMemo, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { Button } from '../ui/Button'

export function Signup() {
  const navigate = useNavigate()

  const [name, setName] = useState('')
  const [identifier, setIdentifier] = useState('')
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)

  const apiBaseUrl = useMemo(() => {
    return import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8080'
  }, [])

  const signupUrl = `${apiBaseUrl}/api/v1/auth/signup`

  return (
    <div className="mx-auto max-w-md">
      <div className="rounded-3xl border border-(--gopay-border) bg-white/75 p-6 shadow-(--gopay-shadow)">
        <h1 className="text-3xl font-extrabold tracking-tight">Create your account</h1>
        <p className="mt-2 text-sm text-(--gopay-muted)">
          This is starter auth. Your account will be stored locally for this demo.
        </p>

        {error ? (
          <div className="mt-4 rounded-xl border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-800">
            {error}
          </div>
        ) : null}

        <form
          className="mt-6 grid gap-4"
          onSubmit={async (e) => {
            e.preventDefault()
            setError(null)

            const trimmedName = name.trim()
            const trimmedIdentifier = identifier.trim()

            if (trimmedName.length < 2) {
              setError('Please enter your name.')
              return
            }
            if (trimmedIdentifier.length < 3) {
              setError('Please enter a valid phone/email.')
              return
            }
            if (password.length < 6) {
              setError('Password must be at least 6 characters.')
              return
            }
            if (password !== confirmPassword) {
              setError('Passwords do not match.')
              return
            }

            setSubmitting(true)
            try {
              const res = await fetch(signupUrl, {
                method: 'POST',
                headers: {
                  'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                  name: trimmedName,
                  identifier: trimmedIdentifier,
                  password,
                }),
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
          }}
        >
          <label className="grid gap-2 text-sm font-semibold">
            <span>Your name</span>
            <input
              className="h-11 rounded-xl border border-(--gopay-border) bg-white px-3 text-sm outline-none focus:ring-4 focus:ring-[rgba(18,179,165,0.20)]"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Harsh"
              required
              autoComplete="name"
            />
          </label>

          <label className="grid gap-2 text-sm font-semibold">
            <span>Phone / Email</span>
            <input
              className="h-11 rounded-xl border border-(--gopay-border) bg-white px-3 text-sm outline-none focus:ring-4 focus:ring-[rgba(18,179,165,0.20)]"
              value={identifier}
              onChange={(e) => setIdentifier(e.target.value)}
              placeholder="you@example.com"
              required
              autoComplete="username"
            />
          </label>

          <label className="grid gap-2 text-sm font-semibold">
            <span>Password</span>
            <input
              type="password"
              className="h-11 rounded-xl border border-(--gopay-border) bg-white px-3 text-sm outline-none focus:ring-4 focus:ring-[rgba(18,179,165,0.20)]"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="••••••••"
              required
              autoComplete="new-password"
            />
          </label>

          <label className="grid gap-2 text-sm font-semibold">
            <span>Confirm password</span>
            <input
              type="password"
              className="h-11 rounded-xl border border-(--gopay-border) bg-white px-3 text-sm outline-none focus:ring-4 focus:ring-[rgba(18,179,165,0.20)]"
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              placeholder="••••••••"
              required
              autoComplete="new-password"
            />
          </label>

          <Button type="submit" className="h-11" disabled={submitting}>
            {submitting ? 'Creating...' : 'Create account'}
          </Button>

          <div className="text-center text-sm text-(--gopay-muted)">
            Already have an account?{' '}
            <Link className="font-semibold text-(--gopay-primary)" to="/login">
              Log in
            </Link>
          </div>
        </form>
      </div>
    </div>
  )
}

