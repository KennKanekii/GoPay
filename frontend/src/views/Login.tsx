import { useMemo, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { Button } from '../ui/Button'

export function Login() {
  const navigate = useNavigate()

  const [identifier, setIdentifier] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)

  const apiBaseUrl = useMemo(() => {
    return import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8080'
  }, [])

  const loginUrl = `${apiBaseUrl}/api/v1/auth/login`

  return (
    <div className="mx-auto max-w-md">
      <div className="rounded-3xl border border-(--gopay-border) bg-white/75 p-6 shadow-(--gopay-shadow)">
        <h1 className="text-3xl font-extrabold tracking-tight">Welcome back</h1>
        <p className="mt-2 text-sm text-(--gopay-muted)">
          This is a starter UI. Next we’ll wire real auth (JWT) via your Spring Cloud Gateway.
        </p>

        <form
          className="mt-6 grid gap-4"
          onSubmit={(e) => {
            e.preventDefault()
            setError(null)
            setSubmitting(true)

            const body = {
              identifier: identifier.trim(),
              password,
            }

            fetch(loginUrl, {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
              },
              body: JSON.stringify(body),
            })
              .then(async (res) => {
                const data: unknown = await res.json().catch(() => null)
                if (!res.ok) {
                  const message = (data as { error?: unknown } | null)?.error
                  throw new Error(typeof message === 'string' ? message : 'Unable to log in.')
                }
                return data as { token: string }
              })
              .then((data) => {
                localStorage.setItem('gopay_token', data.token)
                window.dispatchEvent(new Event('gopay-auth-changed'))
                navigate('/dashboard')
              })
              .catch((err: Error) => {
                setError(err.message)
              })
              .finally(() => setSubmitting(false))
          }}
        >
          <label className="grid gap-2 text-sm font-semibold">
            <span>Phone / Email</span>
            <input
              className="h-11 rounded-xl border border-(--gopay-border) bg-white px-3 text-sm outline-none focus:ring-4 focus:ring-[rgba(18,179,165,0.20)]"
              placeholder="you@example.com"
              required
              value={identifier}
              onChange={(e) => setIdentifier(e.target.value)}
            />
          </label>

          <label className="grid gap-2 text-sm font-semibold">
            <span>Password</span>
            <input
              type="password"
              className="h-11 rounded-xl border border-(--gopay-border) bg-white px-3 text-sm outline-none focus:ring-4 focus:ring-[rgba(18,179,165,0.20)]"
              placeholder="••••••••"
              required
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
          </label>

          {error ? (
            <div className="mt-2 rounded-xl border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-800">
              {error}
            </div>
          ) : null}

          <Button type="submit" className="h-11" disabled={submitting}>
            {submitting ? 'Logging in...' : 'Log in'}
          </Button>

          <div className="text-center text-sm text-(--gopay-muted)">
            Don’t have an account?{' '}
            <Link className="font-semibold text-(--gopay-primary)" to="/signup">
              Create one
            </Link>
          </div>
        </form>
      </div>
    </div>
  )
}

