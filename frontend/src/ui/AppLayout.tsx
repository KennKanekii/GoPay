import { useEffect, useMemo, useState } from 'react'
import { Link, NavLink, Outlet, useNavigate } from 'react-router-dom'
import { Button } from './Button'
import { GoPayMark } from './GoPayMark'

type MeResponse = {
  id: string
  name: string
  identifier: string
}

function getInitials(name: string) {
  return name
    .split(' ')
    .map((p) => p.trim())
    .filter(Boolean)
    .slice(0, 2)
    .map((p) => p[0]?.toUpperCase())
    .join('')
}

function TopNav({
  me,
  loading,
  onLogout,
}: Readonly<{ me: MeResponse | null; loading: boolean; onLogout: () => void }>) {
  const navLinkClass = ({ isActive }: { isActive: boolean }) =>
    [
      'rounded-lg px-3 py-2 text-sm font-semibold transition',
      isActive
        ? 'bg-[color:rgba(18,179,165,0.12)] text-[color:var(--gopay-fg)]'
        : 'text-[color:var(--gopay-muted)] hover:bg-[color:rgba(18,179,165,0.08)] hover:text-[color:var(--gopay-fg)]',
    ].join(' ')

  let rightContent: JSX.Element | null = null
  if (!loading && me) {
    rightContent = (
      <>
        <Link to="/dashboard">
          <div className="flex items-center gap-2 rounded-xl border border-(--gopay-border) bg-white/60 px-3 py-2">
            <div className="flex h-7 w-7 items-center justify-center rounded-full bg-[rgba(18,179,165,0.18)] text-sm font-semibold text-(--gopay-fg)">
              {getInitials(me.name)}
            </div>
            <div className="text-sm font-semibold">{me.name}</div>
          </div>
        </Link>
        <Button variant="ghost" onClick={onLogout}>
          Logout
        </Button>
      </>
    )
  } else if (!loading && !me) {
    rightContent = (
      <>
        <Link to="/login">
          <Button variant="ghost">Log in</Button>
        </Link>
        <Link to="/signup">
          <Button>Create account</Button>
        </Link>
      </>
    )
  }

  return (
    <header className="sticky top-0 z-50 border-b border-(--gopay-border) bg-white/70 backdrop-blur">
      <div className="mx-auto flex w-full max-w-6xl items-center justify-between gap-4 px-4 py-3">
        <Link to="/" className="shrink-0">
          <GoPayMark />
        </Link>

        <nav className="hidden items-center gap-1 md:flex">
          <NavLink to="/" end className={navLinkClass}>
            Home
          </NavLink>
          <NavLink to="/dashboard" className={navLinkClass}>
            Dashboard
          </NavLink>
        </nav>

        <div className="flex items-center gap-2">
          {rightContent}
        </div>
      </div>
    </header>
  )
}

function Footer() {
  return (
    <footer className="border-t border-(--gopay-border) bg-white/60">
      <div className="mx-auto flex w-full max-w-6xl flex-col gap-2 px-4 py-6 text-sm text-(--gopay-muted) md:flex-row md:items-center md:justify-between">
        <div>© {new Date().getFullYear()} GoPay</div>
        <div className="flex gap-4">
          <Link className="hover:text-(--gopay-fg)" to="/terms">
            Terms
          </Link>
          <Link className="hover:text-(--gopay-fg)" to="/privacy">
            Privacy
          </Link>
          <Link className="hover:text-(--gopay-fg)" to="/support">
            Support
          </Link>
        </div>
      </div>
    </footer>
  )
}

export function AppLayout() {
  const navigate = useNavigate()

  const apiBaseUrl = useMemo(() => {
    return import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8080'
  }, [])

  const [me, setMe] = useState<MeResponse | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const loadMe = () => {
      const token = localStorage.getItem('gopay_token')
      if (!token) {
        setMe(null)
        setLoading(false)
        return
      }

      setLoading(true)
      fetch(`${apiBaseUrl}/api/v1/me`, {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      })
        .then(async (res) => {
          const data: unknown = await res.json().catch(() => null)
          if (!res.ok) {
            const message = (data as { error?: unknown } | null)?.error
            throw new Error(
              typeof message === 'string' ? message : 'Unable to fetch profile.',
            )
          }
          return data as MeResponse
        })
        .then((data) => {
          setMe(data)
        })
        .catch(() => {
          localStorage.removeItem('gopay_token')
          setMe(null)
        })
        .finally(() => setLoading(false))
    }

    loadMe()
    const onAuthChanged = () => loadMe()
    window.addEventListener('gopay-auth-changed', onAuthChanged)

    return () => {
      window.removeEventListener('gopay-auth-changed', onAuthChanged)
    }
  }, [apiBaseUrl])

  const onLogout = async () => {
    const token = localStorage.getItem('gopay_token')
    if (token) {
      await fetch(`${apiBaseUrl}/api/v1/auth/logout`, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
      }).catch(() => null)
    }

    localStorage.removeItem('gopay_token')
    setMe(null)
    window.dispatchEvent(new Event('gopay-auth-changed'))
    navigate('/login')
  }

  return (
    <div className="min-h-dvh">
      <TopNav me={me} loading={loading} onLogout={onLogout} />
      <main className="mx-auto w-full max-w-6xl px-4 py-10">
        <Outlet />
      </main>
      <Footer />
    </div>
  )
}

