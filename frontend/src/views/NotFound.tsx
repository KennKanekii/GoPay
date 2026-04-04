import { Link } from 'react-router-dom'
import { Button } from '../ui/Button'

export function NotFound() {
  return (
    <div className="mx-auto max-w-xl text-center">
      <div className="rounded-3xl border border-[color:var(--gopay-border)] bg-white/70 p-10 shadow-[var(--gopay-shadow)]">
        <div className="text-sm font-semibold text-[color:var(--gopay-muted)]">404</div>
        <h1 className="mt-2 text-3xl font-extrabold tracking-tight">Page not found</h1>
        <p className="mt-3 text-sm text-[color:var(--gopay-muted)]">
          The page you’re looking for doesn’t exist yet.
        </p>
        <div className="mt-6 flex justify-center gap-3">
          <Link to="/">
            <Button>Go home</Button>
          </Link>
          <Link to="/dashboard">
            <Button variant="ghost">Dashboard</Button>
          </Link>
        </div>
      </div>
    </div>
  )
}

