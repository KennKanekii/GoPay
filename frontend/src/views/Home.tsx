import { Link } from 'react-router-dom'
import { Button } from '../ui/Button'

function Stat({ label, value }: Readonly<{ label: string; value: string }>) {
  return (
    <div className="rounded-2xl border border-[color:var(--gopay-border)] bg-white/70 p-4 shadow-[var(--gopay-shadow)]">
      <div className="text-xs font-semibold text-[color:var(--gopay-muted)]">{label}</div>
      <div className="mt-1 text-xl font-extrabold tracking-tight">{value}</div>
    </div>
  )
}

export function Home() {
  return (
    <div className="grid items-start gap-10 md:grid-cols-2">
      <section>
        <h1 className="mt-5 text-4xl font-extrabold tracking-tight text-[color:var(--gopay-fg)] md:text-5xl">
          Pay instantly. Track everything. Borrow smarter.
        </h1>
        <p className="mt-4 max-w-prose text-base leading-relaxed text-[color:var(--gopay-muted)]">
          GoPay is your unified wallet + UPI payments + lending dashboard. This is the first UI foundation — next we’ll
          connect it to the services you described (UPI, lending, fraud scoring).
        </p>

        <div className="mt-6 flex flex-wrap gap-3">
          <Link to="/login">
            <Button>Get started</Button>
          </Link>
          <Link to="/dashboard">
            <Button variant="ghost">View dashboard</Button>
          </Link>
        </div>

        <div className="mt-10 grid grid-cols-2 gap-4">
          <Stat label="UPI" value="P2P • P2M" />
          <Stat label="Lending" value="Loans • EMI" />
          <Stat label="Fraud" value="Rules + ML" />
          <Stat label="Events" value="Kafka-ready" />
        </div>
      </section>

      <section className="rounded-3xl border border-[color:var(--gopay-border)] bg-white/70 p-6 shadow-[var(--gopay-shadow)]">
        <div className="flex items-center justify-between">
          <div>
            <div className="text-sm font-semibold text-[color:var(--gopay-muted)]">Quick actions</div>
            <div className="text-xl font-extrabold tracking-tight">Try the UI flow</div>
          </div>
          <div className="h-10 w-10 rounded-2xl bg-[color:var(--gopay-primary-3)]" />
        </div>

        <div className="mt-6 grid gap-3">
          <div className="rounded-2xl border border-[color:var(--gopay-border)] bg-white p-4">
            <div className="text-sm font-semibold">Send money</div>
            <div className="mt-1 text-sm text-[color:var(--gopay-muted)]">VPA → amount → risk score → submit</div>
          </div>
          <div className="rounded-2xl border border-[color:var(--gopay-border)] bg-white p-4">
            <div className="text-sm font-semibold">Apply for loan</div>
            <div className="mt-1 text-sm text-[color:var(--gopay-muted)]">KYC → score → approval → disbursement via UPI</div>
          </div>
          <div className="rounded-2xl border border-[color:var(--gopay-border)] bg-white p-4">
            <div className="text-sm font-semibold">Fraud console</div>
            <div className="mt-1 text-sm text-[color:var(--gopay-muted)]">Alerts → cases → review → block/unblock</div>
          </div>
        </div>

        <div className="mt-6 rounded-2xl bg-[color:rgba(18,179,165,0.12)] p-4">
          <div className="text-sm font-semibold">Next step</div>
          <div className="mt-1 text-sm text-[color:var(--gopay-muted)]">
            I can add real forms + a mocked API layer so the app behaves like a real GoPay product from day one.
          </div>
        </div>
      </section>
    </div>
  )
}

