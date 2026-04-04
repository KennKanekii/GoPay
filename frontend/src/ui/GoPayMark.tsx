export function GoPayMark({ className = '' }: Readonly<{ className?: string }>) {
  return (
    <div className={`flex items-center gap-2 ${className}`}>
      <div className="flex h-9 w-9 items-center justify-center rounded-2xl bg-[color:var(--gopay-primary)] shadow-[var(--gopay-shadow)]">
        <span className="text-sm font-extrabold tracking-tight text-white">GP</span>
      </div>
      <div className="leading-tight">
        <div className="text-base font-extrabold tracking-tight">GoPay</div>
        <div className="text-xs text-[color:var(--gopay-muted)]">Pay • Save • Borrow</div>
      </div>
    </div>
  )
}

