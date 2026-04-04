export function GoPayMark({ className = '' }: Readonly<{ className?: string }>) {
  return (
    <div className={`flex items-center gap-2 ${className}`}>
      <div className="h-9 w-9 rounded-2xl bg-[color:var(--gopay-primary)] shadow-[var(--gopay-shadow)]" />
      <div className="leading-tight">
        <div className="text-base font-extrabold tracking-tight">GoPay</div>
        <div className="text-xs text-[color:var(--gopay-muted)]">Pay • Save • Borrow</div>
      </div>
    </div>
  )
}

