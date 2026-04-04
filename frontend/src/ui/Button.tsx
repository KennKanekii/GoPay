import type { ButtonHTMLAttributes } from 'react'

type Variant = 'primary' | 'ghost'

export function Button({
  className = '',
  variant = 'primary',
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant }) {
  const base =
    'inline-flex items-center justify-center gap-2 rounded-xl px-4 py-2 text-sm font-semibold transition outline-none focus-visible:ring-4 focus-visible:ring-[color:rgba(18,179,165,0.25)] disabled:opacity-60 disabled:cursor-not-allowed'

  const styles =
    variant === 'primary'
      ? 'bg-[color:var(--gopay-primary)] text-white shadow-[var(--gopay-shadow)] hover:bg-[color:var(--gopay-primary-2)]'
      : 'bg-transparent text-[color:var(--gopay-fg)] hover:bg-[color:rgba(18,179,165,0.10)]'

  return <button className={`${base} ${styles} ${className}`} {...props} />
}

