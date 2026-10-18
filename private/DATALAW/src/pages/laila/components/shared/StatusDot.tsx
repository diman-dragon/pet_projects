// components/shared/StatusDot.tsx

interface StatusDotProps {
  /** Tailwind bg-color class, e.g. 'bg-emerald-400' */
  colorClass: string;
  pulse?: boolean;
  size?: 'xs' | 'sm';
}

const SIZE_MAP = { xs: 'size-1.5', sm: 'size-2' };

/** Маленький цветной индикатор статуса */
export function StatusDot({ colorClass, pulse = false, size = 'xs' }: StatusDotProps) {
  return (
    <span
      className={`${SIZE_MAP[size]} rounded-full shrink-0 ${colorClass} ${pulse ? 'animate-pulse' : ''}`}
    />
  );
}
