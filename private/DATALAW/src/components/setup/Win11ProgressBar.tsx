/**
 * Win11ProgressBar — FIXED
 *
 * BUG FIX: indeterminate режим использовал `animation: 'slide 1.5s infinite'`
 * без объявления @keyframes slide — полоска стояла на месте.
 *
 * РЕШЕНИЕ: заменяем инлайн-style на Tailwind translate + keyframes в CSS-переменных,
 * либо используем встроенный animate-[shimmer] через <style> тег в компоненте.
 */

interface Win11ProgressBarProps {
  percent: number;
  indeterminate?: boolean;
}

export function Win11ProgressBar({ percent, indeterminate }: Win11ProgressBarProps) {
  if (indeterminate) {
    return (
      <>
        {/* keyframe объявляется один раз — при первом рендере вставляется в <head> */}
        <style>{`
          @keyframes dl-slide {
            0%   { transform: translateX(-100%); }
            100% { transform: translateX(350%); }
          }
          .dl-indeterminate {
            animation: dl-slide 1.4s cubic-bezier(0.4, 0, 0.6, 1) infinite;
          }
        `}</style>
        <div className="w-full h-[3px] bg-[var(--app-border)] rounded-[2px] overflow-hidden">
          <div
            className="dl-indeterminate h-full w-[30%] bg-[var(--app-accent)] rounded-[2px]"
          />
        </div>
      </>
    );
  }

  return (
    <div className="w-full h-[3px] bg-[var(--app-border)] rounded-[2px] overflow-hidden">
      <div
        className="h-full bg-[var(--app-accent)] rounded-[2px] transition-[width] duration-500 ease-out"
        style={{ width: `${Math.min(100, Math.max(0, percent))}%` }}
      />
    </div>
  );
}
