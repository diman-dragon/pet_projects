// src/pages/knowledge/components/shared/BouncingDots.tsx

interface BouncingDotsProps {
  color?: string; // Tailwind bg class
}

/** Три анимированных точки ожидания */
export function BouncingDots({ color = 'bg-violet-400' }: BouncingDotsProps) {
  return (
    <div className="flex items-center gap-1.5">
      {[0, 0.15, 0.3].map((delay, i) => (
        <span
          key={i}
          className={`size-1.5 rounded-full ${color} animate-bounce`}
          style={{ animationDelay: `${delay}s` }}
        />
      ))}
    </div>
  );
}
