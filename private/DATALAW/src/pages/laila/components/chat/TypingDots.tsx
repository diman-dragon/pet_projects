// components/chat/TypingDots.tsx

/** Три анимированных точки — показываются пока стрим пуст */
export function TypingDots() {
  return (
    <span className="flex items-center gap-1.5 h-4">
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          className="size-1.5 rounded-full bg-violet-400 animate-bounce"
          style={{ animationDelay: `${i * 0.15}s`, animationDuration: '0.8s' }}
        />
      ))}
    </span>
  );
}
