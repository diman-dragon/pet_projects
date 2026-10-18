// components/chat/QuickPrompts.tsx

interface QuickPromptsProps {
  prompts: string[];
  onSelect: (prompt: string) => void;
}

/** Быстрые подсказки — показываются только пока история пустая */
export function QuickPrompts({ prompts, onSelect }: QuickPromptsProps) {
  return (
    <div className="px-6 pb-3 flex flex-wrap gap-2">
      {prompts.map((p) => (
        <button
          key={p}
          onClick={() => onSelect(p)}
          className="px-3 py-1.5 rounded-full border border-black/[0.08] text-[length:var(--text-xs)] font-bold text-black/50
            hover:border-[var(--app-accent)]/30 hover:text-[var(--app-accent)] hover:bg-[var(--app-accent)]/[0.03] transition-all duration-150"
        >
          {p}
        </button>
      ))}
    </div>
  );
}
