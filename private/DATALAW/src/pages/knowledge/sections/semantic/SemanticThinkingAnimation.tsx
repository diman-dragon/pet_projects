export function SemanticThinkingAnimation({ label }: { label: string }) {
  return (
    <div className="flex items-center gap-1.5 px-4 py-3">
      <div className="flex gap-1">
        <span className="h-2 w-2 animate-bounce rounded-full bg-[var(--app-accent)]/60" style={{ animationDelay: '0ms' }} />
        <span className="h-2 w-2 animate-bounce rounded-full bg-[var(--app-accent)]/60" style={{ animationDelay: '150ms' }} />
        <span className="h-2 w-2 animate-bounce rounded-full bg-[var(--app-accent)]/60" style={{ animationDelay: '300ms' }} />
      </div>
      <span className="text-sm text-black/50">{label}</span>
    </div>
  );
}
