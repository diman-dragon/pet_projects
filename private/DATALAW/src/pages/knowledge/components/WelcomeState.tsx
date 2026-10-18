import { Sparkles } from 'lucide-react';

export function WelcomeState() {
  return (
    <div className="flex flex-col items-center justify-center py-10 text-center px-4">
      <div className="relative mb-4">
        <div className="flex size-12 items-center justify-center rounded-2xl bg-[var(--app-accent)]/10">
          <Sparkles size={20} className="text-[var(--app-accent)]" strokeWidth={1.5} />
        </div>
        <div className="absolute -right-1 -top-1 size-3 rounded-full bg-[var(--app-accent)]/30 animate-ping" />
      </div>
      <p className="text-[length:var(--text-xs)] font-semibold text-black/40">
        Семантический поиск
      </p>
      <p className="mt-1.5 text-[10px] text-black/25 leading-relaxed max-w-[200px]">
        Введите 2+ слова — запрос будет векторизован и сравнён с документами базы знаний
      </p>
      <div className="mt-4 grid grid-cols-1 gap-1.5 w-full max-w-[220px]">
        {['договор аренды недвижимости', 'ответственность поставщика', 'срок исковой давности'].map(hint => (
          <div key={hint} className="rounded-lg border border-dashed border-black/[0.08] px-3 py-1.5 text-[10px] text-black/30 text-left">
            {hint}
          </div>
        ))}
      </div>
    </div>
  );
}
