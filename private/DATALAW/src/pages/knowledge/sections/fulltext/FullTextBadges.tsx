import type { SearchResult } from '../../../../services/rag/UnifiedSearchService';

export function MatchBadge({ type, label }: { type: SearchResult['matchType']; label: string }) {
  const tones: Record<SearchResult['matchType'], string> = {
    text: 'bg-blue-50 text-blue-700 border-blue-200',
    vector: 'bg-emerald-50 text-emerald-700 border-emerald-200',
    hybrid: 'bg-amber-50 text-amber-700 border-amber-200',
  };

  return (
    <span
      className={`rounded border px-1.5 py-0.5 text-[10px] font-semibold tracking-wide ${tones[type]}`}
    >
      {label}
    </span>
  );
}

export function FileExtBadge({ filePath }: { filePath: string }) {
  const ext = filePath.split('.').pop()?.toLowerCase();
  if (!ext) {
    return null;
  }

  return (
    <span className="rounded bg-black/[0.04] px-1.5 py-0.5 font-mono text-[10px] font-semibold uppercase text-black/55">
      .{ext}
    </span>
  );
}

export function ScoreBar({ score }: { score: number }) {
  const pct = Math.round(score * 100);
  const color = pct >= 80 ? 'bg-emerald-500' : pct >= 50 ? 'bg-amber-500' : 'bg-rose-500';

  return (
    <div className="flex items-center gap-1.5">
      <div className="h-1.5 w-16 overflow-hidden rounded-full bg-black/[0.06]">
        <div className={`h-full rounded-full ${color}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-[10px] tabular-nums text-black/45">{pct}%</span>
    </div>
  );
}
