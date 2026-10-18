import type { GrammarDashboard as GrammarDashboardType } from "../../../types/domain";

interface GrammarDashboardProps {
  dashboard: GrammarDashboardType | null;
}

export function GrammarDashboard({ dashboard }: GrammarDashboardProps) {
  const stats = [
    ["Всего", dashboard?.total ?? 0],
    ["Не начато", dashboard?.notStarted ?? 0],
    ["В процессе", dashboard?.inProgress ?? 0],
    ["Освоено", dashboard?.mastered ?? 0],
    ["Контроль", dashboard?.dueReviews ?? 0],
  ] as [string, number][];

  return (
    <div className="grid grid-cols-5 gap-3">
      {stats.map(([label, value]) => (
        <div key={label} className="rounded-lg bg-white p-4 shadow-sm ring-1 ring-slate-200">
          <div className="text-xs font-semibold uppercase text-slate-400">{label}</div>
          <div className="mt-1 text-2xl font-black text-slate-900">{value}</div>
        </div>
      ))}
    </div>
  );
}
