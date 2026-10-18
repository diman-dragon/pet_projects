import { ScrollText } from "lucide-react";
import type { AppLogEntry } from "../../types/domain";
import { DangerButton } from "../ui/DangerButton";

interface LogsSectionProps {
  logs: AppLogEntry[];
  onClearLogs: () => void;
}

export function LogsSection({ logs, onClearLogs }: LogsSectionProps) {
  return (
    <div className="rounded-[30px] border border-white/70 bg-white/70 p-8 shadow-xl shadow-gray-200/50">
      <div className="mb-6 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-gradient-to-br from-amber-500 to-amber-600 text-white shadow-lg">
            <ScrollText size={24} />
          </div>
          <div>
            <h3 className="text-lg font-bold text-slate-900">Системные логи</h3>
            <p className="text-sm text-slate-500">События и ошибки приложения</p>
          </div>
        </div>
        <DangerButton label="Очистить" onClick={onClearLogs} />
      </div>

      <div className="max-h-[400px] space-y-3 overflow-y-auto pr-2">
        {logs.length ? logs.map((entry) => <LogRow entry={entry} key={entry.id} />) : <div className="rounded-2xl bg-slate-50/50 p-8 text-center text-slate-400">Нет записей</div>}
      </div>
    </div>
  );
}

function LogRow({ entry }: { entry: AppLogEntry }) {
  const levelStyle = entry.level === "error" ? "bg-rose-100 text-rose-700" : entry.level === "warn" ? "bg-amber-100 text-amber-700" : "bg-emerald-100 text-emerald-700";

  return (
    <div className="rounded-2xl border border-slate-200 bg-slate-50/80 p-4 transition-all hover:border-slate-300">
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <span className={`rounded-full px-3 py-1 text-xs font-semibold ${levelStyle}`}>{entry.level}</span>
          <span className="text-sm font-semibold text-slate-700">{entry.scope}</span>
        </div>
        <span className="text-xs text-slate-400">{entry.created_at}</span>
      </div>
      <div className="mt-2 text-sm leading-6 text-slate-700">{entry.message}</div>
      {entry.details ? <pre className="mt-2 overflow-auto rounded-xl bg-slate-900 p-3 text-xs text-slate-100">{entry.details}</pre> : null}
    </div>
  );
}
