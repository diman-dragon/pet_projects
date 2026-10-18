import { CheckCircle2 } from "lucide-react";
import type { AppSettings } from "../../types/domain";

export function AlgoSummary({ settings }: { settings: AppSettings }) {
  return (
    <div className="flex items-center justify-between rounded-2xl border border-gray-100 bg-gradient-to-r from-slate-50 to-white px-5 py-3 shadow-sm">
      <div className="flex items-center gap-3">
        <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-[#0078D4]/10">
          <span className="text-xs font-black text-[#0078D4]">{settings.studyAlgorithm.toUpperCase()}</span>
        </div>
        <div>
          <div className="text-sm font-semibold text-gray-900">Алгоритм: {settings.studyAlgorithm.toUpperCase()}</div>
          <div className="text-xs text-gray-400">Шаги: {settings.learningSteps} · Лимит сессии: {settings.maxReviewsPerSession} карт</div>
        </div>
      </div>
      <div className="flex items-center gap-1.5 rounded-xl bg-emerald-50 px-3 py-1.5">
        <CheckCircle2 size={13} className="text-emerald-500" />
        <span className="text-xs font-semibold text-emerald-700">{settings.learningLanguage}</span>
      </div>
    </div>
  );
}
