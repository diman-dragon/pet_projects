import { Brain } from "lucide-react";
import type { AppSettings } from "../../types/domain";
import { SelectField } from "../ui/SelectField";

interface AlgorithmSettingsProps {
  settings: AppSettings;
  onChange: (patch: Partial<AppSettings>) => void;
}

export function AlgorithmSettings({ settings, onChange }: AlgorithmSettingsProps) {
  return (
    <div className="rounded-[30px] border border-white/70 bg-white/70 p-8 shadow-xl shadow-gray-200/50">
      <div className="mb-6 flex items-center gap-3">
        <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-gradient-to-br from-emerald-500 to-emerald-600 text-white shadow-lg">
          <Brain size={24} />
        </div>
        <div>
          <h3 className="text-lg font-bold text-slate-900">Механика изучения</h3>
          <p className="text-sm text-slate-500">Алгоритм, шаги и лимиты сессии</p>
        </div>
      </div>

      <div className="grid gap-5">
        <SelectField label="Алгоритм обучения" value={settings.studyAlgorithm} onChange={(value) => onChange({ studyAlgorithm: value as AppSettings["studyAlgorithm"] })} options={[{ value: "sm2", label: "SM-2" }, { value: "cram", label: "Cram" }, { value: "fsrs", label: "FSRS-4.5" }]} />
        <NumberField label="Шаги обучения (минуты через запятую)" type="text" value={settings.learningSteps} onChange={(value) => onChange({ learningSteps: value })} />
        <div className="grid gap-4 sm:grid-cols-3">
          <NumberField label="Graduate (дни)" value={String(settings.graduatingInterval)} onChange={(value) => onChange({ graduatingInterval: Number(value) })} />
          <NumberField label="Easy (дни)" value={String(settings.easyInterval)} onChange={(value) => onChange({ easyInterval: Number(value) })} />
          <NumberField label="Лимит сессии" value={String(settings.maxReviewsPerSession)} onChange={(value) => onChange({ maxReviewsPerSession: Number(value) })} />
        </div>
      </div>
    </div>
  );
}

function NumberField({ label, value, onChange, type = "number" }: { label: string; value: string; onChange: (value: string) => void; type?: "number" | "text" }) {
  return (
    <div className="rounded-2xl border border-slate-100 bg-slate-50/50 p-4">
      <label className="mb-2 block text-xs font-semibold uppercase tracking-wider text-slate-500">{label}</label>
      <input className="w-full rounded-xl border-0 bg-white px-4 py-3 text-slate-900 shadow-sm outline-none ring-1 ring-slate-200 focus:ring-2 focus:ring-[#0078D4]" onChange={(event) => onChange(event.target.value)} type={type} value={value} />
    </div>
  );
}
