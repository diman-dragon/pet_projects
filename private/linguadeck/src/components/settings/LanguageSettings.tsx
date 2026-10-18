import { Globe } from "lucide-react";
import type { AppSettings } from "../../types/domain";

interface LanguageSettingsProps {
  settings: AppSettings;
  onChange: (patch: Partial<AppSettings>) => void;
}

export function LanguageSettings(props: LanguageSettingsProps) {
  return (
    <div className="rounded-[30px] border border-white/70 bg-white/70 p-8 shadow-xl shadow-gray-200/50">
      <div className="mb-6 flex items-center gap-3">
        <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-gradient-to-br from-blue-500 to-blue-600 text-white shadow-lg">
          <Globe size={24} />
        </div>
        <div>
          <h3 className="text-lg font-bold text-slate-900">Языки</h3>
          <p className="text-sm text-slate-500">Основные языки библиотеки</p>
        </div>
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        <TextInput
          label="Изучаемый язык"
          value={props.settings.learningLanguage}
          onChange={(value) => props.onChange({ learningLanguage: value })}
        />
        <TextInput label="Родной язык" value="Русский" onChange={() => undefined} readOnly />
      </div>
    </div>
  );
}

function TextInput({
  label,
  value,
  onChange,
  readOnly = false,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  readOnly?: boolean;
}) {
  return (
    <div className="rounded-2xl border border-slate-100 bg-slate-50/50 p-4">
      <label className="mb-2 block text-xs font-semibold uppercase tracking-wider text-slate-500">
        {label}
      </label>
      <input
        className="w-full rounded-xl border-0 bg-white px-4 py-3 text-slate-900 shadow-sm outline-none ring-1 ring-slate-200 focus:ring-2 focus:ring-[#0078D4]"
        onChange={(event) => onChange(event.target.value)}
        readOnly={readOnly}
        value={value}
      />
    </div>
  );
}
