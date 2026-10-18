import { Download, Import, Library } from "lucide-react";
import type { RefObject } from "react";

interface GrammarHeaderProps {
  importRef: RefObject<HTMLInputElement | null>;
  onImport: (file: File | null) => void;
  onExport: () => void;
  onImportDefault: () => void;
  exportDisabled: boolean;
}

export function GrammarHeader({
  importRef,
  onImport,
  onExport,
  onImportDefault,
  exportDisabled,
}: GrammarHeaderProps) {
  return (
    <div className="flex items-center justify-between">
      <div>
        <h1 className="text-2xl font-black text-slate-900">Грамматика</h1>
        <p className="text-sm text-slate-500">
          Теория, упражнения, автоповторение и связь с карточками по меткам.
        </p>
      </div>
      <div className="flex gap-2">
        <input
          ref={importRef}
          className="hidden"
          type="file"
          accept=".json,.linguadeck-grammar.json,application/json"
          onChange={(e) => onImport(e.target.files?.[0] ?? null)}
        />
        <button
          type="button"
          onClick={() => importRef.current?.click()}
          className="flex items-center gap-2 rounded-lg bg-white px-3 py-2 text-sm font-semibold text-slate-600 shadow-sm ring-1 ring-slate-200"
        >
          <Import size={16} />
          Импорт
        </button>
        <button
          type="button"
          onClick={onExport}
          disabled={exportDisabled}
          className="flex items-center gap-2 rounded-lg bg-white px-3 py-2 text-sm font-semibold text-slate-600 shadow-sm ring-1 ring-slate-200 disabled:opacity-40"
        >
          <Download size={16} />
          Экспорт
        </button>
        <button
          type="button"
          onClick={onImportDefault}
          className="flex items-center gap-2 rounded-lg bg-emerald-600 px-3 py-2 text-sm font-semibold text-white"
        >
          <Library size={16} />
          Английская грамматика
        </button>
      </div>
    </div>
  );
}
