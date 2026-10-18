import { Plus } from "lucide-react";
import type { GrammarUnit } from "../../../types/domain";
import type { GrammarUnitInput } from "../../../services/db/GrammarService";
import { statusLabel } from "./grammarHelpers";

interface GrammarSidebarProps {
  units: GrammarUnit[];
  selectedId: number | null;
  onSelect: (id: number) => void;
  draft: GrammarUnitInput;
  onDraftChange: (draft: GrammarUnitInput) => void;
  onAdd: () => void;
  busy: boolean;
}

export function GrammarSidebar({
  units,
  selectedId,
  onSelect,
  draft,
  onDraftChange,
  onAdd,
  busy,
}: GrammarSidebarProps) {
  return (
    <aside className="custom-scrollbar min-h-0 overflow-auto rounded-lg bg-white p-3 shadow-sm ring-1 ring-slate-200">
      <div className="mb-3 space-y-2 rounded-lg bg-slate-50 p-3">
        <input
          value={draft.title}
          onChange={(e) => onDraftChange({ ...draft, title: e.target.value })}
          className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm"
          placeholder="Новая тема"
        />
        <div className="grid grid-cols-2 gap-2">
          <input
            value={draft.section}
            onChange={(e) => onDraftChange({ ...draft, section: e.target.value })}
            className="rounded-lg border border-slate-200 px-3 py-2 text-sm"
            placeholder="Раздел"
          />
          <input
            value={draft.tags ?? ""}
            onChange={(e) => onDraftChange({ ...draft, tags: e.target.value })}
            className="rounded-lg border border-slate-200 px-3 py-2 text-sm"
            placeholder="Метка"
          />
        </div>
        <textarea
          value={draft.theory}
          onChange={(e) => onDraftChange({ ...draft, theory: e.target.value })}
          className="h-20 w-full rounded-lg border border-slate-200 px-3 py-2 text-sm"
          placeholder="Правило"
        />
        <textarea
          value={draft.examples}
          onChange={(e) => onDraftChange({ ...draft, examples: e.target.value })}
          className="h-16 w-full rounded-lg border border-slate-200 px-3 py-2 text-sm"
          placeholder="Примеры из речи"
        />
        <button
          type="button"
          disabled={busy || !draft.title.trim()}
          onClick={onAdd}
          className="flex w-full items-center justify-center gap-2 rounded-lg bg-blue-600 px-3 py-2 text-sm font-bold text-white disabled:opacity-50"
        >
          <Plus size={16} />
          Создать тему
        </button>
      </div>

      <div className="space-y-2">
        {units.map((unit) => (
          <button
            key={unit.id}
            type="button"
            onClick={() => onSelect(unit.id)}
            className={`w-full rounded-lg p-3 text-left ring-1 ${
              selectedId === unit.id
                ? "bg-blue-50 ring-blue-200"
                : "bg-white ring-slate-200 hover:bg-slate-50"
            }`}
          >
            <div className="flex items-center justify-between gap-2">
              <span className="font-bold text-slate-900">{unit.title}</span>
              <span className="rounded-full bg-slate-100 px-2 py-0.5 text-[11px] font-bold text-slate-500">
                {statusLabel(unit.status)}
              </span>
            </div>
            <div className="mt-1 text-xs text-slate-500">
              {unit.section || "Без раздела"} · {unit.level} · {unit.exercise_count} упр.
            </div>
          </button>
        ))}
      </div>
    </aside>
  );
}
