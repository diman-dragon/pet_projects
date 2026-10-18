import type { GrammarExercise } from "../../../types/domain";
import { parsePayload } from "./grammarHelpers";

interface Props {
  exercise: GrammarExercise;
  value: string;
  onChange: (value: string) => void;
}

export function GrammarExerciseInput({ exercise, value, onChange }: Props) {
  const payload = parsePayload(exercise);

  if (exercise.type === "choice") {
    return (
      <div className="flex flex-wrap gap-2">
        {(payload.options ?? []).map((option) => (
          <button
            key={option}
            type="button"
            onClick={() => onChange(option)}
            className={`rounded-lg border px-3 py-2 text-sm font-semibold ${
              value === option
                ? "border-blue-500 bg-blue-50 text-blue-700"
                : "border-slate-200 bg-white text-slate-600"
            }`}
          >
            {option}
          </button>
        ))}
      </div>
    );
  }

  if (exercise.type === "reorder") {
    return (
      <div className="space-y-2">
        <div className="flex flex-wrap gap-2">
          {(payload.blocks ?? []).map((block) => (
            <button
              key={block}
              type="button"
              onClick={() => onChange(`${value} ${block}`.trim())}
              className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm font-semibold text-slate-700"
            >
              {block}
            </button>
          ))}
        </div>
        <input
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm"
        />
      </div>
    );
  }

  return (
    <input
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm"
      placeholder="Ваш ответ"
    />
  );
}
