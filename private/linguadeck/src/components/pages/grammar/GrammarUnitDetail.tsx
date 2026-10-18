import { BookOpenText, CheckCircle2, GraduationCap, RotateCcw, Sparkles } from "lucide-react";
import type { Card, GrammarExercise, GrammarSessionResult, GrammarUnit } from "../../../types/domain";
import { GrammarExerciseInput } from "./GrammarExerciseInput";
import { normalize, statusLabel } from "./grammarHelpers";

interface Props {
  selected: GrammarUnit;
  practice: GrammarExercise[];
  exercises: GrammarExercise[];
  answers: Record<number, string>;
  checked: Record<number, boolean>;
  result: GrammarSessionResult | null;
  linkedCards: Card[];
  onStartPractice: (control?: boolean) => void;
  onFinishPractice: () => void;
  onAnswer: (id: number, value: string) => void;
  onCheck: (id: number) => void;
}

export function GrammarUnitDetail({
  selected,
  practice,
  exercises,
  answers,
  checked,
  result,
  linkedCards,
  onStartPractice,
  onFinishPractice,
  onAnswer,
  onCheck,
}: Props) {
  return (
    <div className="space-y-5">
      {/* Unit header */}
      <div className="flex items-start justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 text-sm font-semibold text-blue-600">
            <BookOpenText size={16} />{selected.section || "Грамматика"}
          </div>
          <h2 className="mt-1 text-2xl font-black text-slate-900">{selected.title}</h2>
          <div className="mt-2 flex gap-2 text-xs font-bold text-slate-500">
            <span>{selected.language}</span>
            <span>{selected.level}</span>
            <span>{statusLabel(selected.status)}</span>
            <span>{selected.mastery_streak}/3</span>
          </div>
        </div>
        <div className="flex gap-2">
          <button
            type="button"
            onClick={() => onStartPractice(false)}
            className="flex items-center gap-2 rounded-lg bg-blue-600 px-4 py-2 text-sm font-bold text-white"
          >
            <GraduationCap size={16} />Тренировка ({exercises.length})
          </button>
          <button
            type="button"
            onClick={() => onStartPractice(true)}
            className="flex items-center gap-2 rounded-lg bg-slate-100 px-4 py-2 text-sm font-bold text-slate-700"
          >
            <RotateCcw size={16} />Срез
          </button>
        </div>
      </div>

      {/* Theory + examples */}
      <div className="grid grid-cols-2 gap-4">
        <div>
          <h3 className="mb-2 text-sm font-black uppercase text-slate-400">Теория</h3>
          <div className="whitespace-pre-wrap rounded-lg bg-slate-50 p-4 text-sm leading-6 text-slate-700">
            {selected.theory || "Добавьте объяснение правила."}
          </div>
        </div>
        <div>
          <h3 className="mb-2 text-sm font-black uppercase text-slate-400">Примеры</h3>
          <div className="whitespace-pre-wrap rounded-lg bg-emerald-50 p-4 text-sm leading-6 text-emerald-900">
            {selected.examples || "Добавьте примеры из реальной речи."}
          </div>
        </div>
      </div>

      {/* Practice */}
      <div>
        <h3 className="mb-2 text-sm font-black uppercase text-slate-400">Практика</h3>
        <div className="space-y-3">
          {practice.map((exercise, index) => {
            const isChecked = checked[exercise.id];
            const ok = normalize(answers[exercise.id] ?? "") === normalize(exercise.answer);
            return (
              <div key={exercise.id} className="rounded-lg border border-slate-200 p-4">
                <div className="mb-3 flex items-center justify-between gap-3">
                  <div className="text-sm font-bold text-slate-900">
                    {index + 1}. {exercise.prompt}
                  </div>
                  <span className="rounded-full bg-slate-100 px-2 py-0.5 text-[11px] font-bold text-slate-500">
                    {exercise.type}
                  </span>
                </div>
                <GrammarExerciseInput
                  exercise={exercise}
                  value={answers[exercise.id] ?? ""}
                  onChange={(value) => onAnswer(exercise.id, value)}
                />
                <div className="mt-3 flex items-center gap-2">
                  <button
                    type="button"
                    onClick={() => onCheck(exercise.id)}
                    className="rounded-lg bg-slate-900 px-3 py-2 text-xs font-bold text-white"
                  >
                    Проверить
                  </button>
                  {isChecked && (
                    <span className={`text-sm font-bold ${ok ? "text-emerald-600" : "text-rose-600"}`}>
                      {ok ? "Верно" : `Ответ: ${exercise.answer}`}
                    </span>
                  )}
                </div>
                {isChecked && (
                  <div className="mt-2 rounded-lg bg-blue-50 p-3 text-sm text-blue-900">
                    {exercise.explanation}
                  </div>
                )}
              </div>
            );
          })}
        </div>
        {practice.length > 0 && (
          <button
            type="button"
            onClick={onFinishPractice}
            className="mt-3 flex items-center gap-2 rounded-lg bg-emerald-600 px-4 py-2 text-sm font-bold text-white"
          >
            <CheckCircle2 size={16} />Завершить сессию
          </button>
        )}
        {result && (
          <div className="mt-3 rounded-lg bg-emerald-50 p-4 text-sm font-bold text-emerald-800">
            Результат {result.score}%. Статус: {statusLabel(result.status)}. Серия: {result.mastery_streak}/3.
          </div>
        )}
      </div>

      {/* Linked cards */}
      <div>
        <h3 className="mb-2 flex items-center gap-2 text-sm font-black uppercase text-slate-400">
          <Sparkles size={16} />Карточки по меткам
        </h3>
        <div className="grid grid-cols-3 gap-3">
          {linkedCards.map((card) => (
            <div key={card.id} className="rounded-lg bg-slate-50 p-3">
              <div className="font-bold text-slate-900">{card.front}</div>
              <div className="mt-1 text-sm text-slate-500">{card.back}</div>
            </div>
          ))}
          {linkedCards.length === 0 && (
            <div className="text-sm text-slate-500">
              Связанные карточки появятся, когда метки темы совпадут с метками карточек.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
