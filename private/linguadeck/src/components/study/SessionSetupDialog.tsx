import { X, BookOpen, RotateCcw, Sparkles, Settings2, Play } from "lucide-react";
import { useState } from "react";

export type StudyMode = "normal" | "review" | "new" | "hard" | "custom";

export interface SessionConfig {
  mode: StudyMode;
  limit: number;
  includeNew: boolean;
  includeReview: boolean;
  includeLearning: boolean;
  shuffle: boolean;
}

interface SessionSetupDialogProps {
  deckName: string;
  stats: {
    new: number;
    learning: number;
    review: number;
    total: number;
  };
  onStart: (config: SessionConfig) => void;
  onCancel: () => void;
}

const MODES: { id: StudyMode; label: string; icon: React.ReactNode; desc: string }[] = [
  { id: "normal", label: "Обычный", icon: <BookOpen size={20} />, desc: "Все карточки по порядку" },
  { id: "review", label: "Только повторение", icon: <RotateCcw size={20} />, desc: "Только due карточки" },
  { id: "new", label: "Только новые", icon: <Sparkles size={20} />, desc: "Только новые карточки" },
  { id: "hard", label: "Сложные", icon: <Settings2 size={20} />, desc: "Те, что вызывают затруднения" },
  { id: "custom", label: "Настроить", icon: <Settings2 size={20} />, desc: "Выбрать параметры" },
];

const LIMITS = [10, 20, 30, 50, 100, 9999];

export function SessionSetupDialog({ deckName, stats, onStart, onCancel }: SessionSetupDialogProps) {
  const [mode, setMode] = useState<StudyMode>("normal");
  const [limit, setLimit] = useState(20);
  const [includeNew, setIncludeNew] = useState(true);
  const [includeReview, setIncludeReview] = useState(true);
  const [includeLearning, setIncludeLearning] = useState(true);
  const [shuffle, setShuffle] = useState(true);

  const handleStart = () => {
    onStart({
      mode,
      limit,
      includeNew: mode === "new" || (mode === "custom" && includeNew) || (mode === "normal"),
      includeReview: mode === "review" || (mode === "custom" && includeReview) || (mode === "normal"),
      includeLearning: mode === "hard" || (mode === "custom" && includeLearning) || (mode === "normal"),
      shuffle,
    });
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
      <div className="w-full max-w-md max-h-[85vh] overflow-hidden rounded-2xl bg-white shadow-2xl flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-gray-100 px-6 py-4">
          <div>
            <h2 className="text-lg font-bold text-gray-900">Настройка сессии</h2>
            <p className="text-sm text-gray-500">{deckName}</p>
          </div>
          <button onClick={onCancel} className="rounded-lg p-2 hover:bg-gray-100">
            <X size={20} className="text-gray-500" />
          </button>
        </div>

        <div className="space-y-5 p-6 overflow-y-auto max-h-[60vh]">
          {/* Stats */}
          <div className="grid grid-cols-3 gap-2 rounded-xl bg-gray-50 p-3">
            <div className="text-center">
              <div className="text-lg font-bold text-emerald-600">{stats.new}</div>
              <div className="text-xs text-gray-500">Новые</div>
            </div>
            <div className="text-center">
              <div className="text-lg font-bold text-blue-600">{stats.learning}</div>
              <div className="text-xs text-gray-500">Изучается</div>
            </div>
            <div className="text-center">
              <div className="text-lg font-bold text-purple-600">{stats.review}</div>
              <div className="text-xs text-gray-500">Повторение</div>
            </div>
          </div>

          {/* Mode Selection */}
          <div className="space-y-3">
            <label className="text-sm font-medium text-gray-700">Режим обучения</label>
            <div className="grid grid-cols-2 gap-2">
              {MODES.map((m) => (
                <button
                  key={m.id}
                  onClick={() => setMode(m.id)}
                  className={`flex flex-col items-start gap-1 rounded-xl border-2 p-3 text-left transition-all ${mode === m.id
                    ? "border-blue-500 bg-blue-50"
                    : "border-gray-200 hover:border-gray-300"
                    }`}
                >
                  <div className={`${mode === m.id ? "text-blue-600" : "text-gray-500"}`}>
                    {m.icon}
                  </div>
                  <div className="text-sm font-medium text-gray-900">{m.label}</div>
                  <div className="text-xs text-gray-500">{m.desc}</div>
                </button>
              ))}
            </div>
          </div>

          {/* Custom Options */}
          {mode === "custom" && (
            <div className="space-y-2 rounded-xl bg-gray-50 p-3">
              <label className="text-sm font-medium text-gray-700">Включить карточки</label>
              <div className="flex gap-4">
                <label className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={includeNew}
                    onChange={(e) => setIncludeNew(e.target.checked)}
                    className="rounded border-gray-300"
                  />
                  <span className="text-sm text-gray-700">Новые</span>
                </label>
                <label className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={includeLearning}
                    onChange={(e) => setIncludeLearning(e.target.checked)}
                    className="rounded border-gray-300"
                  />
                  <span className="text-sm text-gray-700">Изучается</span>
                </label>
                <label className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={includeReview}
                    onChange={(e) => setIncludeReview(e.target.checked)}
                    className="rounded border-gray-300"
                  />
                  <span className="text-sm text-gray-700">Повторение</span>
                </label>
              </div>
            </div>
          )}

          {/* Limit */}
          <div className="space-y-3">
            <label className="text-sm font-medium text-gray-700">Количество карточек</label>
            <div className="flex flex-wrap gap-2">
              {LIMITS.map((l) => (
                <button
                  key={l}
                  onClick={() => setLimit(l)}
                  className={`rounded-lg px-4 py-2 text-sm font-medium transition-all ${limit === l
                    ? "bg-blue-500 text-white"
                    : "bg-gray-100 text-gray-700 hover:bg-gray-200"
                    }`}
                >
                  {l === 9999 ? "Все" : l}
                </button>
              ))}
            </div>
          </div>

          {/* Shuffle */}
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={shuffle}
              onChange={(e) => setShuffle(e.target.checked)}
              className="rounded border-gray-300"
            />
            <span className="text-sm text-gray-700">Перемешать карточки</span>
          </label>
        </div>

        {/* Footer */}
        <div className="border-t border-gray-100 px-6 py-4">
          <button
            onClick={handleStart}
            className="flex w-full items-center justify-center gap-2 rounded-xl bg-blue-500 py-3 font-medium text-white transition-colors hover:bg-blue-600"
          >
            <Play size={20} />
            Начать сессию
          </button>
        </div>
      </div>
    </div>
  );
}
