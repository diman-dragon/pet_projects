import type { ImportProgress } from "../../../types/domain";

interface ImportProgressBannerProps {
  progress: ImportProgress;
}

export function ImportProgressBanner({ progress }: ImportProgressBannerProps) {
  if (progress.status === "idle") return null;

  const pct =
    progress.total > 0
      ? Math.round((progress.inserted / progress.total) * 100)
      : 0;

  if (progress.status === "running") {
    return (
      <div className="rounded-xl border border-blue-200 bg-blue-50 p-4 shadow-sm">
        <div className="mb-2 flex items-center justify-between text-sm font-medium text-blue-800">
          <span>Импорт карточек…</span>
          <span>{progress.total} карточек</span>
        </div>
        <div className="h-2 w-full overflow-hidden rounded-full bg-blue-200">
          <div
            className="h-2 rounded-full bg-blue-500 transition-all duration-300"
            style={{ width: "60%" }}
          />
        </div>
        <p className="mt-1.5 text-xs text-blue-600">Обработка батча в Rust…</p>
      </div>
    );
  }

  if (progress.status === "done") {
    return (
      <div className="rounded-xl border border-emerald-200 bg-emerald-50 p-4 shadow-sm">
        <div className="flex items-center gap-3">
          <span className="text-xl">✅</span>
          <div>
            <p className="text-sm font-semibold text-emerald-800">
              Импорт завершён успешно
            </p>
            <p className="text-xs text-emerald-600">
              Добавлено {progress.inserted} из {progress.total} карточек
            </p>
          </div>
        </div>
        <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-emerald-200">
          <div className="h-1.5 rounded-full bg-emerald-500" style={{ width: `${pct}%` }} />
        </div>
      </div>
    );
  }

  if (progress.status === "error") {
    return (
      <div className="rounded-xl border border-red-200 bg-red-50 p-4 shadow-sm">
        <div className="flex items-center gap-3">
          <span className="text-xl">❌</span>
          <div>
            <p className="text-sm font-semibold text-red-800">Ошибка импорта</p>
            <p className="text-xs text-red-600">{progress.error ?? "Неизвестная ошибка"}</p>
          </div>
        </div>
      </div>
    );
  }

  return null;
}
