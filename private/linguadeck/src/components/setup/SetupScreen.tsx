import { Card } from "../../ui";
import { PathSelector } from "./PathSelector";
import { SetupActions } from "./SetupActions";

interface SetupScreenProps {
  selectedPath: string | null;
  busy: boolean;
  error: string | null;
  onSelectFolder: () => void;
  onInitialize: () => void;
}

export function SetupScreen({
  selectedPath,
  busy,
  error,
  onSelectFolder,
  onInitialize,
}: SetupScreenProps) {
  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top_left,_rgba(37,99,235,0.22),_transparent_28%),radial-gradient(circle_at_bottom_right,_rgba(14,165,233,0.18),_transparent_24%),linear-gradient(180deg,#eef3fb_0%,#f7f9fc_45%,#eff3f8_100%)] text-slate-900">
      <div className="mx-auto flex min-h-screen max-w-6xl items-center px-8 py-12">
        <div className="grid w-full gap-8 lg:grid-cols-[1.1fr_0.9fr]">
          <div className="flex flex-col justify-center">
            <div className="inline-flex w-fit items-center gap-2 rounded-full border border-white/60 bg-white/55 px-4 py-2 text-xs font-semibold uppercase tracking-[0.24em] text-slate-500 backdrop-blur-xl">
              Final Build
            </div>
            <h1 className="mt-6 max-w-3xl text-5xl font-semibold tracking-tight text-slate-950">
              Linguadeck для глубокой локальной работы с карточками.
            </h1>
            <p className="mt-5 max-w-2xl text-lg leading-8 text-slate-600">
              Библиотека хранится локально в SQLite, медиа лежат в ваших папках, а повторения
              строятся по реальным действиям во время учебы.
            </p>
            <div className="mt-8 grid gap-4 sm:grid-cols-3">
              {[
                ["SM-2 Engine", "Реальные интервалы повторения и история оценок 1-4."],
                ["Behavior Rating", "Автооценка по времени, переворотам и навигации без ручных данных."],
                ["Win11 Flow", "Мягкий shell, mica-стекло, быстрые deck/cards workflows."],
              ].map(([title, text]) => (
                <div
                  key={title}
                  className="rounded-3xl border border-white/60 bg-white/55 p-5 shadow-[0_18px_60px_-28px_rgba(15,23,42,0.35)] backdrop-blur-xl"
                >
                  <div className="text-sm font-semibold text-slate-900">{title}</div>
                  <div className="mt-2 text-sm leading-6 text-slate-600">{text}</div>
                </div>
              ))}
            </div>
          </div>

          <div className="rounded-[40px] border border-white/60 bg-white/70 p-4 shadow-[0_28px_80px_-28px_rgba(15,23,42,0.35)] backdrop-blur-2xl">
            <Card className="!rounded-[32px] border-slate-200/70 bg-white/80 p-8">
              <div className="text-sm font-semibold uppercase tracking-[0.28em] text-slate-400">
                Bootstrap
              </div>
              <h2 className="mt-3 text-3xl font-semibold tracking-tight text-slate-950">
                Настройка библиотеки
              </h2>
              <p className="mt-3 text-sm leading-6 text-slate-600">
                Выберите корневую папку. Внутри автоматически будет создана структура{" "}
                <code>Linguadeck/database</code>, <code>media/audio</code>,{" "}
                <code>media/images</code>, <code>imports</code>, <code>exports</code>.
              </p>

              <PathSelector selectedPath={selectedPath} onSelect={onSelectFolder} />

              <SetupActions busy={busy} canInitialize={!!selectedPath} onInitialize={onInitialize} />

              {error ? (
                <div className="mt-5 rounded-2xl bg-rose-50 border border-rose-100 px-5 py-4 text-sm font-medium text-rose-600 animate-in fade-in slide-in-from-top-2 duration-300">
                  {error}
                </div>
              ) : null}
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}
