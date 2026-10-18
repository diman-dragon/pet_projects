/**
 * Win11StepRow — UPDATED
 *
 * Добавлен проп `indeterminate` для передачи в Win11ProgressBar
 * (нужен для OllamaInstallStep, где прогресс неизвестен).
 */
import { Win11ProgressBar } from './Win11ProgressBar';

type StepStatus = 'pending' | 'loading' | 'done' | 'error';

const statusStyles: Record<StepStatus, string> = {
  pending: 'text-[var(--app-text-muted)]',
  loading: 'text-[var(--app-accent)]',
  done: 'text-[var(--app-success)]',
  error: 'text-[var(--app-danger)]',
};

export function Win11StepRow({
  icon,
  text,
  status,
  progress,
  subText,
  metaText,
  indeterminate,
}: {
  icon: string;
  text: string;
  status: StepStatus;
  progress?: number;
  subText?: string;
  metaText?: string;
  indeterminate?: boolean;
}) {
  const showBar = indeterminate || progress !== undefined;

  return (
    <div className="flex flex-col gap-2 py-2">
      <div className={`flex items-center gap-3 text-sm ${statusStyles[status]}`}>
        <span className="flex h-5 w-5 items-center justify-center font-medium">{icon}</span>
        <span className="flex-1">{text}</span>
      </div>
      {showBar && (
        <div className="pl-8 pr-2">
          <Win11ProgressBar
            percent={progress ?? 0}
            indeterminate={indeterminate}
          />
          {(subText || metaText) && (
            <div className="mt-1 flex items-center justify-between gap-3 text-xs text-[var(--app-text-muted)]">
              <p>{subText}</p>
              {metaText ? <p className="whitespace-nowrap">{metaText}</p> : null}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
