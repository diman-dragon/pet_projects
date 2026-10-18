import { useEffect, useRef, useState } from 'react';
import { listen } from '@tauri-apps/api/event';
import { aiApi } from '../../services/backend/aiApi';
import { Win11StepRow } from './Win11StepRow';

interface PullProgress {
  stage: 'manifest' | 'download' | 'verify' | 'complete';
  percent: number;
  total_mb?: number;
  done_mb?: number;
}

function formatEta(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds <= 0) return 'ETA --';
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = Math.round(seconds % 60);
  return minutes > 0 ? `ETA ${minutes}m ${remainingSeconds}s` : `ETA ${remainingSeconds}s`;
}

interface ModelPullStepProps {
  modelName: string;
  onComplete: () => void;
  /** Если true — модель уже загружена, шаг сразу показывает done и вызывает onComplete. */
  alreadyPulled?: boolean;
}

export function ModelPullStep({ modelName, onComplete, alreadyPulled = false }: ModelPullStepProps) {
  const startedRef = useRef(false);
  const [progress, setProgress] = useState<PullProgress>({ stage: 'manifest', percent: 0 });
  const [status, setStatus] = useState<'pending' | 'loading' | 'done' | 'error'>(
    alreadyPulled ? 'done' : 'pending',
  );
  const [speedMeta, setSpeedMeta] = useState<string>();
  const lastSampleRef = useRef<{ time: number; doneMb: number } | null>(null);

  useEffect(() => {
    // Модель уже есть — сразу завершаем без скачивания
    if (alreadyPulled) {
      onComplete();
      return;
    }

    let active = true;
    const attach = async () => {
      const unlistenProgress = await listen<PullProgress>('ai-pull-progress', (event) => {
        if (!active) return;
        setStatus('loading');
        setProgress(event.payload);

        if (event.payload.stage !== 'download' || event.payload.done_mb === undefined) {
          setSpeedMeta(undefined);
          return;
        }

        const now = Date.now();
        const previous = lastSampleRef.current;
        lastSampleRef.current = { time: now, doneMb: event.payload.done_mb };

        if (!previous || event.payload.total_mb === undefined) return;

        const elapsedSeconds = (now - previous.time) / 1000;
        const transferredMb = event.payload.done_mb - previous.doneMb;
        if (elapsedSeconds <= 0 || transferredMb <= 0) return;

        const speed = transferredMb / elapsedSeconds;
        const remainingMb = Math.max(0, event.payload.total_mb - event.payload.done_mb);
        setSpeedMeta(`${speed.toFixed(1)} MB/s · ${formatEta(remainingMb / speed)}`);
      });
      const unlistenDone = await listen('ai-pull-done', () => {
        if (!active) return;
        setStatus('done');
        onComplete();
      });

      if (!startedRef.current) {
        startedRef.current = true;
        try {
          setStatus('loading');
          await aiApi.pullModelWithProgress(modelName);
        } catch {
          if (active) setStatus('error');
        }
      }

      return () => {
        unlistenProgress();
        unlistenDone();
      };
    };

    let cleanup: (() => void) | undefined;
    attach().then((fn) => { cleanup = fn; });
    return () => {
      active = false;
      cleanup?.();
    };
  }, [alreadyPulled, modelName, onComplete]);

  const icon = status === 'loading' ? '⟳' : status === 'done' ? '✓' : status === 'error' ? '✗' : '○';

  const stageText = alreadyPulled
    ? 'Модель уже загружена'
    : progress.stage === 'verify'
      ? 'Проверка слоёв…'
      : progress.stage === 'complete'
        ? 'Загрузка завершена'
        : progress.stage === 'download'
          ? 'Скачивание модели…'
          : 'Подготовка загрузки…';

  const subText =
    !alreadyPulled && progress.total_mb
      ? `${(progress.done_mb || 0).toFixed(0)} MB / ${(progress.total_mb / 1024).toFixed(progress.total_mb >= 1024 ? 1 : 0)} ${progress.total_mb >= 1024 ? 'GB' : 'MB'}`
      : undefined;

  return (
    <Win11StepRow
      icon={icon}
      text={`${modelName}: ${stageText}`}
      status={status}
      progress={status === 'done' ? 100 : progress.percent}
      subText={subText}
      metaText={status === 'loading' ? speedMeta : undefined}
    />
  );
}
