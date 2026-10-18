/**
 * MultiModelPullStep — FIXED EVENT STRUCTURE
 *
 * CRITICAL BUG FIX:
 * Rust (ai_pull_progress.rs) эмитировал { status, completed, total, percent, digest }
 * но этот компонент ожидал { stage, percent, total_mb, done_mb }.
 * Из-за несовместимых структур прогресс-бар никогда не двигался.
 *
 * После исправления Rust теперь эмитирует правильный формат:
 *   stage:    'manifest' | 'download' | 'verify' | 'complete'
 *   percent:  number (0–100)
 *   total_mb: number | undefined
 *   done_mb:  number | undefined
 *
 * Остальные исправления:
 * - useEffect с models в deps → бесконечный цикл (оставлено исправленным)
 * - modelsRef для доступа к актуальному state в коллбэках
 */
import { useEffect, useRef, useState } from 'react';
import { listen } from '@tauri-apps/api/event';
import { aiApi } from '../../services/backend/aiApi';
import { Win11StepRow } from './Win11StepRow';
import { AI_MODELS } from '../../services/ai/config';
import { resetAvailabilityCache } from '../../services/rag/embedding/local';
import { useI18n } from '../../i18n/AppLanguageProvider';

/** Структура прогресса — совпадает с тем что теперь эмитирует Rust */
interface PullProgress {
  stage: 'manifest' | 'download' | 'verify' | 'complete';
  percent: number;
  total_mb?: number;
  done_mb?: number;
}

type ModelStatus = {
  name: string;
  status: 'pending' | 'loading' | 'done' | 'error';
  progress: PullProgress;
  speedMeta?: string;
};

function formatEta(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds <= 0) return 'ETA --';
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = Math.round(seconds % 60);
  return minutes > 0 ? `ETA ${minutes}m ${remainingSeconds}s` : `ETA ${remainingSeconds}s`;
}

interface MultiModelPullStepProps {
  chatModel: string;
  onComplete: () => void;
  chatModelPulled?: boolean;
  embedModelPulled?: boolean;
}

export function MultiModelPullStep({
  chatModel,
  onComplete,
  chatModelPulled = false,
  embedModelPulled = false,
}: MultiModelPullStepProps) {
  const { t } = useI18n();
  const startedRef = useRef(false);
  const onCompleteRef = useRef(onComplete);
  onCompleteRef.current = onComplete;

  const initialModels: ModelStatus[] = [
    {
      name: AI_MODELS.rag,
      status: embedModelPulled ? 'done' : 'pending',
      progress: { stage: 'manifest', percent: 0 },
    },
    {
      name: chatModel,
      status: chatModelPulled ? 'done' : 'pending',
      progress: { stage: 'manifest', percent: 0 },
    },
  ];

  const [models, setModels] = useState<ModelStatus[]>(initialModels);

  const modelsRef = useRef<ModelStatus[]>(initialModels);
  modelsRef.current = models;

  const currentIndexRef = useRef<number>(-1);
  const progressSamplesRef = useRef<Record<string, { time: number; doneMb: number }>>({});

  const updateModel = (nameOrIndex: string | number, patch: Partial<ModelStatus>) => {
    setModels((prev) => {
      const next = prev.map((m, i) => {
        const match = typeof nameOrIndex === 'string' ? m.name === nameOrIndex : i === nameOrIndex;
        return match ? { ...m, ...patch } : m;
      });
      modelsRef.current = next;
      return next;
    });
  };

  const startNextPending = async (active: { value: boolean }) => {
    const pending = modelsRef.current.findIndex((m) => m.status === 'pending');
    if (pending === -1) return;
    currentIndexRef.current = pending;
    updateModel(pending, { status: 'loading' });
    try {
      await aiApi.pullModelWithProgress(modelsRef.current[pending].name);
    } catch {
      if (active.value) {
        updateModel(pending, { status: 'error' });
      }
    }
  };

  useEffect(() => {
    if (modelsRef.current.every((m) => m.status === 'done')) {
      resetAvailabilityCache();
      onCompleteRef.current();
      return;
    }

    const active = { value: true };

    const attach = async () => {
      // FIX: слушаем event с правильными полями (stage, percent, total_mb, done_mb)
      const unlistenProgress = await listen<PullProgress>('ai-pull-progress', (event) => {
        if (!active.value) return;
        const idx = currentIndexRef.current;
        if (idx < 0) return;

        const modelName = modelsRef.current[idx]?.name;
        let speedMeta: string | undefined;

        // Вычисляем скорость и ETA при наличии done_mb
        if (
          modelName &&
          event.payload.stage === 'download' &&
          event.payload.done_mb !== undefined
        ) {
          const now = Date.now();
          const previous = progressSamplesRef.current[modelName];
          progressSamplesRef.current[modelName] = { time: now, doneMb: event.payload.done_mb };

          if (previous && event.payload.total_mb !== undefined) {
            const elapsedSeconds = (now - previous.time) / 1000;
            const transferredMb = event.payload.done_mb - previous.doneMb;
            if (elapsedSeconds > 0 && transferredMb > 0) {
              const speed = transferredMb / elapsedSeconds;
              const remainingMb = Math.max(0, event.payload.total_mb - event.payload.done_mb);
              speedMeta = `${speed.toFixed(1)} MB/s · ${formatEta(remainingMb / speed)}`;
            }
          }
        }

        updateModel(idx, {
          status: 'loading',
          progress: event.payload,
          speedMeta,
        });
      });

      const unlistenDone = await listen<{ model?: string; success?: boolean }>(
        'ai-pull-done',
        async (event) => {
          if (!active.value) return;
          const success = event.payload.success ?? true;
          const idx = currentIndexRef.current;
          if (idx < 0) return;

          updateModel(idx, {
            status: success ? 'done' : 'error',
            progress: {
              stage: success ? 'complete' : 'verify',
              percent: success ? 100 : (modelsRef.current[idx]?.progress.percent ?? 0),
            },
          });

          if (success) {
            const allDone = modelsRef.current.every(
              (m, i) => (i === idx ? true : m.status === 'done'),
            );
            if (allDone) {
              resetAvailabilityCache();
              onCompleteRef.current();
            } else {
              await startNextPending(active);
            }
          }
        },
      );

      if (!startedRef.current) {
        startedRef.current = true;
        await startNextPending(active);
      }

      return () => {
        unlistenProgress();
        unlistenDone();
      };
    };

    let cleanup: (() => void) | undefined;
    attach().then((fn) => {
      cleanup = fn;
    });

    return () => {
      active.value = false;
      cleanup?.();
    };
    // IMPORTANT: `models` намеренно НЕ в deps — иначе бесконечный цикл
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="space-y-2">
      {models.map((model) => {
        const icon =
          model.status === 'loading'
            ? '⟳'
            : model.status === 'done'
              ? '✓'
              : model.status === 'error'
                ? '✗'
                : '○';

        const stageText =
          model.status === 'done'
            ? t('setup.model.done')
            : model.status === 'error'
              ? t('setup.model.error')
              : model.progress.stage === 'verify'
                ? t('setup.model.verify')
                : model.progress.stage === 'complete'
                  ? t('setup.model.complete')
                  : model.progress.stage === 'download'
                    ? t('setup.model.downloading')
                    : t('setup.model.waiting');

        const subText =
          model.status === 'loading' && model.progress.total_mb
            ? `${(model.progress.done_mb ?? 0).toFixed(0)} MB / ${
                model.progress.total_mb >= 1024
                  ? (model.progress.total_mb / 1024).toFixed(1) + ' GB'
                  : model.progress.total_mb.toFixed(0) + ' MB'
              }`
            : undefined;

        return (
          <Win11StepRow
            key={model.name}
            icon={icon}
            text={`${model.name}: ${stageText}`}
            status={
              model.status === 'pending'
                ? 'pending'
                : model.status === 'done'
                  ? 'done'
                  : model.status === 'error'
                    ? 'error'
                    : 'loading'
            }
            progress={model.status === 'done' ? 100 : model.progress.percent}
            subText={subText}
            metaText={model.status === 'loading' ? model.speedMeta : undefined}
          />
        );
      })}
    </div>
  );
}
