/**
 * OllamaStartStep
 *
 * Запускает Ollama-сервер в фоне и ждёт пока он станет доступен на :11434.
 * Показывает прогресс пользователю без мигания терминала.
 *
 * Проблема которую решает:
 * - После установки/перед pull моделей Ollama нужно запустить
 * - Раньше pull начинался до готовности сервера → ошибки соединения
 * - ai_start_sidecar теперь использует CREATE_NO_WINDOW → нет мигания
 */
import { useEffect, useRef, useState } from 'react';
import { aiApi } from '../../services/backend/aiApi';
import { Win11StepRow } from './Win11StepRow';
import { useI18n } from '../../i18n/AppLanguageProvider';

interface OllamaStartStepProps {
  onComplete: () => void;
  onError?: (msg: string) => void;
  /** Если уже запущен — сразу вызвать onComplete */
  alreadyRunning?: boolean;
}

const MAX_WAIT_MS = 45_000;   // 45 сек максимум
const POLL_INTERVAL_MS = 1_500; // проверяем каждые 1.5 сек

export function OllamaStartStep({
  onComplete,
  onError,
  alreadyRunning = false,
}: OllamaStartStepProps) {
  const { t } = useI18n();
  const startedRef = useRef(false);
  const onCompleteRef = useRef(onComplete);
  const onErrorRef = useRef(onError);
  onCompleteRef.current = onComplete;
  onErrorRef.current = onError;

  const [status, setStatus] = useState<'pending' | 'loading' | 'done' | 'error'>(
    alreadyRunning ? 'done' : 'pending',
  );
  const [attempt, setAttempt] = useState(0);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  useEffect(() => {
    if (alreadyRunning) {
      onCompleteRef.current();
      return;
    }

    if (startedRef.current) return;
    startedRef.current = true;

    let cancelled = false;
    let pollTimer: ReturnType<typeof setTimeout> | null = null;

    const run = async () => {
      try {
        setStatus('loading');
        // Запускаем Ollama (скрытый процесс, без мигания терминала)
        await aiApi.startEngine();
      } catch {
        // startEngine может вернуть ошибку если ollama не найдена —
        // продолжаем polling всё равно (может быть запущена другим способом)
      }

      // Ждём готовности сервера
      const startedAt = Date.now();
      let pollCount = 0;

      const poll = async () => {
        if (cancelled) return;

        try {
          const s = await aiApi.getStatus();
          if (s.running) {
            if (!cancelled) {
              setStatus('done');
              onCompleteRef.current();
            }
            return;
          }
        } catch {
          // Ещё не отвечает — ждём
        }

        pollCount++;
        setAttempt(pollCount);

        if (Date.now() - startedAt > MAX_WAIT_MS) {
          if (!cancelled) {
            const msg = t('setup.ollama.startTimeout');
            setStatus('error');
            setErrorMsg(msg);
            onErrorRef.current?.(msg);
          }
          return;
        }

        if (!cancelled) {
          pollTimer = setTimeout(poll, POLL_INTERVAL_MS);
        }
      };

      // Первый poll через короткую паузу (даём процессу время стартовать)
      pollTimer = setTimeout(poll, 1500);
    };

    run();

    return () => {
      cancelled = true;
      if (pollTimer) clearTimeout(pollTimer);
    };
  }, [alreadyRunning, t]);

  const text = alreadyRunning
    ? t('setup.ollama.alreadyRunning')
    : status === 'done'
      ? t('setup.ollama.started')
      : status === 'error'
        ? t('setup.ollama.startFailed')
        : status === 'loading'
          ? attempt > 0
            ? t('setup.ollama.starting')
            : t('setup.ollama.launching')
          : t('setup.ollama.launching');

  const icon =
    status === 'loading' ? '⟳'
    : status === 'done'  ? '✓'
    : status === 'error' ? '✗'
    : '○';

  return (
    <div className="space-y-2">
      <Win11StepRow
        icon={icon}
        text={text}
        status={status}
        indeterminate={status === 'loading'}
      />
      {errorMsg && (
        <p className="pl-8 text-[length:var(--text-xs)] text-[var(--app-danger)]">{errorMsg}</p>
      )}
    </div>
  );
}
