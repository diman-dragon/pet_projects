/**
 * OllamaInstallStep — FIXED
 *
 * BUG FIX: все строки хардкодированы на русском.
 * РЕШЕНИЕ: использует useI18n() для всех текстов.
 */
import { useEffect, useRef, useState } from 'react';
import { aiApi } from '../../services/backend/aiApi';
import { Win11StepRow } from './Win11StepRow';
import { useI18n } from '../../i18n/AppLanguageProvider';

interface OllamaInstallStepProps {
  onComplete: () => void;
  /** Если true — Ollama уже установлен, шаг сразу показывает done и вызывает onComplete. */
  alreadyInstalled?: boolean;
}

export function OllamaInstallStep({ onComplete, alreadyInstalled = false }: OllamaInstallStepProps) {
  const { t } = useI18n();
  const startedRef = useRef(false);
  const [status, setStatus] = useState<'pending' | 'loading' | 'done' | 'error'>(
    alreadyInstalled ? 'done' : 'pending',
  );
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (alreadyInstalled) {
      onComplete();
      return;
    }

    if (startedRef.current) return;
    startedRef.current = true;

    const run = async () => {
      try {
        setStatus('loading');
        const result = await aiApi.installOllama();
        if (!result.installed) throw new Error(result.message);
        setStatus('done');
        onComplete();
      } catch (err: any) {
        setStatus('error');
        setError(err?.message ?? t('setup.ollama.installFailed'));
      }
    };

    run();
  }, [alreadyInstalled, onComplete, t]);

  const text = alreadyInstalled
    ? t('setup.ollama.alreadyInstalled')
    : status === 'done'
      ? t('setup.ollama.installed')
      : status === 'loading'
        ? t('setup.ollama.installing')
        : t('setup.ollama.installPending');

  return (
    <div className="space-y-3">
      <Win11StepRow
        icon={status === 'loading' ? '⟳' : status === 'done' ? '✓' : status === 'error' ? '✗' : '○'}
        text={text}
        status={status}
        progress={status === 'loading' ? undefined : undefined}
        indeterminate={status === 'loading'}
      />
      {error && (
        <p className="pl-8 text-[length:var(--text-xs)] text-[var(--app-danger)]">{error}</p>
      )}
    </div>
  );
}
