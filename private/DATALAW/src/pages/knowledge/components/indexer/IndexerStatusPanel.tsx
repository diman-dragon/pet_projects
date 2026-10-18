import { useState, type ReactNode } from 'react';
import { useEmbeddingStatus } from '../../hooks/useEmbeddingStatus';
import { useI18n } from '@/i18n/AppLanguageProvider';
import { useIndexerStore } from '../../../../store/indexerStore';
import { aiApi } from '../../../../services/backend/aiApi';
import type { TranslationKey } from '../../../../i18n/translations';

function StatusCard({
  tone,
  text,
  children,
}: {
  tone: 'green' | 'amber' | 'blue';
  text: string;
  children?: ReactNode;
}) {
  const tones = {
    green: 'text-green-600 bg-green-50',
    amber: 'text-amber-600 bg-amber-50',
    blue: 'text-blue-600 bg-blue-50',
  } as const;

  const dotTones = {
    green: 'bg-green-500',
    amber: 'bg-amber-500',
    blue: 'bg-blue-500',
  } as const;

  return (
    <div className={`flex items-start gap-2 rounded-md px-4 py-3 text-sm ${tones[tone]}`}>
      <span className={`mt-1.5 h-2 w-2 rounded-full ${dotTones[tone]}`} />
      <div className="flex-1">
        <p>{text}</p>
        {children}
      </div>
    </div>
  );
}

export function IndexerStatusPanel() {
  const { t } = useI18n();
  const { mode, model, checked } = useEmbeddingStatus();
  const forceReembed = useIndexerStore((state) => state.forceReembed);
  const [pulling, setPulling] = useState(false);

  const handlePullModel = async () => {
    setPulling(true);
    try {
      await aiApi.pullModelWithProgress(model);
    } finally {
      setPulling(false);
    }
  };

  if (!checked) {
    return <div className="px-4 py-3 text-sm text-[var(--app-text-muted)]">{t('common.loading' as TranslationKey)}</div>;
  }

  if (mode === 'real') {
    return (
      <StatusCard tone="green" text={t('knowledge.semantic.active' as TranslationKey, { model })} />
    );
  }

  if (mode === 'unavailable') {
    return (
      <StatusCard tone="amber" text={t('knowledge.semantic.unavailable' as TranslationKey)}>
        <button
          className="mt-1 text-xs underline hover:no-underline disabled:opacity-50"
          onClick={() => void handlePullModel()}
          disabled={pulling}
        >
          {pulling ? '...' : t('knowledge.semantic.download' as TranslationKey)}
        </button>
      </StatusCard>
    );
  }

  return (
    <StatusCard tone="blue" text={t('knowledge.semantic.hashMode' as TranslationKey)}>
        <button
          className="mt-1 text-xs underline hover:no-underline"
          onClick={() => void forceReembed()}
        >
          {t('knowledge.semantic.reindex' as TranslationKey)}
        </button>
    </StatusCard>
  );
}
