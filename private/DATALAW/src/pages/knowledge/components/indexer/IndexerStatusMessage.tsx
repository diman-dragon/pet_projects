import { useI18n } from '../../../../i18n/AppLanguageProvider';
import type { IndexerStatus } from '../../../../store/indexerStore';

export function IndexerStatusMessage({
  status,
  error,
  model,
}: {
  status: IndexerStatus;
  error: string | null;
  model: string;
}) {
  const { t } = useI18n();

  if (status === 'done' && !error) {
    return (
      <p className="indexer-panel__message indexer-panel__message--success">
        {t('knowledge.indexer.status.done')}
      </p>
    );
  }

  if (status === 'no-model') {
    return (
      <div className="indexer-panel__message indexer-panel__message--error">
        <p>{t('knowledge.indexer.status.noModel')}</p>
        <pre className="indexer-panel__code">{model}</pre>
      </div>
    );
  }

  if (error) {
    return (
      <div className="indexer-panel__message indexer-panel__message--error">
        <p className="whitespace-pre-wrap font-[var(--font-mono)] text-[length:var(--text-xs)]">
          ! {error}
        </p>
      </div>
    );
  }

  return null;
}
