import { Button } from '../../../../components/ui/form/Button';
import { AppPaths } from '../../../../services/config/AppPaths';
import { useI18n } from '../../../../i18n/AppLanguageProvider';
import { useIndexerStore } from '../../../../store/indexerStore';
import { EmbeddingStatusBanner } from './EmbeddingStatusBanner';
import { IndexerProgress } from './IndexerProgress';
import { IndexerStatusMessage } from './IndexerStatusMessage';
import { useEmbeddingBanner } from './useEmbeddingBanner';

function StatBadge({ label, value }: { label: string; value: number | undefined | null }) {
  return (
    <div className="stat-badge">
      <span className="stat-badge__value">{(value ?? 0).toLocaleString()}</span>
      <span className="stat-badge__label">{label}</span>
    </div>
  );
}

export function IndexerPanel({ onStartIndexing }: { onStartIndexing: () => void }) {
  const { t } = useI18n();
  const { status, progress, error, stats } = useIndexerStore();
  const abort = useIndexerStore((state) => state.abort);
  const forceReembed = useIndexerStore((state) => state.forceReembed);
  const isRunning = status === 'running' || status === 'checking';
  const { banner, bannerBusy, embeddingModel } = useEmbeddingBanner(forceReembed, stats?.embeddingsCount);

  return (
    <div className="indexer-panel">
      <div className="indexer-panel__header">
        <div className="indexer-panel__title">
          <span className="indexer-panel__icon">KB</span>
          <div>
            <h3>{t('knowledge.indexer.title')}</h3>
            <p className="indexer-panel__subtitle">
              {t('knowledge.indexer.backendLabel')}: <code>{embeddingModel}</code>
            </p>
          </div>
        </div>

        <div className="flex flex-wrap gap-2">
          {isRunning ? (
            <Button variant="danger" size="sm" onClick={abort}>
              {t('knowledge.indexer.abort')}
            </Button>
          ) : (
            <>
              <Button
                variant="primary"
                size="sm"
                onClick={onStartIndexing}
                disabled={status === 'no-model'}
              >
                {status === 'done'
                  ? t('knowledge.indexer.updateIndex')
                  : t('knowledge.indexer.startIndexing')}
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={forceReembed}
                disabled={status === 'no-model'}
                title={t('knowledge.indexer.reindexAllHint')}
              >
                {t('knowledge.indexer.reindexAll')}
              </Button>
            </>
          )}
        </div>
      </div>

      {banner ? <EmbeddingStatusBanner banner={banner} busy={bannerBusy} /> : null}

      {stats ? (
        <div className="indexer-panel__stats">
          <StatBadge label={t('knowledge.indexer.stats.docs')} value={stats.documentsCount} />
          <StatBadge label={t('knowledge.indexer.stats.chunks')} value={stats.chunksCount} />
          <StatBadge label={t('knowledge.indexer.stats.embeddings')} value={stats.embeddingsCount} />
        </div>
      ) : null}

      <p className="label-mono-id break-all px-1 pt-2">
        {t('knowledge.indexer.kbPathLabel')}:{' '}
        {AppPaths.getKbPath() ?? t('knowledge.indexer.kbPathMissing')}
      </p>

      <IndexerProgress status={status} progress={progress} />
      <IndexerStatusMessage status={status} error={error} model={embeddingModel} />
    </div>
  );
}
