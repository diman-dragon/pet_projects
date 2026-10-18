import { useI18n } from '../../../../i18n/AppLanguageProvider';
import { useIndexerStore } from '../../../../store/indexerStore';

function ProgressBar({
  label,
  pct,
  detail,
  fallbackLabel,
}: {
  label?: string;
  pct: number | null;
  detail?: string;
  fallbackLabel: string;
}) {
  return (
    <div className="mt-1.5">
      {label ? <p className="indexer-panel__current-file">{label}</p> : null}
      <div className="mt-1">
        <div className="indexer-panel__bar-wrap">
          <div
            className="indexer-panel__bar"
            style={{ width: `${pct ?? 8}%`, opacity: pct === null ? 0.4 : 1 }}
          />
        </div>
        <span className="indexer-panel__bar-label">{detail ?? fallbackLabel}</span>
      </div>
    </div>
  );
}

export function IndexerProgress({
  status,
  progress,
}: {
  status: ReturnType<typeof useIndexerStore.getState>['status'];
  progress: ReturnType<typeof useIndexerStore.getState>['progress'];
}) {
  const { t } = useI18n();
  const isRunning = status === 'running' || status === 'checking';
  if (!isRunning) {
    return null;
  }

  const docPct = progress?.docChunksTotal
    ? Math.min(100, Math.round((progress.docChunksDone / progress.docChunksTotal) * 100))
    : null;
  const filePct = progress?.filesTotal
    ? Math.min(100, Math.round((progress.filesDone / progress.filesTotal) * 100))
    : null;
  const chunkPct = progress?.chunksTotal
    ? Math.min(100, Math.round((progress.chunksDone / progress.chunksTotal) * 100))
    : null;

  return (
    <div className="indexer-panel__progress">
      {status === 'checking' ? (
        <p className="indexer-panel__status-text">
          <span className="spinner" /> {t('knowledge.indexer.status.checking')}
        </p>
      ) : (
        <>
          {progress?.currentFile ? (
            <ProgressBar
              label={`FILE ${progress.currentFile}`}
              pct={docPct}
              fallbackLabel={t('knowledge.indexer.status.processing')}
              detail={
                docPct !== null
                  ? t('knowledge.indexer.status.docProgress', {
                      percent: docPct,
                      done: progress.docChunksDone,
                      total: progress.docChunksTotal,
                    })
                  : undefined
              }
            />
          ) : null}
          <ProgressBar
            pct={filePct ?? chunkPct}
            fallbackLabel={t('knowledge.indexer.status.processing')}
            detail={
              filePct !== null
                ? t('knowledge.indexer.status.fileProgress', {
                    percent: filePct,
                    done: progress?.filesDone ?? 0,
                    total: progress?.filesTotal ?? 0,
                  })
                : chunkPct !== null
                  ? t('knowledge.indexer.status.chunkProgress', {
                      percent: chunkPct,
                      done: progress?.chunksDone ?? 0,
                      total: progress?.chunksTotal ?? 0,
                    })
                  : undefined
            }
          />
        </>
      )}
    </div>
  );
}
