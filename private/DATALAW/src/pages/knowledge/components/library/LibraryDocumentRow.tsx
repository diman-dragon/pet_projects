import { useState } from 'react';
import {
  AlertCircle,
  CheckCircle2,
  Circle,
  Clock,
  RefreshCw,
  Trash2,
} from 'lucide-react';
import { useI18n } from '../../../../i18n/AppLanguageProvider';
import { appLanguageToBcp47 } from '../../../../i18n/config';
import type { LibraryDocumentWithStatus } from '../../../../services/rag/knowledgeLibrary/types';
import { fileIcon, formatDate } from './libraryDocumentUtils';

function StatusIcon({ doc }: { doc: LibraryDocumentWithStatus }) {
  if (doc.isIndexed) return <CheckCircle2 size={13} className="text-emerald-500" />;
  if (doc.indexJobStatus === 'running') {
    return <RefreshCw size={13} className="animate-spin text-[var(--app-accent)]" />;
  }
  if (doc.indexJobStatus === 'error') return <AlertCircle size={13} className="text-red-400" />;
  return <Circle size={13} className="text-black/20" />;
}

function IndexingProgress({ doc }: { doc: LibraryDocumentWithStatus }) {
  const { t } = useI18n();
  const chunks = doc.chunk_count ?? 0;
  const embeddings = doc.embedding_count ?? 0;

  if (doc.isIndexed) {
    return (
      <span className="text-[9px] text-emerald-600 flex items-center gap-0.5">
        <CheckCircle2 size={9} />
        {embeddings > 0
          ? t('knowledge.library.embeddingCount', { count: embeddings })
          : t('knowledge.library.status.indexed')}
      </span>
    );
  }

  if (doc.indexJobStatus === 'running' || chunks > 0) {
    return (
      <span className="text-[9px] text-[var(--app-accent)] flex items-center gap-0.5">
        <RefreshCw size={9} className="animate-spin" />
        {chunks > 0
          ? t('knowledge.library.chunkCount', { count: chunks })
          : t('knowledge.library.status.indexing')}
      </span>
    );
  }

  return (
    <span className="text-[9px] text-black/25 flex items-center gap-0.5">
      <Circle size={9} />
      {t('knowledge.library.status.pending')}
    </span>
  );
}

export function LibraryDocumentRow({
  doc,
  onDelete,
  onReindex,
}: {
  doc: LibraryDocumentWithStatus;
  onDelete: (id: number) => Promise<void> | void;
  onReindex: (filePath: string, id: number) => void;
}) {
  const { t, language } = useI18n();
  const locale = appLanguageToBcp47(language);
  const [deletingId, setDeletingId] = useState<number | null>(null);
  const filename = doc.file_path.split(/[/\\]/).pop() ?? doc.file_path;
  const isDeleting = deletingId === doc.id;

  const handleDelete = async () => {
    setDeletingId(doc.id);
    await onDelete(doc.id);
    setDeletingId(null);
  };

  return (
    <div
      className={`
        group flex items-center gap-3 rounded-xl border border-black/[0.06] bg-white p-3
        transition-all duration-150 hover:border-black/[0.10] hover:shadow-sm
        ${isDeleting ? 'opacity-50 pointer-events-none' : ''}
      `}
    >
      <span className="shrink-0 text-[10px] font-semibold text-black/50">{fileIcon(doc.file_type)}</span>

      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-1.5">
          <StatusIcon doc={doc} />
          <span
            className="text-[length:var(--text-xs)] font-semibold text-black/75 truncate"
            title={doc.file_path}
          >
            {doc.title || filename}
          </span>
        </div>

        <div className="mt-0.5 flex items-center gap-2 flex-wrap">
          {doc.category ? (
            <span className="rounded bg-black/[0.05] px-1.5 py-0.5 text-[9px] text-black/40">
              {doc.category}
            </span>
          ) : null}
          <IndexingProgress doc={doc} />
          <span className="text-[9px] text-black/25 flex items-center gap-0.5">
            <Clock size={9} />
            {formatDate(doc.indexed_at ?? doc.created_at, locale)}
          </span>
        </div>
      </div>

      <div className="flex shrink-0 items-center gap-1 opacity-0 transition-opacity group-hover:opacity-100">
        <button
          onClick={() => onReindex(doc.file_path, doc.id)}
          className="flex size-7 items-center justify-center rounded-lg text-black/30 transition-colors hover:bg-[var(--app-accent)]/10 hover:text-[var(--app-accent)]"
          title={t('knowledge.library.reindexTitle')}
        >
          <RefreshCw size={12} />
        </button>
        <button
          onClick={() => void handleDelete()}
          disabled={isDeleting}
          className="flex size-7 items-center justify-center rounded-lg text-black/30 transition-colors hover:bg-red-50 hover:text-red-400"
          title={t('knowledge.library.deleteTitle')}
        >
          <Trash2 size={12} />
        </button>
      </div>
    </div>
  );
}
