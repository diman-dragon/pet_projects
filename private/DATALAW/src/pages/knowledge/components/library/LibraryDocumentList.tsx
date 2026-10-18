import { useI18n } from '../../../../i18n/AppLanguageProvider';
import type { LibraryDocumentWithStatus } from '../../../../services/rag/knowledgeLibrary/types';
import { LibraryDocumentEmptyState } from './LibraryDocumentEmptyState';
import { LibraryDocumentRow } from './LibraryDocumentRow';

interface LibraryDocumentListProps {
  documents?: LibraryDocumentWithStatus[];
  loading: boolean;
  onDelete: (id: number) => void;
  onReindex: (filePath: string, id: number) => void;
}

export function LibraryDocumentList({
  documents = [],
  loading,
  onDelete,
  onReindex,
}: LibraryDocumentListProps) {
  const { t } = useI18n();

  if (loading) {
    return (
      <div className="space-y-2">
        {[1, 2, 3].map((index) => (
          <div key={index} className="h-14 rounded-xl bg-black/[0.03] animate-pulse" />
        ))}
      </div>
    );
  }

  if (documents.length === 0) {
    return <LibraryDocumentEmptyState />;
  }

  const indexedCount = documents.filter((doc) => doc.isIndexed).length;

  return (
    <div>
      <div className="mb-4 flex items-center justify-between text-[10px] text-black/35">
        <span>{t('knowledge.library.documentCount', { count: documents.length })}</span>
        <span>{t('knowledge.library.indexedCount', { count: indexedCount })}</span>
      </div>

      <div className="space-y-2">
        {documents.map((doc) => (
          <LibraryDocumentRow
            key={doc.id}
            doc={doc}
            onDelete={onDelete}
            onReindex={onReindex}
          />
        ))}
      </div>
    </div>
  );
}
