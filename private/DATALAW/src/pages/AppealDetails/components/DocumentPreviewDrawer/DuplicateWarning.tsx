import React from 'react';
import { Copy } from 'lucide-react';
import { useI18n } from '../../../../i18n/AppLanguageProvider';
import type { AppealDocument } from '../../../../types/entities';

interface DuplicateWarningProps {
  dbDoc: AppealDocument;
}

export const DuplicateWarning: React.FC<DuplicateWarningProps> = ({ dbDoc }) => {
  const { t } = useI18n();
  const shortPath = dbDoc.duplicate_of_path
    ? (dbDoc.duplicate_of_path.split('/').pop()?.split('\\').pop() ?? dbDoc.duplicate_of_path)
    : '';

  return (
    <div className="mx-4 mb-3 rounded-xl border border-amber-200 bg-amber-50 px-3 py-2.5 flex gap-2">
      <Copy size={13} className="text-amber-600 shrink-0 mt-0.5" />
      <div className="min-w-0">
        <p className="text-[length:var(--text-xs)] font-black uppercase tracking-widest text-amber-700 mb-0.5">
          {t('documentPreview.duplicate')}
        </p>
        <p className="text-[length:var(--text-xs)] text-amber-700/80 leading-[var(--leading-body)]">
          {t('documentPreview.duplicateBody')}
          {dbDoc.duplicate_appeal_id != null && (
            <>
              {' '}
              <span className="font-mono font-bold">#{dbDoc.duplicate_appeal_id}</span>
            </>
          )}
        </p>
        {shortPath && (
          <p
            className="text-[length:var(--text-xs)] font-mono text-amber-600/70 truncate mt-0.5"
            title={dbDoc.duplicate_of_path ?? ''}
          >
            {shortPath}
          </p>
        )}
      </div>
    </div>
  );
};
