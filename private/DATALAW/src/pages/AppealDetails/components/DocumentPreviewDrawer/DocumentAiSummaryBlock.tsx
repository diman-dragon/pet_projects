import React, { useState } from 'react';
import { Sparkles, ChevronDown, ChevronUp } from 'lucide-react';
import { useI18n } from '../../../../i18n/AppLanguageProvider';
import type { AppealDocument } from '../../../../types/entities';

interface DocumentAiSummaryBlockProps {
  dbDoc: AppealDocument;
}

export const DocumentAiSummaryBlock: React.FC<DocumentAiSummaryBlockProps> = ({ dbDoc }) => {
  const { t } = useI18n();
  const [expanded, setExpanded] = useState(false);
  const keyPoints: string[] = (() => {
    try {
      return dbDoc.key_points ? JSON.parse(dbDoc.key_points) : [];
    } catch {
      return [];
    }
  })();

  return (
    <div className="mx-4 mb-3 rounded-xl border border-violet-200 bg-violet-50/60 overflow-hidden">
      <button
        className="w-full flex items-center gap-2 px-3 py-2.5 text-left"
        onClick={() => setExpanded((v) => !v)}
      >
        <div className="w-5 h-5 rounded-md bg-violet-600 flex items-center justify-center shrink-0">
          <Sparkles size={10} className="text-white" />
        </div>
        <span className="text-[length:var(--text-xs)] font-black uppercase tracking-widest text-violet-700 flex-1">
          {t('documentPreview.aiSummary')}
        </span>
        {expanded ? (
          <ChevronUp size={12} className="text-violet-400" />
        ) : (
          <ChevronDown size={12} className="text-violet-400" />
        )}
      </button>

      {expanded && (
        <div className="px-3 pb-3 space-y-2 border-t border-violet-200">
          <p className="text-[length:var(--text-xs)] text-violet-900/80 leading-[var(--leading-body)] pt-2">
            {dbDoc.summary}
          </p>
          {keyPoints.length > 0 && (
            <div className="space-y-1.5 pt-1">
              {keyPoints.map((p, i) => (
                <div key={i} className="flex gap-2 items-start">
                  <span className="shrink-0 w-4 h-4 rounded-full bg-violet-200 text-violet-700 text-[length:var(--text-xs)] font-bold flex items-center justify-center mt-0.5">
                    {i + 1}
                  </span>
                  <p className="text-[length:var(--text-xs)] text-violet-800/80 leading-[var(--leading-body)]">
                    {p}
                  </p>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};
