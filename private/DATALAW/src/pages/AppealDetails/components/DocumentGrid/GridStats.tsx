import React from 'react';
import { Sparkles, Copy, FileWarning } from 'lucide-react';
import { useI18n } from '../../../../i18n/AppLanguageProvider';
import type { ScannedFile } from '../../types';
import type { AppealDocument } from '../../../../types/entities';

interface GridStatsProps {
  filteredCount: number;
  totalCount: number;
  lastScanned: Date | null;
  localeTag: string;
  docMap: Map<string, AppealDocument>;
  files: ScannedFile[];
}

export const GridStats: React.FC<GridStatsProps> = ({
  filteredCount,
  totalCount,
  lastScanned,
  localeTag,
  docMap,
  files,
}) => {
  const { t } = useI18n();
  const duplicatesCount = Array.from(docMap.values()).filter((d) => d.duplicate_of_path).length;
  const withSummaryCount = Array.from(docMap.values()).filter((d) => d.summary).length;
  const nonStandardCount = files.filter((f) => !f.isStandard).length;

  return (
    <div className="flex items-center gap-3 mb-2 shrink-0 flex-wrap">
      <span className="text-[length:var(--text-xs)] font-mono text-black/30">
        {t('documentGrid.fileCount', { current: filteredCount, total: totalCount })}
      </span>
      {lastScanned && (
        <span className="text-[length:var(--text-xs)] font-mono text-black/20">
          · {lastScanned.toLocaleTimeString(localeTag, { hour: '2-digit', minute: '2-digit' })}
        </span>
      )}
      {withSummaryCount > 0 && (
        <span className="flex items-center gap-1 text-[length:var(--text-xs)] text-violet-600 font-semibold">
          <Sparkles size={9} />
          {t('documentGrid.aiLabeled', { count: withSummaryCount })}
        </span>
      )}
      {duplicatesCount > 0 && (
        <span className="flex items-center gap-1 text-[length:var(--text-xs)] text-amber-600 font-semibold">
          <Copy size={9} />
          {t('documentGrid.duplicatesShort', { count: duplicatesCount })}
        </span>
      )}
      {nonStandardCount > 0 && (
        <span className="ml-auto flex items-center gap-1 text-[length:var(--text-xs)] text-amber-600 font-semibold">
          <FileWarning size={10} />
          {t('documentGrid.nonStandardShort', { count: nonStandardCount })}
        </span>
      )}
    </div>
  );
};
