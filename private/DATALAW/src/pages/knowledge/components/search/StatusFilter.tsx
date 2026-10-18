// src/pages/knowledge/components/search/StatusFilter.tsx

import { useI18n } from '../../../../i18n/AppLanguageProvider';
import type { TranslationKey } from '../../../../i18n/translations';
import { DocStatus } from '../../types';
import { STATUS_OPTIONS } from '../../data';

interface StatusFilterProps {
  value: DocStatus;
  onChange: (v: DocStatus) => void;
}

const STATUS_LABEL_KEY: Record<string, TranslationKey> = {
  all: 'knowledge.jurisdiction.all',
  active: 'knowledge.caseStatus.active',
  complaint: 'knowledge.caseStatus.complaint',
  decided: 'knowledge.caseStatus.decided',
};

/** Фильтр по статусу документа (вступило в силу / жалоба / решено) */
export function StatusFilter({ value, onChange }: StatusFilterProps) {
  const { t } = useI18n();

  return (
    <div className="flex items-center gap-1 flex-wrap">
      <span className="text-[length:var(--text-xs)] text-black/30 font-bold uppercase tracking-wide mr-1">
        {t('knowledge.filter.statusLabel')}
      </span>
      {STATUS_OPTIONS.map(({ value: v }) => (
        <button
          key={v}
          onClick={() => onChange(v as DocStatus)}
          className={`px-2 py-0.5 rounded-md text-[length:var(--text-xs)] font-bold transition-all duration-150
            ${
              value === v
                ? 'bg-[var(--app-accent)] text-white'
                : 'text-black/40 hover:bg-black/[0.06]'
            }`}
        >
          {t(STATUS_LABEL_KEY[v])}
        </button>
      ))}
    </div>
  );
}
