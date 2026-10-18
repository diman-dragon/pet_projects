// src/pages/knowledge/components/search/JurisdictionPills.tsx

import { useI18n } from '../../../../i18n/AppLanguageProvider';
import { Jurisdiction } from '../../types';
import { JURISDICTIONS } from '../../data';

interface JurisdictionPillsProps {
  value: Jurisdiction;
  onChange: (v: Jurisdiction) => void;
}

/** Горизонтальный ряд кнопок-пилюль для выбора юрисдикции */
export function JurisdictionPills({ value, onChange }: JurisdictionPillsProps) {
  const { t } = useI18n();

  return (
    <div className="flex items-center gap-1 overflow-x-auto">
      {JURISDICTIONS.map((j) => (
        <button
          key={j}
          onClick={() => onChange(j as Jurisdiction)}
          className={`shrink-0 px-2 py-0.5 rounded-md text-[length:var(--text-xs)] font-black uppercase tracking-wide transition-all duration-150
            ${
              value === j
                ? 'bg-[var(--app-accent)] text-white shadow-sm shadow-[#0067c0]/20'
                : 'text-black/40 hover:bg-black/[0.06] hover:text-black/60'
            }`}
        >
          {j === 'all' ? t('knowledge.jurisdiction.all') : j}
        </button>
      ))}
    </div>
  );
}
