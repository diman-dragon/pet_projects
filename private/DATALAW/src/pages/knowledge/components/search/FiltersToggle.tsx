// src/pages/knowledge/components/search/FiltersToggle.tsx

import { useI18n } from '../../../../i18n/AppLanguageProvider';

interface FiltersToggleProps {
  isOpen: boolean;
  onToggle: () => void;
}

/** Кнопка раскрытия панели фильтров */
export function FiltersToggle({ isOpen, onToggle }: FiltersToggleProps) {
  const { t } = useI18n();

  return (
    <button
      onClick={onToggle}
      className={`shrink-0 px-2 py-1 rounded-lg text-[length:var(--text-xs)] font-bold transition-all flex items-center gap-1
        ${
          isOpen
            ? 'bg-[var(--app-accent)]/10 text-[var(--app-accent)]'
            : 'text-black/30 hover:text-black/60 hover:bg-black/[0.04]'
        }`}
    >
      {t('knowledge.filter.toggle')}
    </button>
  );
}
