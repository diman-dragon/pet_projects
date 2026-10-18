import React from 'react';
import { useI18n } from '../../../../i18n/AppLanguageProvider';
import type { TranslationKey } from '../../../../i18n/translations';
import { STATUS_CONFIG } from '../../config';

interface StatusBadgeProps {
  status: string;
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({ status }) => {
  const { t } = useI18n();
  const cfg = STATUS_CONFIG[status] ?? { color: 'bg-black/[0.06] text-black/50' };
  const key = `status.${status.toLowerCase()}` as TranslationKey;
  const translated = t(key);
  const label = translated === key ? status : translated;

  return (
    <span
      className={`h-6 flex items-center px-2.5 rounded-md text-[length:var(--text-xs)] font-bold border ${cfg.color}`}
    >
      {label}
    </span>
  );
};
