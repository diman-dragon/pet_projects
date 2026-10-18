// components/status/ModuleStatusBadge.tsx

import { useI18n } from '../../../../i18n/AppLanguageProvider';
import type { TranslationKey } from '../../../../i18n/translations';
import { ModuleStatus } from '../../types';
import { StatusDot } from '../shared/StatusDot';

const BADGE_STYLES: Record<ModuleStatus, string> = {
  online: 'bg-emerald-50 text-emerald-700 border-emerald-200/50',
  standby: 'bg-amber-50 text-amber-700 border-amber-200/50',
  offline: 'bg-black/[0.03] text-black/40 border-black/[0.06]',
};

const DOT_COLORS: Record<ModuleStatus, string> = {
  online: 'bg-emerald-500',
  standby: 'bg-amber-400',
  offline: 'bg-blue-400/40',
};

interface ModuleStatusBadgeProps {
  status: ModuleStatus;
}

/** Бейдж статуса модуля с цветным индикатором */
export function ModuleStatusBadge({ status }: ModuleStatusBadgeProps) {
  const { t } = useI18n();
  const label = t(`laila.moduleStatus.${status}` as TranslationKey);

  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-md border text-[length:var(--text-xs)] font-bold shrink-0 ${BADGE_STYLES[status]}`}
    >
      <StatusDot colorClass={DOT_COLORS[status]} pulse={status === 'online'} />
      {label}
    </span>
  );
}
