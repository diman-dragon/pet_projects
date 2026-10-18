// components/status/SystemIdentityBlock.tsx

import { UiText } from '../../../../components/ui/atoms/UiText';
import { useI18n } from '../../../../i18n/AppLanguageProvider';
import { LailaOrb } from '../orb/LailaOrb';
import { StatusDot } from '../shared/StatusDot';

interface SystemIdentityBlockProps {
  online: number;
  standby: number;
  offline: number;
}

/** Блок идентификации системы: орб, название, версия, счётчики */
export function SystemIdentityBlock({ online, standby, offline }: SystemIdentityBlockProps) {
  const { t } = useI18n();

  return (
    <div className="flex items-center gap-5 p-5 rounded-2xl bg-gradient-to-br from-violet-50/80 to-blue-50/60 border border-violet-100/60">
      <LailaOrb size="lg" />

      <div className="flex-1">
        <UiText
          as="h2"
          size="md"
          weight="bold"
          tone="secondary"
          heading
          className="tracking-tight text-black/80"
        >
          {t('laila.system.title')}
        </UiText>
        <UiText as="p" size="xs" tone="muted" className="mt-0.5">
          {t('laila.system.subtitle')}
        </UiText>

        <div className="flex items-center gap-4 mt-3">
          <div className="flex items-center gap-1.5">
            <StatusDot colorClass="bg-emerald-500" />
            <UiText as="span" size="xs" weight="bold" className="text-emerald-600">
              {t('laila.system.online', { count: online })}
            </UiText>
          </div>
          <div className="flex items-center gap-1.5">
            <StatusDot colorClass="bg-amber-400" />
            <UiText as="span" size="xs" weight="bold" className="text-amber-600">
              {t('laila.system.standby', { count: standby })}
            </UiText>
          </div>
          <div className="flex items-center gap-1.5">
            <StatusDot colorClass="bg-black/20" />
            <UiText as="span" size="xs" weight="bold" className="text-black/35">
              {t('laila.system.offline', { count: offline })}
            </UiText>
          </div>
        </div>
      </div>

      <div className="text-right shrink-0">
        <UiText as="p" size="xs" weight="bold" className="text-black/25 uppercase tracking-wider">
          {t('laila.system.versionLabel')}
        </UiText>
        <UiText as="p" size="sm" weight="bold" className="text-black/50 mt-0.5">
          {t('laila.system.version')}
        </UiText>
        <UiText as="p" size="xs" tone="muted" className="text-black/25 mt-0.5">
          {t('laila.system.buildNote')}
        </UiText>
      </div>
    </div>
  );
}
