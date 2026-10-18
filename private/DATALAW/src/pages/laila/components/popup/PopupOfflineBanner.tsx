// components/popup/PopupOfflineBanner.tsx

import { WifiOff } from 'lucide-react';
import { UiText } from '../../../../components/ui/atoms/UiText';
import { useI18n } from '../../../../i18n/AppLanguageProvider';

/** Предупреждение о том, что LocalMind не запущен */
export function PopupOfflineBanner() {
  const { t } = useI18n();

  return (
    <div className="px-4 pt-3 shrink-0">
      <div className="mb-3 px-3 py-2.5 rounded-xl bg-amber-50 border border-amber-200/70 flex items-start gap-2.5">
        <WifiOff size={12} className="text-amber-500 mt-0.5 shrink-0" />
        <div>
          <UiText as="p" size="xs" weight="bold" className="text-amber-700">
            {t('laila.popup.offlineTitle')}
          </UiText>
          <UiText
            as="p"
            size="xs"
            className="text-amber-600/80 mt-0.5 leading-[var(--leading-body)]"
          >
            {t('laila.popup.offlineBodyPrefix')}{' '}
            <span className="font-medium">AI Control Center</span>
          </UiText>
        </div>
      </div>
    </div>
  );
}
