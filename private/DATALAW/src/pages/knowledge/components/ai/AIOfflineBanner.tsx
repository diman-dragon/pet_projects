import { useNavigate } from 'react-router-dom';
import { useI18n } from '../../../../i18n/AppLanguageProvider';
import { UiText } from '../../../../components/ui/atoms/UiText';

interface AIOfflineBannerProps {
  timedOut?: boolean;
}

export function AIOfflineBanner({ timedOut = false }: AIOfflineBannerProps) {
  const { t } = useI18n();
  const navigate = useNavigate();

  const handleOpenAIControl = () => {
    navigate('/settings');
  };

  return (
    <UiText as="p" size="xs" tone="muted" className="italic leading-[var(--leading-body)]">
      {timedOut ? (
        t('knowledge.ai.offlineTimeout')
      ) : (
        <>
          {t('knowledge.ai.offlineBannerBefore')}{' '}
          <button
            onClick={handleOpenAIControl}
            className="rounded bg-black/[0.04] px-1 font-mono not-italic cursor-pointer hover:bg-black/[0.08] transition-colors"
            title="Go to AI Control Center"
          >
            AI Control Center
          </button>{' '}
          {t('knowledge.ai.offlineBannerAfter')}
        </>
      )}
    </UiText>
  );
}
