import { DorlaneLogo } from '../../common/DorlaneLogo';
import { useI18n } from '../../../i18n/AppLanguageProvider';
import { UiText } from '../../ui/atoms/UiText';
export function HeaderLogo() {
  const { t } = useI18n();

  return (
    <div className="flex min-w-[220px] items-center gap-3">
      <DorlaneLogo size="sm" showText={false} />

      <div className="flex flex-col border-l border-[var(--app-border)] pl-3 leading-tight">
        <UiText as="h1" size="sm" weight="bold" tone="default" className="uppercase tracking-tight">
          {t('header.brandTitle')}
        </UiText>
        <UiText
          as="span"
          size="xs"
          tone="muted"
          weight="bold"
          className="mt-0.5 uppercase tracking-[0.12em] leading-[var(--leading-heading)]"
        >
          {t('header.brandTagline')}
        </UiText>
      </div>
    </div>
  );
}
