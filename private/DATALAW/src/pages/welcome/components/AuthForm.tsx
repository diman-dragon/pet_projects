import type { ChangeEvent } from 'react';
import { Field, Input } from '../../../components/ui';
import { UiText } from '../../../components/ui/atoms/UiText';
import { useI18n } from '../../../i18n/AppLanguageProvider';
import type { WelcomeLogic } from '../useWelcomeLogic';

export function AuthForm({ logic }: { logic: WelcomeLogic }) {
  const { t } = useI18n();

  return (
    <form id="auth-form" onSubmit={logic.handleLogin} className="space-y-4">
      <div className="flex h-9 items-center rounded-[var(--app-radius-md)] border border-[var(--app-border)] bg-[var(--app-surface-soft)] px-4">
        <UiText as="span" size="sm" weight="semibold" tone="secondary" className="truncate">
          {logic.selectedUser?.full_name}
        </UiText>
      </div>
      <Field label={t('auth.accessKey')} error={logic.error ? t('auth.denied') : undefined} required>
        <Input
          autoFocus
          required
          type="password"
          placeholder={t('auth.passwordPlaceholder')}
          value={logic.password}
          onChange={(e: ChangeEvent<HTMLInputElement>) => logic.setPassword(e.target.value)}
          error={logic.error ? t('auth.denied') : undefined}
        />
      </Field>
    </form>
  );
}
