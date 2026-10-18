import type { ChangeEvent } from 'react';
import { Field, Input } from '../../../components/ui';
import { useI18n } from '../../../i18n/AppLanguageProvider';
import type { WelcomeLogic } from '../useWelcomeLogic';

export function GenesisForm({ logic }: { logic: WelcomeLogic }) {
  const { t } = useI18n();

  return (
    <form id="gen-form" onSubmit={logic.handleCreateAdmin} className="space-y-4">
      <Field label={t('genesis.login')} required>
        <Input
          required
          placeholder={t('genesis.loginPlaceholder')}
          value={logic.adminForm.login}
          onChange={(e: ChangeEvent<HTMLInputElement>) =>
            logic.setAdminForm({ ...logic.adminForm, login: e.target.value })
          }
        />
      </Field>
      <Field label={t('genesis.fullName')} required>
        <Input
          required
          placeholder={t('genesis.fullNamePlaceholder')}
          value={logic.adminForm.full_name}
          onChange={(e: ChangeEvent<HTMLInputElement>) =>
            logic.setAdminForm({ ...logic.adminForm, full_name: e.target.value })
          }
        />
      </Field>
      <Field label={t('genesis.password')} required>
        <Input
          required
          type="password"
          placeholder={t('genesis.passwordPlaceholder')}
          value={logic.adminForm.password}
          onChange={(e: ChangeEvent<HTMLInputElement>) =>
            logic.setAdminForm({ ...logic.adminForm, password: e.target.value })
          }
        />
      </Field>
    </form>
  );
}
