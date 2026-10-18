import type { ChangeEvent } from 'react';
import { useMemo } from 'react';
import { Field, Input, Select } from '../../../components/ui';
import { RequiredFieldsLegend } from '../../../components/ui/form/RequiredFieldsLegend';
import { useI18n } from '../../../i18n/AppLanguageProvider';
import type { TranslationKey } from '../../../i18n/translations';
import type { User } from '../types';

interface Props {
  formData: Partial<User>;
  setFormData: (data: Partial<User>) => void;
  canAssignRole?: (role: string) => boolean;
}

const ROLE_VALUES = ['ADMIN', 'LAWYER', 'ASSISTANT', 'ANALYST'] as const;

export default function UserEditForm({ formData, setFormData, canAssignRole }: Props) {
  const { t } = useI18n();
  const set = (key: keyof User, value: unknown) => setFormData({ ...formData, [key]: value });
  const handle = (key: keyof User) => (e: ChangeEvent<HTMLInputElement>) => set(key, e.target.value);

  const roles = useMemo(
    () =>
      ROLE_VALUES.filter((role) => (canAssignRole ? canAssignRole(role) : true)).map((value) => ({
        value,
        label: t(`role.${value}` as TranslationKey),
      })),
    [canAssignRole, t],
  );

  return (
    <div className="space-y-5">
      <RequiredFieldsLegend />
      <Field label={t('users.form.fullName')}>
        <Input
          autoFocus
          placeholder={t('users.form.fullNamePlaceholder')}
          value={formData.full_name ?? ''}
          onChange={handle('full_name')}
        />
      </Field>
      <div className="grid grid-cols-2 gap-4">
        <Field label={t('users.form.login')}>
          <Input
            placeholder={t('users.form.loginPlaceholder' as TranslationKey)}
            className="font-[var(--font-mono)]"
            disabled={Boolean(formData._isEdit)}
            value={formData.login ?? ''}
            onChange={handle('login')}
          />
        </Field>
        <Field label={t('users.form.role')}>
          <Select options={roles} value={formData.role ?? 'LAWYER'} onChange={(value) => set('role', value)} />
        </Field>
      </div>
      <Field label={t('users.form.password')} hint={formData._isEdit ? t('users.form.passwordHint') : undefined}>
        <Input
          type="password"
          placeholder={t('users.form.passwordPlaceholder' as TranslationKey)}
          value={formData.password ?? ''}
          onChange={handle('password')}
        />
      </Field>
    </div>
  );
}
