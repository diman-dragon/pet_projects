import type { ReactNode } from 'react';
import { useI18n } from '../../../i18n/AppLanguageProvider';
import { UiText } from '../atoms/UiText';

interface FieldProps {
  label: string;
  hint?: string;
  error?: string;
  required?: boolean;
  children: ReactNode;
  className?: string;
}

export function Field({
  label,
  hint,
  error,
  required = false,
  children,
  className = '',
}: FieldProps) {
  const { t } = useI18n();

  return (
    <div className={['flex w-full flex-col gap-1.5', className].join(' ')}>
      <label className="flex cursor-default select-none items-center gap-1">
        <UiText
          as="span"
          size="xs"
          weight="bold"
          tone="muted"
          className="label-caps"
        >
          {label}
        </UiText>
        {required && (
          <span className="font-black text-[var(--app-danger)]" title={t('field.requiredField')}>
            *
          </span>
        )}
      </label>

      <div className="group relative">{children}</div>

      <div className="min-h-[1rem]">
        {error ? (
          <UiText as="span" size="xs" className="text-[var(--app-danger)]">
            {error}
          </UiText>
        ) : hint ? (
          <UiText as="span" size="xs" tone="muted">
            {hint}
          </UiText>
        ) : null}
      </div>
    </div>
  );
}
