import type { ReactNode } from 'react';
import { SearchInput } from '../../components/ui/form/SearchInput';
import { UiText } from '../../components/ui/atoms/UiText';
import { useI18n } from '@/i18n/AppLanguageProvider';

interface RegistryToolbarProps {
  title?: string;
  subtitle?: string;
  searchPlaceholder?: string;
  searchValue: string;
  onSearchChange: (val: string) => void;
  filters?: ReactNode;
  rightContent?: ReactNode;
}

export function RegistryToolbar({
  title,
  subtitle,
  searchPlaceholder,
  searchValue,
  onSearchChange,
  filters,
  rightContent,
}: RegistryToolbarProps) {
  const { t } = useI18n();
  const resolvedPlaceholder = searchPlaceholder ?? t('common.search');

  return (
    <div className="section-divider flex flex-col gap-4 bg-[var(--app-panel-bg)]">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div className="min-w-[320px] flex-1">
          {title ? (
            <div className="mb-3">
              <UiText as="p" size="xs" weight="bold" tone="muted" className="label-caps">
                {title}
              </UiText>
              {subtitle ? (
                <UiText as="p" size="xs" tone="secondary" className="mt-1">
                  {subtitle}
                </UiText>
              ) : null}
            </div>
          ) : null}
          <div className="max-w-[400px]">
            <SearchInput value={searchValue} onChange={onSearchChange} placeholder={resolvedPlaceholder} autoFocus />
          </div>
        </div>
        <div className="flex items-center gap-3">
          {filters ? <div className="flex items-center gap-2 rounded-[var(--app-radius-md)] border border-[var(--app-border)] bg-[var(--app-surface)] p-2">{filters}</div> : null}
          {rightContent}
        </div>
      </div>
    </div>
  );
}
