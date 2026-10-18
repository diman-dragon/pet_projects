import type { ReactNode } from 'react';
import { ArrowLeft } from 'lucide-react';
import { Button } from '../form/Button';
import { UiText } from '../atoms/UiText';

interface PageHeaderProps {
  title: ReactNode;
  subtitle?: ReactNode;
  icon?: ReactNode;
  clientName?: string;
  onBack?: () => void;
  actions?: ReactNode;
  rightContent?: ReactNode;
}

export function PageHeader({
  title,
  subtitle,
  icon,
  clientName,
  onBack,
  actions,
  rightContent,
}: PageHeaderProps) {
  const resolvedActions = actions ?? rightContent;
  return (
    <div className="flex flex-col gap-4 border-b border-[var(--app-border)] bg-[var(--app-surface)] px-[var(--app-spacing-6,1.5rem)] py-5">
      {clientName ? (
        <UiText as="div" size="xs" weight="bold" tone="muted" className="label-caps">
          {clientName}
        </UiText>
      ) : null}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div className="flex min-w-0 items-center gap-3">
          {onBack ? (
            <Button variant="ghost" size="sm" icon={<ArrowLeft size={16} />} onClick={onBack} />
          ) : null}
          {icon ? (
            <div className="flex h-10 w-10 items-center justify-center rounded-[var(--app-radius-md)] border border-[var(--app-border)] bg-[var(--app-surface-soft)] text-[var(--app-accent)]">
              {icon}
            </div>
          ) : null}
          <div className="min-w-0">
            <UiText as="h1" size="lg" weight="semibold" heading>
              {title}
            </UiText>
            {subtitle ? (
              <UiText as="div" size="sm" tone="secondary" className="mt-1">
                {subtitle}
              </UiText>
            ) : null}
          </div>
        </div>
        {resolvedActions ? <div className="flex items-center gap-2">{resolvedActions}</div> : null}
      </div>
    </div>
  );
}
