import type { ReactNode } from 'react';
import { UiText } from '../atoms/UiText';

interface ModalTitleProps {
  icon: ReactNode;
  title: string;
  subtitle?: string;
}

export function ModalTitle({ icon, title, subtitle }: ModalTitleProps) {
  return (
    <div className="flex items-center gap-3">
      <div className="flex size-10 items-center justify-center rounded-xl border border-[var(--app-border)] bg-[var(--app-surface-soft)] text-[var(--app-text-muted)] shadow-[inset_0_1px_0_color-mix(in_srgb,var(--app-surface-raised)_70%,transparent)]">
        {icon}
      </div>
      <div className="min-w-0">
        <UiText
          as="div"
          size="md"
          weight="semibold"
          tone="default"
          heading
          className="tracking-tight"
        >
          {title}
        </UiText>
        {subtitle && (
          <UiText as="div" size="xs" tone="secondary" className="mt-0.5">
            {subtitle}
          </UiText>
        )}
      </div>
    </div>
  );
}
