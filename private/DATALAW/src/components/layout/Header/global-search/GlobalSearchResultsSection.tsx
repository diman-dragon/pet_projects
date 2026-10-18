// src/components/layout/Header/global-search/GlobalSearchResultsSection.tsx
// ИТЕРАЦИЯ 4: prop count используется, без изменений в логике.

import type { ReactNode } from 'react';
import { UiText } from '../../../ui/atoms/UiText';

export type GlobalSearchResultsSectionProps = {
  title: string;
  children: ReactNode;
  count?: number;
};

export function GlobalSearchResultsSection({
  title,
  children,
  count,
}: GlobalSearchResultsSectionProps) {
  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <UiText
          as="div"
          size="xs"
          weight="semibold"
          tone="muted"
          className="uppercase tracking-[0.18em]"
        >
          {title}
        </UiText>
        {count !== undefined && count > 0 && (
          <span className="rounded-full bg-[color-mix(in_srgb,var(--app-accent)_12%,transparent)] px-1.5 py-px text-[10px] font-medium text-[var(--app-accent)]">
            {count}
          </span>
        )}
      </div>
      <div className="space-y-2">{children}</div>
    </div>
  );
}
