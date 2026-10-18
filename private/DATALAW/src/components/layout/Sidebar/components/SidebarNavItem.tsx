import React, { type ReactNode } from 'react';
import { UiText } from '../../../ui/atoms/UiText';

const ICON_BOX = 'flex size-5 shrink-0 items-center justify-center text-[var(--app-text-muted)]';

interface SidebarNavItemProps {
  id: string;
  label: string;
  icon: ReactNode;
  isActive: boolean;
  isLocked: boolean;
  isBeta?: boolean;
  onClick: () => void;
  title?: string;
}

export const SidebarNavItem: React.FC<SidebarNavItemProps> = ({
  label,
  icon,
  isActive,
  isLocked,
  isBeta,
  onClick,
  title,
}) => {
  const activeClass =
    'bg-[color-mix(in_srgb,var(--app-accent)_8%,transparent)] text-[var(--app-accent)] [&_.nav-ic]:text-[var(--app-accent)] shadow-none';
  const inactiveClass =
    'text-black/60 hover:bg-black/[0.04] hover:text-black [&_.nav-ic]:hover:text-black';
  const lockedClass = 'pointer-events-none cursor-not-allowed opacity-40 grayscale';

  return (
    <button
      type="button"
      disabled={isLocked}
      onClick={onClick}
      title={title}
      className={`group relative flex h-auto min-h-9 w-full items-start gap-2.5 rounded-lg px-2.5 py-2 text-left transition-all duration-200 ${isActive ? activeClass : inactiveClass} ${isLocked ? lockedClass : 'cursor-pointer'} active:scale-95`}
    >
      <span className={`nav-ic mt-0.5 ${ICON_BOX}`}>{icon}</span>
      <UiText as="span" size="xs" weight="medium" heading className="flex-1 whitespace-normal">
        {label}
        {isBeta && (
          <span className="ml-1.5 inline-block size-1.5 animate-pulse rounded-full bg-[color-mix(in_srgb,var(--app-accent)_55%,var(--app-text-muted))]" />
        )}
      </UiText>
      {isLocked && (
        <div className="absolute right-2 top-3">
          <div className="size-1 rounded-full bg-[color-mix(in_srgb,var(--app-text)_12%,transparent)]" />
        </div>
      )}
    </button>
  );
};
