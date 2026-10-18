import React, { type ReactNode } from 'react';
import { ChevronRight } from 'lucide-react';
import { IconButton } from '../../../ui/atoms/IconButton';
import { UiText } from '../../../ui/atoms/UiText';
import { ClientList } from '../../../../pages/clients/sidebar/ClientList';

const ICON_BOX = 'flex size-5 shrink-0 items-center justify-center text-[var(--app-text-muted)]';

interface SidebarClientsItemProps {
  label: string;
  icon: ReactNode;
  isActive: boolean;
  isLocked: boolean;
  isExpanded: boolean;
  onToggle: () => void;
  onNavigate: () => void;
  onSelectClient: (id: number) => void;
  onSelectCase: (clientId: number, caseId: number) => void;
  onSelectLine: (clientId: number, caseId: number, lineId: number) => void;
  onSelectAppeal: (clientId: number, caseId: number, lineId: number, appealId: number) => void;
  activeIds: {
    activeClientId: number | string | null;
    activeCaseId: number | string | null;
    activeLineId: number | string | null;
    activeAppealId: number | string | null;
  };
}

export const SidebarClientsItem: React.FC<SidebarClientsItemProps> = ({
  label,
  icon,
  isActive,
  isLocked,
  isExpanded,
  onToggle,
  onNavigate,
  onSelectClient,
  onSelectCase,
  onSelectLine,
  onSelectAppeal,
  activeIds,
}) => {
  const activeClass =
    'bg-[color-mix(in_srgb,var(--app-accent)_8%,transparent)] text-[var(--app-accent)] [&_.nav-ic]:text-[var(--app-accent)]';
  const inactiveClass =
    'text-[var(--app-text-secondary)] hover:bg-[color-mix(in_srgb,var(--app-text)_4%,transparent)] hover:text-[var(--app-text)] [&_.nav-ic]:hover:text-[var(--app-text)]';
  const lockedClass = 'pointer-events-none cursor-not-allowed opacity-40 grayscale';

  return (
    <div className="flex flex-col">
      <div className={`flex items-start ${isLocked ? lockedClass : ''}`}>
        <IconButton
          disabled={isLocked}
          onClick={onToggle}
          className="!h-9 !w-6 shrink-0 rounded-l-lg rounded-r-none hover:bg-[color-mix(in_srgb,var(--app-text)_4%,transparent)]"
        >
          <ChevronRight
            size={16}
            strokeWidth={2}
            className={`transition-transform duration-200 ${isExpanded ? 'rotate-90' : ''}`}
          />
        </IconButton>
        <button
          type="button"
          disabled={isLocked}
          onClick={onNavigate}
          className={`group relative flex h-auto min-h-9 flex-1 items-start gap-2 rounded-r-lg py-2 pr-2.5 text-left transition-all duration-150 ${isActive ? activeClass : inactiveClass}`}
        >
          <span className={`nav-ic mt-0.5 ${ICON_BOX}`}>{icon}</span>
          <UiText as="span" size="xs" weight="medium" heading className="flex-1 whitespace-normal">
            {label}
          </UiText>
        </button>
      </div>

      {!isLocked && isExpanded && (
        <div className="mb-1 ml-5 mt-1.5 border-l border-[var(--app-border)] pl-2">
          <ClientList
            {...activeIds}
            onSelectClient={onSelectClient}
            onSelectCase={onSelectCase}
            onSelectLine={onSelectLine}
            onSelectAppeal={onSelectAppeal}
          />
        </div>
      )}
    </div>
  );
};
