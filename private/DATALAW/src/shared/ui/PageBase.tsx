import type { ReactNode } from 'react';
import { PageHeader } from '../../components/ui/layout/PageHeader';
import { SegmentedControl } from '../../components/ui/layout/SegmentedControl';
import { Window } from '../../components/ui/layout/Window';
import { PageLoader } from '../../components/ui/feedback/PageLoader';

interface PageBaseProps {
  title?: ReactNode;
  subtitle?: ReactNode;
  icon?: ReactNode;
  clientName?: string;
  onBack?: () => void;
  activeTab?: string;
  onTabChange?: (id: string) => void;
  tabs?: { id: string; label: string; icon?: ReactNode }[];
  action?: ReactNode;
  actions?: ReactNode;
  loading?: boolean;
  children: ReactNode;
  extra?: ReactNode;
  noPadding?: boolean;
}

export function PageBase({
  title,
  subtitle,
  icon,
  clientName,
  onBack,
  activeTab,
  onTabChange,
  tabs,
  action,
  actions,
  loading,
  children,
  extra,
  noPadding = false,
}: PageBaseProps) {
  const headerActions = (
    <div className="flex flex-wrap items-center gap-3">
      {tabs && activeTab && onTabChange ? (
        <SegmentedControl active={activeTab} onChange={onTabChange} segments={tabs} />
      ) : null}
      {actions ?? action}
    </div>
  );

  return (
    <div className="flex h-full flex-col overflow-hidden bg-[var(--app-bg)]">
      {title ? (
        <PageHeader
          title={title}
          subtitle={subtitle}
          icon={icon}
          clientName={clientName}
          onBack={onBack}
          actions={headerActions}
        />
      ) : null}
      <Window noPadding={noPadding}>
        {loading ? (
          <PageLoader />
        ) : (
          <div className={noPadding ? 'h-full' : 'mx-auto h-full w-full max-w-6xl px-[var(--app-spacing-6,1.5rem)] py-[var(--app-spacing-6,1.5rem)]'}>
            {children}
          </div>
        )}
      </Window>
      {extra}
    </div>
  );
}
