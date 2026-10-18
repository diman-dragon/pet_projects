import React from 'react';
import { hasPermission } from '../../../services/auth/permissions';
import type { UserProfile } from '../../../types/entities';
import { LailaWidget } from '../../../pages/laila/components/popup/LailaWidget';
import Header from '../Header/Header';
import Sidebar from '../Sidebar/Sidebar';

interface ShellProps {
  user: UserProfile;
  onOpenAudit: () => void;
  onLogout: () => void;
  activeServices: Record<string, boolean>;
  currentStatus: string;
  onNavigate: (id: string, params?: Record<string, unknown>) => void;
  children: React.ReactNode;
  activeClientId?: number | string | null;
  activeCaseId?: number | string | null;
  activeLineId?: number | string | null;
  activeAppealId?: number | string | null;
}

export default function Shell({
  user,
  children,
  onLogout,
  onOpenAudit,
  onNavigate,
  activeServices,
  currentStatus,
  activeClientId,
  activeCaseId,
  activeLineId,
  activeAppealId,
}: ShellProps) {
  return (
    <div className="flex h-screen w-screen flex-col overflow-hidden bg-[var(--app-bg)] font-sans">
      <Header
        user={user}
        onLogout={onLogout}
        onOpenAudit={onOpenAudit}
        onNavigate={onNavigate}
        activeServices={activeServices}
      />

      <div className="flex flex-1 overflow-hidden gap-3 p-3">
        <Sidebar
          user={user}
          activeServices={activeServices}
          currentStatus={currentStatus}
          onNavigate={onNavigate}
          activeClientId={activeClientId}
          activeCaseId={activeCaseId}
          activeLineId={activeLineId}
          activeAppealId={activeAppealId}
        />

        <main className="relative flex min-w-0 flex-1 flex-col overflow-hidden rounded-[12px] border border-black/[0.08] bg-white shadow-[0_4px_14px_rgba(0,0,0,0.04)]">
          <div className="custom-scroll h-full flex-1 overflow-x-hidden overflow-y-auto">
            {children}
          </div>
        </main>
      </div>

      {hasPermission(user, 'ai:use') && (
        <div className="fixed bottom-6 right-6 z-[100] transition-transform hover:scale-105 active:scale-95">
          <LailaWidget />
        </div>
      )}
    </div>
  );
}
