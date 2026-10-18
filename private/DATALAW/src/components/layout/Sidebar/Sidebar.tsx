import { useEffect, useState, type ReactNode } from 'react';
import {
  LayoutDashboard,
  Users,
  FolderKanban,
  BookOpen,
  Sparkles,
  FileText,
  Globe,
  Brain,
  Layers,
  Network,
} from 'lucide-react';
import { SIDEBAR_MENU, type NavItem } from '../../../config/navigation';
import { useI18n } from '../../../i18n/AppLanguageProvider';
import type { TranslationKey } from '../../../i18n/translations';
import { hasPermission } from '../../../services/auth/permissions';
import type { UserProfile } from '../../../types/entities';
import { SidebarFooter } from './SidebarFooter';
import { SidebarNavItem } from './components/SidebarNavItem';
import { SidebarClientsItem } from './components/SidebarClientsItem';

const NAV_ICON_SZ = 16;
const NAV_ICONS: Record<string, ReactNode> = {
  dashboard: <LayoutDashboard size={NAV_ICON_SZ} strokeWidth={2} />,
  users: <Users size={NAV_ICON_SZ} strokeWidth={2} />,
  clients: <FolderKanban size={NAV_ICON_SZ} strokeWidth={2} />,
  knowledge: <BookOpen size={NAV_ICON_SZ} strokeWidth={2} />,
  'ai-control': <Brain size={NAV_ICON_SZ} strokeWidth={2} />,
  laila: <Sparkles size={NAV_ICON_SZ} strokeWidth={2} />,
  systematizer: <Layers size={NAV_ICON_SZ} strokeWidth={2} />,
  knowledge_graph: <Network size={NAV_ICON_SZ} strokeWidth={2} />,
  doc_gen: <FileText size={NAV_ICON_SZ} strokeWidth={2} />,
  interpol: <Globe size={NAV_ICON_SZ} strokeWidth={2} />,
};

interface SidebarProps {
  user: UserProfile;
  activeServices: Record<string, boolean>;
  currentStatus: string;
  onNavigate: (route: string, params?: Record<string, unknown>) => void;
  activeClientId?: number | string | null;
  activeCaseId?: number | string | null;
  activeLineId?: number | string | null;
  activeAppealId?: number | string | null;
}

export default function Sidebar({
  user,
  activeServices,
  currentStatus,
  onNavigate,
  activeClientId = null,
  activeCaseId = null,
  activeLineId = null,
  activeAppealId = null,
}: SidebarProps) {
  const { t } = useI18n();
  const [isClientsExpanded, setIsClientsExpanded] = useState(false);

  useEffect(() => {
    if (
      currentStatus === 'clients' ||
      activeClientId ||
      activeCaseId ||
      activeLineId ||
      activeAppealId
    ) {
      setIsClientsExpanded(true);
    }
  }, [currentStatus, activeClientId, activeCaseId, activeLineId, activeAppealId]);

  return (
    <aside className="flex h-full w-[clamp(14rem,18vw,16rem)] min-w-[14rem] shrink-0 select-none flex-col rounded-[12px] border border-black/[0.08] mica-effect shadow-[0_4px_12px_rgba(0,0,0,0.03)]">
      <div className="h-4 shrink-0" />
      <nav className="custom-scroll flex-1 space-y-1 overflow-y-auto px-3">
        {SIDEBAR_MENU.map((item: NavItem) => {
          if (item.permission && !hasPermission(user, item.permission)) return null;
          const isLocked =
            Boolean(item.requiresService && !activeServices[item.requiresService]) ||
            (item.id in activeServices ? !activeServices[item.id] : false);
          const label = t(item.labelKey as TranslationKey);
          const icon = NAV_ICONS[item.id] ?? (
            <span className="text-[length:var(--text-md)] leading-none">{item.icon}</span>
          );

          if (item.id === 'clients') {
            return (
              <SidebarClientsItem
                key={item.id}
                label={label}
                icon={icon}
                isActive={currentStatus === 'clients'}
                isLocked={isLocked}
                isExpanded={isClientsExpanded}
                onToggle={() => setIsClientsExpanded(!isClientsExpanded)}
                onNavigate={() => onNavigate('clients')}
                onSelectClient={(id) => onNavigate('clients', { clientId: id })}
                onSelectCase={(clientId, caseId) => onNavigate('clients', { clientId, caseId })}
                onSelectLine={(clientId, caseId, lineId) =>
                  onNavigate('clients', { clientId, caseId, lineId })
                }
                onSelectAppeal={(clientId, caseId, lineId, appealId) =>
                  onNavigate('clients', { clientId, caseId, lineId, appealId })
                }
                activeIds={{ activeClientId, activeCaseId, activeLineId, activeAppealId }}
              />
            );
          }

          return (
            <SidebarNavItem
              key={item.id}
              id={item.id}
              label={label}
              icon={icon}
              isActive={currentStatus === item.id}
              isLocked={isLocked}
              isBeta={item.isBeta}
              onClick={() => onNavigate(item.id)}
              title={isLocked ? t('common.locked' as any) : label}
            />
          );
        })}
      </nav>
      <SidebarFooter onNavigate={onNavigate} />
    </aside>
  );
}
