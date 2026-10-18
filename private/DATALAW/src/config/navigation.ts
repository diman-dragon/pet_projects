import type { TranslationKey } from '../i18n/translations';
import type { AppPermission } from '../services/auth/permissions';

export type SystemService = 'interpol' | 'laila' | 'core_db' | 'doc_gen' | 'monitoring' | 'billing';

export const SYSTEM_STATUS: Record<SystemService, boolean> = {
  core_db: true,
  laila: true,
  interpol: false,
  doc_gen: false,
  monitoring: false,
  billing: false,
};

export interface NavItem {
  id: string;
  labelKey: TranslationKey;
  icon: string;
  permission?: AppPermission;
  requiresService?: SystemService;
  isBeta?: boolean;
}

export const SIDEBAR_MENU: NavItem[] = [
  { id: 'dashboard', labelKey: 'sidebar.menu.dashboard', icon: '⌂', permission: 'dashboard:view' },
  { id: 'users', labelKey: 'sidebar.menu.users', icon: '◉', permission: 'users:manage-admins' },
  { id: 'clients', labelKey: 'sidebar.menu.clients', icon: '◆', permission: 'clients:view' },
  { id: 'knowledge', labelKey: 'sidebar.menu.knowledge', icon: '▣', permission: 'knowledge:view' },
  { id: 'knowledge_graph', labelKey: 'sidebar.menu.knowledge_graph', icon: '◎', permission: 'knowledge:view', isBeta: true },
  { id: 'laila', labelKey: 'sidebar.menu.laila', icon: '◌', permission: 'ai:use', isBeta: true },
  { id: 'systematizer', labelKey: 'sidebar.menu.systematizer', icon: '◰', permission: 'ai:use' },
  {
    id: 'doc_gen',
    labelKey: 'sidebar.menu.doc_gen',
    icon: '✎',
    requiresService: 'doc_gen',
    permission: 'ai:use',
  },
  {
    id: 'interpol',
    labelKey: 'sidebar.menu.interpol',
    icon: '◎',
    requiresService: 'interpol',
    permission: 'knowledge:view',
  },
];
