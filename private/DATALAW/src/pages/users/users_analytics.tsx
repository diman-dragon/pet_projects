import { Briefcase, Shield, UserCheck, Users } from 'lucide-react';
import { translate } from '../../i18n/translations';
import type { AppLanguage } from '../../i18n/config';
import type { User, UserRole } from './types';

const countRole = (items: User[], role: UserRole | UserRole[]) =>
  items.filter((user) => {
    const current = (user.role as string).toUpperCase() as UserRole;
    return Array.isArray(role) ? role.includes(current) : current === role;
  }).length;

export const userAnalytics = {
  stats: (items: User[], language: AppLanguage = 'ru') => [
    {
      icon: <Users size={18} />,
      value: items.length,
      label: translate(language, 'users.title'),
      color: 'blue' as const,
    },
    {
      icon: <Shield size={18} />,
      value: countRole(items, ['SUPERADMIN', 'ADMIN']),
      label: translate(language, 'users.analytics.admins'),
      color: 'amber' as const,
    },
    {
      icon: <Briefcase size={18} />,
      value: countRole(items, 'LAWYER'),
      label: translate(language, 'users.analytics.lawyers'),
      color: 'blue' as const,
    },
    {
      icon: <UserCheck size={18} />,
      value: countRole(items, ['ASSISTANT', 'ANALYST']),
      label: translate(language, 'users.analytics.operations'),
      color: 'purple' as const,
    },
  ],

  chartData: (items: User[], language: AppLanguage = 'ru') => [
    {
      name: translate(language, 'users.analytics.chartSuperadmins'),
      value: countRole(items, 'SUPERADMIN'),
    },
    { name: translate(language, 'users.analytics.chartAdmins'), value: countRole(items, 'ADMIN') },
    {
      name: translate(language, 'users.analytics.chartLawyers'),
      value: countRole(items, 'LAWYER'),
    },
    {
      name: translate(language, 'users.analytics.chartAssistants'),
      value: countRole(items, 'ASSISTANT'),
    },
    {
      name: translate(language, 'users.analytics.chartAnalysts'),
      value: countRole(items, 'ANALYST'),
    },
  ],
};
