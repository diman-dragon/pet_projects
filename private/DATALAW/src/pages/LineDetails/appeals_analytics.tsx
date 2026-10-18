// src/pages/LineDetails/appeals_analytics.tsx
import React from 'react';
import { FileText, Send, CheckCircle2, Archive } from 'lucide-react';
import type { BadgeVariant } from '../../components/ui/data/Badge';
import type { Appeal } from '../../types/entities';
import type { TranslationKey } from '../../i18n/translations';

export interface AppealStatItem {
  icon: React.ReactNode;
  labelKey: TranslationKey;
  value: number;
  variant: BadgeVariant;
}

export function getAppealStatsConfig(items: Appeal[] = []): AppealStatItem[] {
  const safe = Array.isArray(items) ? items : [];
  const by = (s: string) => safe.filter((i) => (i.status || '').toUpperCase() === s).length;

  return [
    {
      icon: <FileText size={16} />,
      labelKey: 'appeals.analytics.total',
      value: safe.length,
      variant: 'blue',
    },
    {
      icon: <Send size={16} />,
      labelKey: 'appeals.analytics.sent',
      value: by('SENT'),
      variant: 'amber',
    },
    {
      icon: <CheckCircle2 size={16} />,
      labelKey: 'appeals.analytics.closed',
      value: by('CLOSED'),
      variant: 'green',
    },
    {
      icon: <Archive size={16} />,
      labelKey: 'appeals.analytics.archived',
      value: by('ARCHIVE'),
      variant: 'gray',
    },
  ];
}
