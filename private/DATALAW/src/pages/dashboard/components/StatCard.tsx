// src/pages/dashboard/components/StatCard.tsx

import { Briefcase, Users, Network, FileText, TrendingUp, Clock, Activity } from 'lucide-react';
import { Badge } from '../../../components/ui';

const ICON_MAP = {
  briefcase: Briefcase,
  users: Users,
  network: Network,
  'file-text': FileText,
  trending: TrendingUp,
  clock: Clock,
  activity: Activity,
};

const ACCENT_STYLES = {
  blue: {
    bg: 'bg-blue-500',
    iconBg: 'bg-blue-50',
    iconColor: 'text-blue-600',
    badge: 'blue',
  },
  green: {
    bg: 'bg-green-500',
    iconBg: 'bg-green-50',
    iconColor: 'text-green-600',
    badge: 'green',
  },
  amber: {
    bg: 'bg-amber-500',
    iconBg: 'bg-amber-50',
    iconColor: 'text-amber-600',
    badge: 'amber',
  },
  violet: {
    bg: 'bg-violet-500',
    iconBg: 'bg-violet-50',
    iconColor: 'text-violet-600',
    badge: 'violet',
  },
};

interface StatCardProps {
  title: string;
  value: string | number;
  sub: string;
  icon: keyof typeof ICON_MAP;
  accent: keyof typeof ACCENT_STYLES;
}

export function StatCard({ title, value, sub, icon, accent }: StatCardProps) {
  const Icon = ICON_MAP[icon];
  const style = ACCENT_STYLES[accent];

  return (
    <div className="bg-white rounded-xl border border-black/[0.08] overflow-hidden relative group transition-all hover:shadow-md hover:border-black/[0.12]">
      <div className={`absolute top-0 left-0 w-1 h-full ${style.bg}`} />
      <div className="p-4">
        <div className="flex items-center justify-between mb-3">
          <div className={`size-8 rounded-lg ${style.iconBg} flex items-center justify-center`}>
            <Icon size={14} className={style.iconColor} />
          </div>
          <Badge label={title} variant={style.badge as any} />
        </div>
        <div className="text-[28px] font-bold text-black/85 tracking-tight">
          {value?.toLocaleString() ?? 0}
        </div>
        <p className="text-[10px] font-medium text-black/30 mt-1 tracking-tight">{sub}</p>
      </div>
    </div>
  );
}
