import { Crown, Shield, Scale, ClipboardList, BarChart2 } from 'lucide-react';

export const ROLE_STYLE_MAP = {
  SUPERADMIN: {
    className: 'text-fuchsia-700 bg-fuchsia-50 border border-fuchsia-200/80',
    Icon: Crown,
  },
  ADMIN: {
    className: 'text-amber-700 bg-amber-50 border border-amber-200/80',
    Icon: Shield,
  },
  LAWYER: {
    className: 'text-sky-700 bg-sky-50 border border-sky-200/80',
    Icon: Scale,
  },
  ASSISTANT: {
    className: 'text-emerald-700 bg-emerald-50 border border-emerald-200/80',
    Icon: ClipboardList,
  },
  ANALYST: {
    className: 'text-cyan-700 bg-cyan-50 border border-cyan-200/80',
    Icon: BarChart2,
  },
} as const;
