// components/status/ModuleLoadBar.tsx

import { ModuleStatus } from '../../types';

interface ModuleLoadBarProps {
  status: ModuleStatus;
  load: number; // 0–100
}

const BAR_COLORS: Record<ModuleStatus, string> = {
  online: 'bg-gradient-to-r from-violet-400 to-[#0067c0]',
  standby: 'bg-amber-300',
  offline: 'bg-black/10',
};

/** Тонкая полоса загрузки модуля */
export function ModuleLoadBar({ status, load }: ModuleLoadBarProps) {
  const width = status === 'online' ? load : 0;

  return (
    <div className="w-full h-1 rounded-full bg-black/[0.04] overflow-hidden">
      <div
        className={`h-full rounded-full transition-all duration-1000 ${BAR_COLORS[status]}`}
        style={{ width: `${width}%` }}
      />
    </div>
  );
}
