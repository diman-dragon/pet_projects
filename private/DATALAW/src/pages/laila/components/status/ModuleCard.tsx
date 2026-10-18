// components/status/ModuleCard.tsx

import { LailaSystemModule } from '../../types';
import { ModuleStatusBadge } from './ModuleStatusBadge';
import { ModuleLoadBar } from './ModuleLoadBar';

interface ModuleCardProps {
  module: LailaSystemModule;
}

/** Карточка одного системного модуля */
export function ModuleCard({ module }: ModuleCardProps) {
  return (
    <div className="p-4 rounded-xl border border-black/[0.06] bg-white hover:border-black/[0.1] transition-all duration-150">
      <div className="flex items-start justify-between mb-3">
        <div className="pr-2">
          <p className="text-[length:var(--text-xs)] font-black text-black/70 tracking-tight">
            {module.label}
          </p>
          <p className="text-[length:var(--text-xs)] text-black/35 mt-0.5 leading-tight">
            {module.description}
          </p>
        </div>
        <ModuleStatusBadge status={module.status} />
      </div>
      <ModuleLoadBar status={module.status} load={module.load ?? 0} />
    </div>
  );
}
