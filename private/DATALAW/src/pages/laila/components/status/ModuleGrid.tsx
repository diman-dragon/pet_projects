// components/status/ModuleGrid.tsx

import { useI18n } from '../../../../i18n/AppLanguageProvider';
import { LailaSystemModule } from '../../types';
import { ModuleCard } from './ModuleCard';

interface ModuleGridProps {
  modules: LailaSystemModule[];
}

/** Сетка карточек подсистем */
export function ModuleGrid({ modules }: ModuleGridProps) {
  const { t } = useI18n();

  return (
    <div>
      <p className="text-[length:var(--text-xs)] font-black text-black/25 uppercase tracking-wider mb-3">
        {t('laila.status.subsystemsTitle')}
      </p>
      <div className="grid grid-cols-2 gap-3 pb-6">
        {modules.map((m) => (
          <ModuleCard key={m.id} module={m} />
        ))}
      </div>
    </div>
  );
}
