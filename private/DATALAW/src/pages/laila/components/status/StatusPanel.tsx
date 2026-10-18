// components/status/StatusPanel.tsx

import { useMemo } from 'react';
import { useI18n } from '../../../../i18n/AppLanguageProvider';
import { getSystemModules } from '../../data';
import { SystemIdentityBlock } from './SystemIdentityBlock';
import { RoadmapNote } from './RoadmapNote';
import { ModuleGrid } from './ModuleGrid';

/**
 * Вкладка "Статус системы".
 * Собирает: блок идентификации → дорожная карта → сетка подсистем.
 */
export function StatusPanel() {
  const { t } = useI18n();
  const modules = useMemo(() => getSystemModules(t), [t]);

  const online = modules.filter((m) => m.status === 'online').length;
  const standby = modules.filter((m) => m.status === 'standby').length;
  const offline = modules.filter((m) => m.status === 'offline').length;

  return (
    <div className="p-6 h-full overflow-y-auto custom-scroll space-y-6">
      <SystemIdentityBlock online={online} standby={standby} offline={offline} />
      <RoadmapNote />
      <ModuleGrid modules={modules} />
    </div>
  );
}
