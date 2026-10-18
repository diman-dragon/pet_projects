// data/index.ts

import type { TranslationKey } from '../../../i18n/translations';
import type { LailaSystemModule } from '../types';

const SYSTEM_MODULE_ROWS: readonly {
  id: string;
  status: LailaSystemModule['status'];
  load: number;
}[] = [
  { id: 'kb', status: 'standby', load: 0 },
  { id: 'nlp', status: 'standby', load: 0 },
  { id: 'rag', status: 'standby', load: 0 },
  { id: 'doc', status: 'offline', load: 0 },
  { id: 'voice', status: 'offline', load: 0 },
  { id: 'integration', status: 'offline', load: 0 },
];

export function getSystemModules(t: (key: TranslationKey) => string): LailaSystemModule[] {
  return SYSTEM_MODULE_ROWS.map((row) => ({
    id: row.id,
    status: row.status,
    load: row.load,
    label: t(`laila.module.${row.id}.label` as TranslationKey),
    description: t(`laila.module.${row.id}.desc` as TranslationKey),
  }));
}
