// components/status/RoadmapNote.tsx

import { useI18n } from '../../../../i18n/AppLanguageProvider';

/** Блок с описанием дорожной карты развития Laila */
export function RoadmapNote() {
  const { t } = useI18n();

  return (
    <div className="p-4 rounded-xl border border-dashed border-[var(--app-accent)]/20 bg-[var(--app-accent)]/[0.02]">
      <p className="text-[length:var(--text-xs)] font-black text-[var(--app-accent)]/60 uppercase tracking-wider mb-1.5">
        {t('laila.roadmap.title')}
      </p>
      <p className="text-[length:var(--text-xs)] text-black/45 leading-[var(--leading-body)]">
        {t('laila.roadmap.body')}
      </p>
    </div>
  );
}
