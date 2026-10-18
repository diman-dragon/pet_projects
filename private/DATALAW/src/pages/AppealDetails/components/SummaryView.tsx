// src/pages/AppealDetails/components/SummaryView.tsx

import { Sparkles } from 'lucide-react';
import { useI18n } from '../../../i18n/AppLanguageProvider';
import type { AppealSummary } from '../types';
import { UiText } from '../../../components/ui/atoms/UiText';

interface SummaryViewProps {
  summary: AppealSummary | null;
  appealSummary?: string | null;
}

function EmptyState() {
  const { t } = useI18n();

  return (
    <div className="flex h-full flex-col items-center justify-center gap-3 py-16 text-center">
      <div className="flex h-10 w-10 items-center justify-center rounded-full bg-[color-mix(in_srgb,var(--app-text)_5%,transparent)]">
        <Sparkles size={18} strokeWidth={2} className="text-[var(--app-text-muted)]" />
      </div>
      <UiText
        as="p"
        size="sm"
        tone="secondary"
        className="font-medium leading-[var(--leading-body)]"
      >
        {t('appealSummary.emptyTitle')}
      </UiText>
      <UiText as="p" size="xs" tone="muted" className="max-w-[240px] leading-[var(--leading-body)]">
        {t('appealSummary.emptyBody')}
      </UiText>
    </div>
  );
}

function SummaryBlock({ title, text, date }: { title: string; text: string; date?: string }) {
  const { language } = useI18n();
  const locale = language === 'en' ? 'en-US' : language === 'sr' ? 'sr-Latn-RS' : 'ru-RU';

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2 border-b border-[var(--app-border)] pb-3">
        <div className="flex h-6 w-6 shrink-0 items-center justify-center rounded-md bg-[var(--app-text)]">
          <Sparkles size={12} strokeWidth={2} className="text-white" />
        </div>
        <span className="text-[length:var(--text-xs)] font-black uppercase leading-[var(--leading-heading)] tracking-widest text-[var(--app-text-muted)]">
          {title}
        </span>
        {date && (
          <span className="ml-auto font-mono text-[length:var(--text-xs)] leading-[var(--leading-body)] text-[var(--app-text-muted)]">
            {new Date(date).toLocaleDateString(locale)}
          </span>
        )}
      </div>
      <p className="text-[length:var(--text-sm)] font-medium leading-[var(--leading-body)] text-[var(--app-text)]">
        {text}
      </p>
    </div>
  );
}

export function SummaryView({ summary, appealSummary }: SummaryViewProps) {
  const { t } = useI18n();
  const hasNew = !!appealSummary;
  const hasLegacy = !!summary;

  if (!hasNew && !hasLegacy) return <EmptyState />;

  return (
    <div className="mx-auto max-w-2xl space-y-8">
      {hasNew && <SummaryBlock title={t('appealSummary.blockAppeal')} text={appealSummary!} />}

      {hasLegacy && summary!.text && (
        <div className="space-y-5">
          {!hasNew && (
            <SummaryBlock
              title={t('appealSummary.blockLegacy')}
              text={summary!.text}
              date={summary!.updated_at}
            />
          )}

          {summary!.key_points && summary!.key_points.length > 0 && (
            <div className="space-y-2 pt-2">
              <div className="mb-3 text-[length:var(--text-xs)] font-black uppercase leading-[var(--leading-heading)] tracking-widest text-[var(--app-text-muted)]">
                {t('appealSummary.keyPoints')}
              </div>
              {summary!.key_points.map((point, i) => (
                <div key={i} className="flex items-start gap-3">
                  <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-[color-mix(in_srgb,var(--app-text)_5%,transparent)] text-[length:var(--text-xs)] font-bold leading-[var(--leading-heading)] text-[var(--app-text-muted)]">
                    {i + 1}
                  </span>
                  <p className="text-[length:var(--text-sm)] leading-[var(--leading-body)] text-[var(--app-text-secondary)]">
                    {point}
                  </p>
                </div>
              ))}
            </div>
          )}

          {summary!.generated_by && (
            <p className="pt-2 font-mono text-[length:var(--text-xs)] leading-[var(--leading-body)] text-[var(--app-text-muted)]">
              {summary!.generated_by}
            </p>
          )}
        </div>
      )}
    </div>
  );
}
