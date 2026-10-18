// src/pages/AppealDetails/components/InfoCards.tsx

import { useI18n } from '../../../i18n/AppLanguageProvider';
import { appLanguageToBcp47 } from '../../../i18n/config';
import type { TranslationKey } from '../../../i18n/translations';
import { formatDate, STATUS_CONFIG } from '../config';
import type { Appeal } from '../../../types/entities';

interface InfoCardsProps {
  appeal: Appeal | null;
}

function Row({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="text-[length:var(--text-xs)] font-black uppercase tracking-widest text-black/25">
        {label}
      </span>
      <span className="text-[length:var(--text-xs)] font-medium text-black/75 leading-tight">
        {value || '—'}
      </span>
    </div>
  );
}

export function InfoCards({ appeal }: InfoCardsProps) {
  const { t, language } = useI18n();
  const localeTag = appLanguageToBcp47(language);
  if (!appeal) return null;
  const statusCfg = STATUS_CONFIG[appeal.status ?? ''];
  const statusKey = `status.${(appeal.status ?? '').toLowerCase()}` as TranslationKey;
  const statusTranslated = t(statusKey);
  const statusLabel = statusTranslated === statusKey ? appeal.status : statusTranslated;

  return (
    <div className="space-y-3">
      <Row
        label={t('appeal.field.agency')}
        value={
          <span className="font-mono text-[length:var(--text-xs)] font-bold text-black/60">
            {appeal.agency_code}
          </span>
        }
      />
      <Row
        label={t('appeal.field.status')}
        value={
          statusCfg ? (
            <span
              className={`inline-flex px-2 py-0.5 rounded-md text-[length:var(--text-xs)] font-semibold ${statusCfg.color}`}
            >
              {statusLabel}
            </span>
          ) : (
            appeal.status
          )
        }
      />
      {appeal.appeal_date && (
        <Row
          label={t('appeal.field.filingDate')}
          value={formatDate(appeal.appeal_date, localeTag)}
        />
      )}
      {appeal.recipient && <Row label={t('appeal.field.recipient')} value={appeal.recipient} />}
      {appeal.outgoing_number && (
        <Row
          label={t('appeal.field.outgoing')}
          value={
            <span className="font-mono text-[length:var(--text-xs)]">{appeal.outgoing_number}</span>
          }
        />
      )}
    </div>
  );
}
