import { useCallback, useState, useEffect } from 'react';
import { Folder, Hash } from 'lucide-react';
import { listen } from '@tauri-apps/api/event';
import { AISummaryBlock } from '../../../components/ui/data/AISummaryBlock';
import { useI18n } from '../../../i18n/AppLanguageProvider';
import type { TranslationKey } from '../../../i18n/translations';
import { businessApi } from '@/services/backend/businessApi';
import type { Case } from '../types';

const STATUS_STYLES: Record<string, string> = {
  active: 'bg-[var(--app-info-soft)] text-[var(--app-accent)] border-[var(--app-info-border)]',
  hand: 'bg-[var(--app-warning-soft)] text-[var(--app-warning)] border-[var(--app-warning-border)]',
  closed: 'bg-[var(--app-success-soft)] text-[var(--app-success)] border-[var(--app-success-border)]',
  archive: 'bg-[var(--app-surface-soft)] text-[var(--app-text-secondary)] border-[var(--app-border)]',
};

interface FieldRowProps {
  label: string;
  value?: string | null;
  mono?: boolean;
  emptyLabel: string;
}

function FieldRow({ label, value, mono = false, emptyLabel }: FieldRowProps) {
  return (
    <div>
      <div className="text-[9px] font-black uppercase tracking-widest text-black/30 mb-1">
        {label}
      </div>
      <div
        className={`text-[13px] text-black/80 font-medium leading-snug ${mono ? 'font-mono' : ''}`}
      >
        {value ?? <span className="text-black/20 italic">{emptyLabel}</span>}
      </div>
    </div>
  );
}

interface CaseInfoPanelProps {
  case_: Case | null;
}

export function CaseInfoPanel({ case_ }: CaseInfoPanelProps) {
  const { t } = useI18n();
  const status = (case_?.status ?? 'active').toLowerCase();
  const statusStyle = STATUS_STYLES[status] ?? STATUS_STYLES.active;
  const phaseKey = `case.phase.${status}` as TranslationKey;
  const statusTranslated = t(phaseKey);
  const statusLabel = statusTranslated === phaseKey ? case_?.status || 'active' : statusTranslated;
  const [summaryText, setSummaryText] = useState<string | null>(case_?.summary ?? null);
  const [summaryLoading, setSummaryLoading] = useState(false);

  useEffect(() => {
    if (!case_) return;
    let canceled = false;
    let unlisten: (() => void) | undefined;

    listen<{ type: string; id: number; summary: string }>(
      'summary-completed',
      (event) => {
        if (event.payload.type === 'case' && event.payload.id === case_.id) {
          setSummaryText(event.payload.summary);
          setSummaryLoading(false);
        }
      },
    ).then((fn) => {
      if (canceled) {
        fn();
      } else {
        unlisten = fn;
      }
    });

    return () => {
      canceled = true;
      unlisten?.();
    };
  }, [case_?.id]);

  const handleGenerateSummary = useCallback(async () => {
    if (!case_) return;
    setSummaryLoading(true);
    try {
      await businessApi.generateCaseSummary(case_.id);
    } catch (err) {
      console.error('Failed to start summary generation:', err);
      setSummaryLoading(false);
    }
  }, [case_?.id]);

  const empty = t('line.field.notSpecified');

  if (!case_) return null;

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-start gap-4 pb-5 border-b border-black/[0.06]">
        <div className="flex-1">
          <div className="text-[16px] font-bold text-black/90 leading-tight">{case_.title}</div>
          <div className="flex items-center gap-2 mt-1.5">
            {case_.case_type && (
              <span className="text-[11px] text-black/40">{case_.case_type}</span>
            )}
            <span
              className={`inline-flex px-2 py-0.5 rounded text-[10px] font-semibold border ${statusStyle}`}
            >
              {statusLabel}
            </span>
          </div>
        </div>
      </div>

      <AISummaryBlock
        summary={summaryText}
        label={t('case.aiSummary.label')}
        onGenerate={handleGenerateSummary}
        loading={summaryLoading}
      />

      <div className="grid grid-cols-2 gap-5">
        <div className="flex items-start gap-2">
          <Hash size={12} className="text-black/20 mt-1 shrink-0" />
          <FieldRow
            label={t('case.field.caseNumber')}
            value={case_.case_number}
            mono
            emptyLabel={empty}
          />
        </div>
        <div className="flex items-start gap-2">
          <Folder size={12} className="text-black/20 mt-1 shrink-0" />
          <FieldRow
            label={t('case.field.workFolder')}
            value={case_.folder_name}
            mono
            emptyLabel={empty}
          />
        </div>
      </div>

      {case_.case_type === 'contract' && (
        <div className="rounded-2xl border border-[var(--app-info-border)] bg-[var(--app-info-soft)] p-4 text-[length:var(--text-sm)] text-[var(--app-text)]">
          <div className="font-semibold">Структура договорного дела</div>
          <div className="mt-1 leading-relaxed text-[length:var(--text-sm)] text-[var(--app-text-secondary)]">
            Файлы хранятся в структуре: <span className="font-semibold">Год → Контрагент → Documents</span>.
          </div>
        </div>
      )}

      {case_.counterparty && (
        <div className="mt-4">
          <div className="text-[9px] font-black uppercase tracking-widest text-black/30 mb-1">
            Контрагент
          </div>
          <div className="text-[13px] text-black/80 font-medium leading-snug">
            {case_.counterparty}
          </div>
        </div>
      )}

      {case_.metadata && (
        <div>
          <div className="text-[9px] font-black uppercase tracking-widest text-black/30 mb-2">
            {t('case.section.notes')}
          </div>
          <pre className="text-[12px] text-black/60 whitespace-pre-wrap font-mono leading-relaxed bg-black/[0.02] border border-black/[0.05] rounded-lg p-4">
            {case_.metadata}
          </pre>
        </div>
      )}
    </div>
  );
}
