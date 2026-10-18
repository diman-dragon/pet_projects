import { useCallback, useState, useEffect } from 'react';
import { Folder, Layers } from 'lucide-react';
import { listen } from '@tauri-apps/api/event';
import { AISummaryBlock } from '../../../components/ui/data/AISummaryBlock';
import { useI18n } from '../../../i18n/AppLanguageProvider';
import { businessApi } from '@/services/backend/businessApi';
import { Line } from '../../../types/entities';

function FieldRow({
  label,
  value,
  emptyLabel,
}: {
  label: string;
  value?: string | null;
  emptyLabel: string;
}) {
  return (
    <div>
      <div className="text-[9px] font-black uppercase tracking-widest text-black/30 mb-1">
        {label}
      </div>
      <div className="text-[13px] text-black/80 font-medium leading-snug">
        {value || <span className="text-black/20 italic">{emptyLabel}</span>}
      </div>
    </div>
  );
}

interface Props {
  line: Line;
}

export function LineInfoPanel({ line }: Props) {
  const { t } = useI18n();
  const [summaryText, setSummaryText] = useState<string | null>(line.summary ?? null);
  const [summaryLoading, setSummaryLoading] = useState(false);
  const empty = t('line.field.notSpecified');

  useEffect(() => {
    let canceled = false;
    let unlisten: (() => void) | undefined;

    listen<{ type: string; id: number; summary: string }>(
      'summary-completed',
      (event) => {
        if (event.payload.type === 'line' && event.payload.id === line.id) {
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
  }, [line.id]);

  const handleGenerateSummary = useCallback(async () => {
    setSummaryLoading(true);
    try {
      await businessApi.generateLineSummary(line.id);
    } catch (err) {
      console.error('Failed to start summary generation:', err);
      setSummaryLoading(false);
    }
  }, [line.id]);

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center gap-4 pb-5 border-b border-black/[0.06]">
        <div className="w-12 h-12 rounded-xl border flex items-center justify-center shrink-0 bg-indigo-50 border-indigo-100 text-indigo-500">
          <Layers size={20} />
        </div>
        <div>
          <div className="text-[16px] font-bold text-black/90 leading-tight">{line.title}</div>
          <div className="flex items-center gap-2 mt-1.5">
            <span className="text-[11px] text-black/40">{t('line.field.defenseLine')}</span>
            <span className="inline-flex px-2 py-0.5 rounded text-[10px] font-semibold border bg-indigo-50 text-indigo-600 border-indigo-200">
              #{line.position_index}
            </span>
          </div>
        </div>
      </div>

      <AISummaryBlock
        summary={summaryText}
        label={t('line.aiSummary.label')}
        onGenerate={handleGenerateSummary}
        loading={summaryLoading}
      />

      <div className="grid grid-cols-2 gap-5">
        <div className="flex items-start gap-2">
          <Folder size={12} className="text-black/20 mt-1 shrink-0" />
          <FieldRow
            label={t('line.field.workFolder')}
            value={line.folder_name}
            emptyLabel={empty}
          />
        </div>
        <FieldRow
          label={t('line.field.position')}
          value={String(line.position_index)}
          emptyLabel={empty}
        />
      </div>

      {line.description && (
        <div>
          <div className="text-[9px] font-black uppercase tracking-widest text-black/30 mb-2">
            {t('line.section.description')}
          </div>
          <pre className="text-[12px] text-black/60 whitespace-pre-wrap font-mono leading-relaxed bg-black/[0.02] border border-black/[0.05] rounded-lg p-4">
            {line.description}
          </pre>
        </div>
      )}
    </div>
  );
}
