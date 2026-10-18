// src/pages/ClientDetails/components/ClientInfoPanel/KadManualImport.tsx
// [4.2B] Ручной импорт КАД из буфера обмена — кнопка + парсинг JSON

import { useState, useCallback } from 'react';
import { ClipboardPaste, CheckCircle2, XCircle, AlertTriangle } from 'lucide-react';
import { UiText, Button } from '@/components/ui';
import { useI18n } from '@/i18n/AppLanguageProvider';

interface KadManualImportProps {
  clientId: number;
  onImported: (data: KadArbitrManifest) => void;
}

// Структура, ожидаемая из буфера КАД Арбитр JSON
export interface KadArbitrManifest {
  case_count: number;
  cases: Array<{
    case_number: string;
    card_url?: string;
    court: string;
    date?: string;
    case_type?: string;
    status?: string;
    side?: string;
    amount?: number;
    description?: string;
    plaintiffs?: string[];
    respondents?: string[];
  }>;
  statistics?: {
    total: number;
    won: number;
    lost: number;
    pending: number;
  };
  last_update?: string;
  error?: string;
}

type ImportStatus = 'idle' | 'success' | 'error' | 'invalid';

function parseKadJson(raw: string): KadArbitrManifest | null {
  try {
    const parsed = JSON.parse(raw.trim());

    // Гибкая нормализация — принимаем разные форматы экспорта КАД
    if (Array.isArray(parsed)) {
      // Формат: просто массив дел
      return {
        case_count: parsed.length,
        cases: parsed,
        statistics: {
          total: parsed.length,
          won: 0,
          lost: 0,
          pending: parsed.length,
        },
      };
    }

    if (parsed && typeof parsed === 'object') {
      // Формат: объект с полем cases или items
      const cases = parsed.cases ?? parsed.items ?? parsed.data ?? [];
      return {
        case_count: parsed.case_count ?? parsed.total ?? cases.length,
        cases: Array.isArray(cases) ? cases : [],
        statistics: parsed.statistics ?? {
          total: cases.length,
          won: 0,
          lost: 0,
          pending: cases.length,
        },
        last_update: parsed.last_update ?? new Date().toISOString(),
        error: parsed.error,
      };
    }

    return null;
  } catch {
    return null;
  }
}

export function KadManualImport({ onImported }: KadManualImportProps) {
  const { t } = useI18n();
  const [status, setStatus] = useState<ImportStatus>('idle');
  const [errorMsg, setErrorMsg] = useState('');
  const [caseCount, setCaseCount] = useState(0);

  const handlePaste = useCallback(async () => {
    setStatus('idle');
    setErrorMsg('');

    let raw = '';
    try {
      raw = await navigator.clipboard.readText();
    } catch {
      setStatus('error');
      setErrorMsg('Нет доступа к буферу обмена. Разрешите доступ в настройках браузера.');
      return;
    }

    if (!raw.trim()) {
      setStatus('error');
      setErrorMsg('Буфер обмена пуст. Скопируйте JSON из КАД Арбитр и попробуйте снова.');
      return;
    }

    const parsed = parseKadJson(raw);
    if (!parsed) {
      setStatus('invalid');
      setErrorMsg(
        'Не удалось разобрать JSON. Убедитесь, что скопированы данные из КАД Арбитр в формате JSON.',
      );
      return;
    }

    setCaseCount(parsed.cases.length);
    setStatus('success');
    onImported(parsed);
  }, [onImported]);

  return (
    <div className="rounded-xl border border-dashed border-[var(--app-border)] bg-[var(--app-bg)]/40 p-5">
      <div className="mb-3 flex items-center gap-2">
        <ClipboardPaste size={16} className="text-[var(--app-accent)]" />
        <UiText size="xs" weight="black" className="uppercase tracking-widest">
          {t('dossier.kad.manualImport.title' as any) ?? 'Ручной импорт КАД'}
        </UiText>
      </div>

      <UiText size="xs" tone="muted" className="mb-4 leading-relaxed">
        {t('dossier.kad.manualImport.hint' as any) ??
          'Откройте КАД Арбитр, экспортируйте данные в JSON, скопируйте и нажмите кнопку ниже.'}
      </UiText>

      <Button
        variant="secondary"
        size="sm"
        onClick={handlePaste}
        icon={<ClipboardPaste size={14} />}
        className="mb-3"
      >
        {t('dossier.kad.manualImport.button' as any) ?? 'Загрузить из буфера'}
      </Button>

      {status === 'success' && (
        <div className="flex items-center gap-2 rounded-lg bg-[var(--app-success-soft,#ecfdf5)] px-3 py-2">
          <CheckCircle2 size={14} className="shrink-0 text-[var(--app-online)]" />
          <UiText size="xs" weight="semibold" className="text-[var(--app-online)]">
            {t('dossier.kad.manualImport.success' as any) ??
              `Импортировано ${caseCount} дел из буфера`}
          </UiText>
        </div>
      )}

      {(status === 'error' || status === 'invalid') && (
        <div className="flex items-start gap-2 rounded-lg border border-[var(--app-warning-border)] bg-[var(--app-warning-soft)] px-3 py-2">
          {status === 'invalid' ? (
            <AlertTriangle size={14} className="mt-0.5 shrink-0 text-[var(--app-warning)]" />
          ) : (
            <XCircle size={14} className="mt-0.5 shrink-0 text-[var(--app-danger)]" />
          )}
          <UiText
            size="xs"
            className={
              status === 'invalid'
                ? 'text-[var(--app-warning)]'
                : 'text-[var(--app-danger)]'
            }
          >
            {errorMsg}
          </UiText>
        </div>
      )}
    </div>
  );
}
