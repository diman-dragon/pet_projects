// src/pages/ClientDetails/components/ClientInfoPanel/KadArbitrDetails.tsx
// FIX V3: разделена ошибка endpoint (404/503) от "дел не найдено" (пустой результат).
// Ранее любая строка "404" в тексте ошибки помечалась как "Дела не найдены" — что вводило в заблуждение.

import { useState, type ReactNode } from 'react';
import { openPath } from '@tauri-apps/plugin-opener';
import {
  AlertCircle,
  ChevronRight,
  ClipboardPaste,
  ExternalLink,
  Globe,
  MapPin,
  RefreshCw,
  Scale,
  ShieldAlert,
  TrendingDown,
  TrendingUp,
  WifiOff,
} from 'lucide-react';
import { Badge, Button, UiText } from '@/components/ui';
import { useI18n } from '@/i18n/AppLanguageProvider';
import { KadManualImport, type KadArbitrManifest } from './KadManualImport';

interface KadArbitrCase {
  case_number: string;
  card_url?: string;
  court: string;
  date?: string;
  case_type?: string;
  status?: string;
  plaintiffs?: string[];
  respondents?: string[];
}

interface KadArbitrProvider {
  case_count: number;
  cases: KadArbitrCase[];
  error?: string | { key?: string; message?: string; securityText?: string };
  statistics?: { total: number; won: number; lost: number; pending: number };
  last_update?: string;
}

// FIX V3: улучшена классификация ошибок
// Разделяем:
//   isCloudflare  — Cloudflare 403 (JS-защита)
//   isEndpointError — техническая ошибка нашего кода/сети (500, таймаут, недоступность)
//   isEmptyResult — успешный запрос вернул 0 дел (реальное отсутствие дел)
function resolveError(error: KadArbitrProvider['error']) {
  const normalized =
    typeof error === 'string' ? { message: error } : error || { message: '' };
  const combined = `${normalized.message ?? ''} ${normalized.securityText ?? ''}`.toLowerCase();

  const isCloudflare =
    combined.includes('cloudflare') ||
    combined.includes('securitytext') ||
    normalized.key === 'errors.kad.cloudflare';

  // FIX V3: 404 это техническая ошибка нашего endpoint (был неверный URL),
  // а не "дела не найдены" — разделяем эти случаи явно
  const isEndpointError =
    combined.includes('404') ||
    combined.includes('not found') ||
    combined.includes('503') ||
    combined.includes('502') ||
    combined.includes('сетевая') ||
    combined.includes('недоступн');

  const isTimeout = combined.includes('timeout') || combined.includes('превышено время');

  return {
    ...normalized,
    isCloudflare,
    isEndpointError,
    isTimeout,
  };
}

export function KadArbitrDetails({
  data,
  clientId,
  inn,
  onRefresh,
}: {
  data: KadArbitrProvider | null;
  clientId?: number;
  inn?: string;
  onRefresh?: () => void;
}) {
  const { t } = useI18n();
  const [manualData, setManualData] = useState<KadArbitrManifest | null>(null);
  const [showImport, setShowImport] = useState(false);
  const effectiveData = manualData ?? data;
  const formatDate = (date?: string | null) =>
    date ? new Date(date).toLocaleDateString('ru-RU') : '—';

  if (!effectiveData) {
    return (
      <KadState
        icon={<Globe size={32} />}
        title={t('dossier.empty.noData' as any)}
        text={t('dossier.empty.hint' as any)}
        showImport={showImport}
        setShowImport={setShowImport}
        clientId={clientId}
        onImported={setManualData}
        onRefresh={onRefresh}
      />
    );
  }

  if (effectiveData.error && !manualData) {
    const error = resolveError(effectiveData.error);

    // FIX V3: правильные заголовки и тексты для каждого типа ошибки
    let title: string;
    let text: string;
    let icon: ReactNode;

    if (error.isCloudflare) {
      title = 'КАД Арбитр временно недоступен (защита Cloudflare)';
      text = error.securityText || 'Откройте поиск в браузере или загрузите JSON вручную.';
      icon = <ShieldAlert size={32} />;
    } else if (error.isTimeout) {
      title = 'Превышено время ожидания ответа';
      text = 'КАД Арбитр не ответил вовремя. Попробуйте повторить позже.';
      icon = <WifiOff size={32} />;
    } else if (error.isEndpointError) {
      // FIX V3: техническая ошибка запроса — НЕ "дела не найдены"
      title = 'Ошибка при запросе к КАД Арбитр';
      text = error.message || 'Не удалось выполнить поиск. Проверьте подключение и повторите.';
      icon = <AlertCircle size={32} />;
    } else {
      title = 'Ошибка синхронизации';
      text = error.message || 'Не удалось получить данные КАД Арбитр.';
      icon = <AlertCircle size={32} />;
    }

    return (
      <KadState
        icon={icon}
        title={title}
        text={text}
        showImport={showImport}
        setShowImport={setShowImport}
        clientId={clientId}
        onImported={setManualData}
        onRefresh={onRefresh}
        extraActions={
          error.isCloudflare ? (
            <button
              type="button"
              onClick={() =>
                openPath(`https://kad.arbitr.ru/?inn=${encodeURIComponent(inn || '')}`)
              }
              className="flex items-center gap-2 rounded-lg bg-[var(--app-accent)] px-5 py-2 text-xs font-bold text-white transition-all hover:brightness-110"
            >
              <Globe size={14} />
              Открыть в браузере
            </button>
          ) : null
        }
      />
    );
  }

  // FIX V3: пустой результат — отдельный случай (реально нет дел, а не ошибка)
  if (!effectiveData.error && effectiveData.case_count === 0 && effectiveData.cases.length === 0 && !manualData) {
    return (
      <KadState
        icon={<Scale size={32} />}
        title="Арбитражных дел не найдено"
        text="По данному ИНН в КАД Арбитр дела не обнаружены. Возможно, организация не участвовала в арбитражных разбирательствах."
        showImport={showImport}
        setShowImport={setShowImport}
        clientId={clientId}
        onImported={setManualData}
        onRefresh={onRefresh}
      />
    );
  }

  return (
    <div className="space-y-8">
      <div className="flex items-center justify-between">
        <UiText size="xs" tone="muted">
          {effectiveData.last_update
            ? `Обновлено: ${formatDate(effectiveData.last_update)}`
            : manualData
            ? 'Данные из буфера'
            : ''}
        </UiText>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setShowImport((v) => !v)}
          icon={<ClipboardPaste size={13} />}
        >
          Загрузить вручную
        </Button>
      </div>

      {showImport && clientId != null && (
        <KadManualImport
          clientId={clientId}
          onImported={(payload) => {
            setManualData(payload);
            setShowImport(false);
          }}
        />
      )}

      {effectiveData.statistics && (
        <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
          <StatCard label="Всего"    value={effectiveData.statistics.total}   icon={<Scale size={16} />} />
          <StatCard label="Выиграно" value={effectiveData.statistics.won}     icon={<TrendingUp size={16} />}   color="green" />
          <StatCard label="Проиграно" value={effectiveData.statistics.lost}   icon={<TrendingDown size={16} />} color="red" />
          <StatCard label="В работе" value={effectiveData.statistics.pending} icon={<ChevronRight size={16} />} color="blue" />
        </div>
      )}

      <div className="space-y-4">
        <div className="flex items-center justify-between px-2">
          <UiText size="xs" weight="black" className="uppercase tracking-widest">
            {t('dossier.arbitration.listTitle' as any)}
          </UiText>
          <Badge label={`${effectiveData.case_count}`} variant="gray" />
        </div>
        <div className="custom-scroll max-h-[500px] space-y-3 overflow-y-auto pr-2">
          {effectiveData.cases.map((item, index) => (
            <div
              key={`${item.case_number}-${index}`}
              className="rounded-xl border border-[var(--app-border)] bg-[var(--app-surface)] p-5"
            >
              <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
                <div className="min-w-0 flex-1 space-y-3">
                  <div className="flex flex-wrap items-center gap-2">
                    <UiText weight="black" className="font-mono text-sm text-[var(--app-accent)]">
                      {item.case_number}
                    </UiText>
                    {item.status && <Badge label={item.status} variant="gray" />}
                  </div>
                  <div className="flex items-center gap-4">
                    <div className="flex min-w-0 items-center gap-1.5">
                      <MapPin size={12} className="shrink-0 text-[var(--app-text-muted)]" />
                      <UiText size="xs" tone="muted" className="truncate">
                        {item.court}
                      </UiText>
                    </div>
                    <UiText size="xs" weight="bold" className="font-mono text-[var(--app-text-muted)]">
                      {formatDate(item.date)}
                    </UiText>
                  </div>
                  <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                    <MiniPart title="Истцы"     value={item.plaintiffs?.join(', ')  || '—'} />
                    <MiniPart title="Ответчики" value={item.respondents?.join(', ') || '—'} />
                  </div>
                </div>
                <a
                  href={item.card_url || `https://kad.arbitr.ru/Card/${item.case_number}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex shrink-0 items-center justify-center rounded-lg bg-[var(--app-bg)] p-2 text-[var(--app-text-muted)] transition-all hover:bg-[var(--app-accent)] hover:text-white"
                >
                  <ExternalLink size={16} />
                </a>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function KadState({
  icon,
  title,
  text,
  showImport,
  setShowImport,
  clientId,
  onImported,
  onRefresh,
  extraActions,
}: {
  icon: ReactNode;
  title: string;
  text: string;
  showImport: boolean;
  setShowImport: (value: boolean | ((prev: boolean) => boolean)) => void;
  clientId?: number;
  onImported: (payload: KadArbitrManifest) => void;
  onRefresh?: () => void;
  extraActions?: ReactNode;
}) {
  return (
    <div className="space-y-6">
      <div className="flex flex-col items-center rounded-2xl border border-[var(--app-warning)]/20 bg-[var(--app-warning)]/8 p-8 text-center">
        <div className="mb-4 rounded-xl bg-[var(--app-surface)] p-3 text-[var(--app-warning)] shadow-sm">
          {icon}
        </div>
        <UiText size="md" weight="bold" className="mb-2">{title}</UiText>
        <UiText size="xs" tone="muted" className="mb-6 max-w-md leading-relaxed">{text}</UiText>
        <div className="flex flex-wrap items-center justify-center gap-3">
          {extraActions}
          {onRefresh && (
            <Button variant="secondary" size="sm" onClick={onRefresh} icon={<RefreshCw size={14} />}>
              Повторить
            </Button>
          )}
          {/* FIX V3: кнопка ручного импорта — primary variant как основной workaround */}
          <Button
            variant="primary"
            size="sm"
            onClick={() => setShowImport((v) => !v)}
            icon={<ClipboardPaste size={14} />}
          >
            Загрузить вручную
          </Button>
        </div>
      </div>
      {showImport && clientId != null && (
        <KadManualImport
          clientId={clientId}
          onImported={(payload) => {
            onImported(payload);
            setShowImport(false);
          }}
        />
      )}
    </div>
  );
}

function MiniPart({ title, value }: { title: string; value: string }) {
  return (
    <div className="rounded-lg border border-[var(--app-border)] bg-[var(--app-bg)]/40 p-3">
      <UiText size="xs" weight="black" tone="muted" className="mb-1 block uppercase tracking-widest text-[8px]">
        {title}
      </UiText>
      <UiText size="xs" weight="semibold" className="block truncate leading-snug">
        {value}
      </UiText>
    </div>
  );
}

function StatCard({
  label,
  value,
  icon,
  color,
}: {
  label: string;
  value: number;
  icon: ReactNode;
  color?: 'green' | 'red' | 'blue';
}) {
  const colorClass =
    color === 'green'
      ? 'text-[var(--app-success)]'
      : color === 'red'
      ? 'text-[var(--app-danger)]'
      : color === 'blue'
      ? 'text-[var(--app-accent)]'
      : 'text-[var(--app-text-muted)]';
  return (
    <div className="rounded-xl border border-[var(--app-border)] bg-[var(--app-surface)] p-4 shadow-sm">
      <div className={`mb-2 ${colorClass}`}>{icon}</div>
      <UiText size="xs" weight="black" tone="muted" className="mb-1 block uppercase tracking-widest">
        {label}
      </UiText>
      <UiText size="md" weight="black">{value}</UiText>
    </div>
  );
}
