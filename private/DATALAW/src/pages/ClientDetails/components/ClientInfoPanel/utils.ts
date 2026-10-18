import type { ClientDossierProvider, AppErrorLike } from '@/services/backend/businessApi';

export function formatProviderStatus(provider: ClientDossierProvider): string {
  switch (provider.status) {
    case 'ready':
      return 'успешно';
    case 'configured':
      return 'готов к запросу';
    case 'degraded':
      return 'ошибка / частично';
    default:
      return 'следующий этап';
  }
}

export function providerTone(provider: ClientDossierProvider): string {
  switch (provider.status) {
    case 'ready':
      return 'border-emerald-200 bg-emerald-50 text-emerald-700';
    case 'degraded':
      return 'border-red-200 bg-red-50 text-red-700';
    default:
      return 'border-amber-200 bg-amber-50 text-amber-700';
  }
}

export function formatAppError(error: unknown): string {
  if (typeof error === 'object' && error !== null) {
    const typed = error as AppErrorLike;
    return typed.message || typed.key || String(error);
  }
  return String(error);
}

export const formatDate = (dateStr?: string): string => {
  if (!dateStr) return 'Не указано';
  try {
    let d: Date | null = null;

    // .NET JSON date: "/Date(1234567890000)/" or "/Date(ms+tz)/"
    const dotnetMatch = dateStr.match(/\/Date\((-?\d+)([+-]\d+)?\)\//);
    if (dotnetMatch) {
      d = new Date(parseInt(dotnetMatch[1], 10));
    }
    // DD.MM.YYYY (from EGRUL raw field)
    else if (/^\d{2}\.\d{2}\.\d{4}$/.test(dateStr)) {
      const [dd, mm, yyyy] = dateStr.split('.');
      d = new Date(`${yyyy}-${mm}-${dd}T00:00:00`);
    }
    // ISO-8601 and everything else — let the engine parse
    else {
      d = new Date(dateStr);
    }

    if (!d || isNaN(d.getTime())) return dateStr;
    return d.toLocaleDateString('ru-RU', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
    });
  } catch {
    return dateStr;
  }
};
