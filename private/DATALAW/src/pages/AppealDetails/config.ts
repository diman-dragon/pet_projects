// src/pages/AppealDetails/config.ts

export const STATUS_CONFIG: Record<string, { color: string }> = {
  DRAFT: { color: 'bg-black/[0.06] text-black/50' },
  SENT: { color: 'bg-blue-50 text-blue-600 border border-blue-200' },
  RECEIVED: { color: 'bg-green-50 text-green-600 border border-green-200' },
  CLOSED: { color: 'bg-red-50 text-red-500 border border-red-200' },
  REJECTED: { color: 'bg-orange-50 text-orange-600 border border-orange-200' },
};

/** Стили бейджей типов документов (код типа в таблице — как в имени файла) */
export const DOC_TYPE_META: Record<string, { color: string; dot: string }> = {
  DOC: { color: 'bg-slate-100 text-slate-600', dot: 'bg-slate-400' },
  ISK: { color: 'bg-blue-50 text-blue-700', dot: 'bg-blue-500' },
  ZAY: { color: 'bg-indigo-50 text-indigo-700', dot: 'bg-indigo-500' },
  ZHB: { color: 'bg-orange-50 text-orange-700', dot: 'bg-orange-500' },
  OTV: { color: 'bg-green-50 text-green-700', dot: 'bg-green-500' },
  ACT: { color: 'bg-yellow-50 text-yellow-700', dot: 'bg-yellow-500' },
  DOG: { color: 'bg-purple-50 text-purple-700', dot: 'bg-purple-500' },
  PRO: { color: 'bg-cyan-50 text-cyan-700', dot: 'bg-cyan-500' },
  REK: { color: 'bg-red-50 text-red-700', dot: 'bg-red-500' },
  UVD: { color: 'bg-teal-50 text-teal-700', dot: 'bg-teal-500' },
  SPR: { color: 'bg-lime-50 text-lime-700', dot: 'bg-lime-500' },
  SUD: { color: 'bg-rose-50 text-rose-700', dot: 'bg-rose-600' },
  QUE: { color: 'bg-sky-50 text-sky-700', dot: 'bg-sky-500' },
  RPT: { color: 'bg-violet-50 text-violet-700', dot: 'bg-violet-500' },
  OTH: { color: 'bg-gray-50 text-gray-600', dot: 'bg-gray-400' },
};

export const DEFAULT_DOC_META = { color: 'bg-gray-50 text-gray-600', dot: 'bg-gray-400' };

// ИСПРАВЛЕНО: Добавлен экспорт UPLOAD_CONFIG для AppealFileService
export const UPLOAD_CONFIG = {
  MAX_FILE_SIZE_BYTES: 50 * 1024 * 1024,
  ALLOWED_EXTENSIONS: [
    '.pdf',
    '.doc',
    '.docx',
    '.xls',
    '.xlsx',
    '.jpg',
    '.jpeg',
    '.png',
    '.webp',
    '.txt',
  ],
};

// ── Утилиты форматирования ────────────────────────────────────────────────────

export function formatDate(date?: string | null, localeTag = 'ru-RU'): string {
  if (!date) return '—';
  try {
    return new Date(date).toLocaleDateString(localeTag, {
      day: '2-digit',
      month: '2-digit',
      year: 'numeric',
    });
  } catch {
    return date;
  }
}

export function formatDateShort(date: Date, localeTag = 'ru-RU'): string {
  return date.toLocaleDateString(localeTag, { day: '2-digit', month: '2-digit', year: '2-digit' });
}

/** Иконка расширения */
export function extIcon(ext: string): string {
  const e = ext.toLowerCase();
  if (['.pdf'].includes(e)) return '📄';
  if (['.doc', '.docx'].includes(e)) return '📝';
  if (['.xls', '.xlsx'].includes(e)) return '📊';
  if (['.jpg', '.jpeg', '.png', '.webp'].includes(e)) return '🖼';
  if (['.txt'].includes(e)) return '📃';
  return '📎';
}

/** Цвет бейджа расширения */
export function extColor(ext: string): string {
  const e = ext.toLowerCase();
  if (e === '.pdf') return 'bg-red-100 text-red-700';
  if (['.doc', '.docx'].includes(e)) return 'bg-blue-100 text-blue-700';
  if (['.xls', '.xlsx'].includes(e)) return 'bg-emerald-100 text-emerald-700';
  if (['.jpg', '.jpeg', '.png', '.webp'].includes(e)) return 'bg-violet-100 text-violet-700';
  if (e === '.txt') return 'bg-gray-100 text-gray-600';
  return 'bg-gray-100 text-gray-500';
}
