export function fileIcon(type: string) {
  if (type === 'pdf') return 'PDF';
  if (type === 'docx') return 'DOC';
  if (type === 'md' || type === 'markdown') return 'MD';
  return 'FILE';
}

export function formatDate(dt: string | null, locale: string): string {
  if (!dt) {
    return '—';
  }
  try {
    return new Date(dt).toLocaleDateString(locale, {
      day: '2-digit',
      month: '2-digit',
      year: '2-digit',
    });
  } catch {
    return dt;
  }
}
