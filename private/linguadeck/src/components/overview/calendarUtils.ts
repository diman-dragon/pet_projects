// FIX: ранее при времени < 60с показывалось «0м».
// Теперь для коротких сессий показываем секунды.
export function formatDuration(ms: number): string {
  if (ms <= 0) return "0м";

  const totalSeconds = Math.floor(ms / 1000);
  const totalMinutes = Math.floor(ms / 60000);
  const hours = Math.floor(totalMinutes / 60);
  const minutes = totalMinutes % 60;

  if (hours > 0) {
    return `${hours}ч ${minutes}м`;
  }
  if (totalMinutes > 0) {
    return `${minutes}м`;
  }
  return `${totalSeconds}с`;
}

export function formatNumber(n: number): string {
  return n.toLocaleString("ru-RU");
}

export function formatDateKey(date: Date): string {
  const y = date.getFullYear();
  const m = String(date.getMonth() + 1).padStart(2, "0");
  const d = String(date.getDate()).padStart(2, "0");
  return `${y}-${m}-${d}`;
}

export function getActivityColor(count: number): string {
  if (count === 0) return "#f1f5f9";
  if (count < 5) return "#93c5fd";
  if (count < 15) return "#3b82f6";
  return "#1e40af";
}
