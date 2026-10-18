export const WEEKDAYS = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"];

export const RATING_META = {
  1: { label: "Снова",  short: "С", color: "text-rose-700",    bg: "bg-rose-100",    bar: "bg-rose-400"    },
  2: { label: "Сложно", short: "Т", color: "text-orange-700",  bg: "bg-orange-100",  bar: "bg-orange-400"  },
  3: { label: "Хорошо", short: "Х", color: "text-blue-700",    bg: "bg-blue-100",    bar: "bg-blue-400"    },
  4: { label: "Легко",  short: "Л", color: "text-emerald-700", bg: "bg-emerald-100", bar: "bg-emerald-400" },
} as const;

export function fmtMs(ms: number): string {
  if (ms < 1000) return `${ms}мс`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}с`;
  return `${Math.floor(ms / 60_000)}м ${Math.round((ms % 60_000) / 1000)}с`;
}

export function fmtInterval(days: number): string {
  if (!days || days < 0.05) return "< 1 дня";
  if (days < 30) return `${Math.round(days)} дн.`;
  if (days < 365) return `${Math.round(days / 30)} мес.`;
  return `${(days / 365).toFixed(1)} г.`;
}

export function formatDateRU(iso: string): string {
  const [y, m, d] = iso.split("-");
  const months = [
    "января", "февраля", "марта", "апреля", "мая", "июня",
    "июля", "августа", "сентября", "октября", "ноября", "декабря",
  ];
  return `${parseInt(d)} ${months[parseInt(m) - 1]} ${y}`;
}

export function Chip({
  icon, value, label,
  color = "text-gray-600",
  bg = "bg-gray-50",
}: {
  icon: React.ReactNode;
  value: string | number;
  label: string;
  color?: string;
  bg?: string;
}) {
  return (
    <div className={`flex items-center gap-1.5 rounded-xl px-3 py-2 ${bg}`}>
      <span className={`shrink-0 ${color}`}>{icon}</span>
      <div>
        <div className={`text-sm font-black tabular-nums leading-tight ${color}`}>{value}</div>
        <div className="text-[10px] text-gray-400 leading-tight">{label}</div>
      </div>
    </div>
  );
}
