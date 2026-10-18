import type { ForecastDay } from "../../types/domain";

function formatDateRU(iso: string) {
  const [y, m, d] = iso.split("-");
  const months = ["янв","фев","мар","апр","май","июн","июл","авг","сен","окт","ноя","дек"];
  return `${parseInt(d)} ${months[parseInt(m)-1]} ${y}`;
}

export function heatColor(kind: ForecastDay["kind"], count: number, max: number, reviewed?: number): string {
  if (kind === "today") {
    const done = reviewed ?? 0;
    if (done === 0 && count === 0) return "bg-[#0078D4]/30 border border-[#0078D4]/40";
    if (done === 0) return "bg-[#0078D4]/40 border border-[#0078D4]/50";
    return done >= count * 0.7 ? "bg-[#0078D4]" : "bg-[#0078D4]/70";
  }
  if (count === 0) {
    if (kind === "past") return "bg-gray-100";
    return "bg-slate-50 border border-dashed border-slate-200";
  }
  const r = count / Math.max(max, 1);
  if (kind === "past")  return r >= 0.7 ? "bg-emerald-500" : r >= 0.35 ? "bg-emerald-400" : "bg-emerald-200";
  return r >= 0.7 ? "bg-rose-400" : r >= 0.35 ? "bg-amber-400" : "bg-amber-200";
}

interface ForecastDayCellProps {
  day: ForecastDay;
  max: number;
  selected?: boolean;
  onClick?: () => void;
}

export function ForecastDayCell({ day, max, selected, onClick }: ForecastDayCellProps) {
  const isToday  = day.kind === "today";
  const isPast   = day.kind === "past";
  const isFuture = day.kind === "future";
  const reviewedCount = day.reviewed ?? 0;
  const bg = heatColor(day.kind, day.count, max, day.reviewed);

  const tooltip =
    isPast
      ? (day.count > 0 ? `Повторений: ${day.count}` : "Нет занятий")
      : isToday
        ? (reviewedCount > 0
            ? `Сегодня: ${reviewedCount} повторено, осталось ${day.count}`
            : `Сегодня: запланировано ${day.count}`)
        : (day.count > 0 ? `Запланировано: ${day.count}` : "Ничего не запланировано");

  // Кликабельны все дни где что-то было — и past и today
  const hasData = isPast ? day.count > 0 : isToday ? (day.count > 0 || reviewedCount > 0) : false;
  const clickable = hasData;

  return (
    <div className="relative group">
      <button
        type="button"
        onClick={clickable ? onClick : undefined}
        className={[
          "h-7 w-7 rounded-lg transition-all",
          bg,
          isToday ? "ring-2 ring-[#0078D4] ring-offset-1 scale-110 shadow-md shadow-blue-400/30" : "",
          selected ? "ring-2 ring-offset-1 ring-gray-800 scale-110 shadow-md" : "",
          clickable ? "cursor-pointer hover:scale-110 hover:shadow-md hover:brightness-110" : "cursor-default",
          isFuture && day.count === 0 ? "opacity-40" : "",
        ].filter(Boolean).join(" ")}
      />
      {/* Tooltip */}
      <div className="pointer-events-none absolute bottom-full left-1/2 z-30 mb-2 -translate-x-1/2 opacity-0 group-hover:opacity-100 transition-opacity duration-150">
        <div className="whitespace-nowrap rounded-xl bg-gray-900/95 backdrop-blur-sm px-3 py-2 text-center shadow-xl">
          <div className="text-[11px] font-semibold text-white">{tooltip}</div>
          <div className="text-[10px] text-gray-400 mt-0.5">{formatDateRU(day.date)}</div>
          {clickable && !selected && (
            <div className="text-[10px] text-blue-300 mt-0.5">↓ кликни — список карточек</div>
          )}
        </div>
        <div className="mx-auto mt-0.5 h-0 w-0 border-x-4 border-x-transparent border-t-4 border-t-gray-900/95" />
      </div>
    </div>
  );
}
