import { useEffect, useState } from "react";
import { Calendar, TrendingUp } from "lucide-react";
import { StudyService } from "../../services/db/StudyService";
import type { ForecastDay } from "../../types/domain";
import { ForecastDayCell } from "./ForecastDayCell";
import { DayPanel } from "./forecast/DayPanel";
import { WEEKDAYS } from "./forecast/forecastUtils";

export function ForecastChart({ days = 14 }: { days?: number }) {
  const [data, setData] = useState<ForecastDay[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedDate, setSelected] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    const load = () => {
      StudyService.getDueForecast(days)
        .then((d) => { if (mounted) { setData(d); setLoading(false); } })
        .catch(() => { if (mounted) setLoading(false); });
    };
    load();
    const interval = setInterval(load, 30_000);
    return () => { mounted = false; clearInterval(interval); };
  }, [days]);

  if (loading) return (
    <div className="rounded-2xl border border-gray-100 bg-white p-5 shadow-sm">
      <div className="flex items-center gap-2 text-sm text-gray-400">
        <div className="h-4 w-4 animate-spin rounded-full border-2 border-[#0078D4] border-t-transparent" />
        Загрузка...
      </div>
    </div>
  );

  if (data.length === 0) return null;

  const maxCount = Math.max(...data.map((d) => d.count), 1);
  const firstWd = data[0]?.weekday ?? 0;
  const activeDays = data.filter((d) => d.kind === "past" && d.count > 0).length;
  const totalFuture = data.filter((d) => d.kind === "future").reduce((s, d) => s + d.count, 0);
  const todayEntry = data.find((d) => d.kind === "today");

  return (
    <div className="rounded-2xl border border-gray-100 bg-white shadow-sm overflow-hidden">
      <div className={`flex transition-all ${selectedDate ? "divide-x divide-gray-100" : ""}`}>

        {/* Calendar */}
        <div className="flex-1 min-w-0 p-5">

          {/* Header */}
          <div className="mb-4 flex items-start justify-between">
            <div className="flex items-center gap-3">
              <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-[#0078D4]/10">
                <Calendar size={16} className="text-[#0078D4]" />
              </div>
              <div>
                <h3 className="text-sm font-bold text-gray-900">История и прогноз</h3>
                <p className="text-[11px] text-gray-400">последние 30 дней · {days} дней вперёд</p>
              </div>
            </div>
            <div className="flex flex-col items-end gap-1">
              <div className="flex items-center gap-1.5 rounded-full bg-emerald-50 px-2.5 py-1 text-[11px] text-emerald-700 font-medium">
                <TrendingUp size={10} />{activeDays} активных дней
              </div>
              {totalFuture > 0 && (
                <div className="flex items-center gap-1.5 rounded-full bg-amber-50 px-2.5 py-1 text-[11px] text-amber-700 font-medium">
                  <Calendar size={10} />{totalFuture} впереди
                </div>
              )}
            </div>
          </div>

          {/* Today highlight */}
          {todayEntry && (
            <button
              type="button"
              onClick={() => setSelected((prev) => (prev === todayEntry.date ? null : todayEntry.date))}
              className="mb-4 w-full flex items-center justify-between rounded-xl border border-[#0078D4]/20 bg-[#0078D4]/5 px-4 py-2.5 hover:bg-[#0078D4]/10 transition-all cursor-pointer text-left"
            >
              <div className="flex items-center gap-2">
                <div className="h-2.5 w-2.5 rounded-full bg-[#0078D4] animate-pulse" />
                <span className="text-[12px] font-semibold text-[#0078D4]">
                  {(todayEntry.reviewed ?? 0) > 0
                    ? `Сегодня: ${todayEntry.reviewed} повторений, ещё ${todayEntry.count} осталось`
                    : `Сегодня: ${todayEntry.count} карточек к повторению`}
                </span>
              </div>
              <span className="text-[10px] text-[#0078D4]/60">открыть →</span>
            </button>
          )}

          {/* Weekday labels */}
          <div className="mb-1.5 grid grid-cols-7 gap-1">
            {WEEKDAYS.map((d) => (
              <div key={d} className="text-center text-[10px] font-semibold text-gray-300">{d}</div>
            ))}
          </div>

          {/* Grid */}
          <div className="grid grid-cols-7 gap-1">
            {Array.from({ length: firstWd }).map((_, i) => <div key={`p${i}`} className="h-7 w-7" />)}
            {data.map((day) => (
              <ForecastDayCell
                key={day.date}
                day={day}
                max={maxCount}
                selected={day.date === selectedDate}
                onClick={() => setSelected((prev) => (prev === day.date ? null : day.date))}
              />
            ))}
          </div>

          {/* Legend */}
          <div className="mt-4 flex flex-wrap items-center gap-x-3 gap-y-1.5 text-[10px] text-gray-400">
            {[
              { bg: "bg-gray-100",     label: "Нет"          },
              { bg: "bg-emerald-200",  label: "Мало"         },
              { bg: "bg-emerald-500",  label: "Много"        },
              { bg: "bg-[#0078D4]",    label: "Сегодня"      },
              { bg: "bg-amber-300",    label: "Запланир."    },
              { bg: "bg-rose-400",     label: "Много (план)" },
            ].map(({ bg, label }) => (
              <div key={label} className="flex items-center gap-1">
                <div className={`h-2.5 w-2.5 rounded ${bg}`} />{label}
              </div>
            ))}
          </div>
        </div>

        {/* Detail panel */}
        {selectedDate && (
          <div className="w-80 shrink-0 border-l border-gray-100">
            <DayPanel date={selectedDate} onClose={() => setSelected(null)} />
          </div>
        )}
      </div>
    </div>
  );
}
