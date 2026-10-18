import { useState, useEffect, useMemo } from "react";
import { Flame } from "lucide-react";
import type { ForecastDay } from "../../types/domain";
import { StudyService } from "../../services/db/StudyService";
import { formatDateKey, getActivityColor } from "./calendarUtils";

interface ActivityCalendarProps {
  onDayClick?: (date: string) => void;
  selectedDate?: string;
}

const WEEK_DAYS = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"];
const MONTHS = ["Янв", "Фев", "Мар", "Апр", "Май", "Июн", "Июл", "Авг", "Сен", "Окт", "Ноя", "Дек"];

export function ActivityCalendar({ onDayClick, selectedDate }: ActivityCalendarProps) {
  const [forecast, setForecast] = useState<ForecastDay[]>([]);
  const [today, setToday] = useState<string>("");

  useEffect(() => {
    const now = new Date();
    setToday(formatDateKey(now));
    StudyService.getDueForecast(42).then(setForecast);
  }, []);

  const weeks = useMemo(() => {
    const todayDate = new Date();
    const start = new Date(todayDate);
    start.setDate(todayDate.getDate() - 27);

    const days = Array.from({ length: 28 }, (_, i) => {
      const d = new Date(start);
      d.setDate(start.getDate() + i);
      const key = formatDateKey(d);
      const forecastDay = forecast.find((f) => f.date === key);
      return { date: key, count: forecastDay?.count ?? 0, isToday: key === today };
    });

    return Array.from({ length: 4 }, (_, i) => days.slice(i * 7, (i + 1) * 7));
  }, [forecast, today]);

  const monthLabel = useMemo(() => {
    if (!weeks.length) return "";
    const first = new Date(weeks[0][0].date + "T00:00:00").getMonth();
    const last = new Date(weeks[3][6].date + "T00:00:00").getMonth();
    return first === last ? MONTHS[first] : `${MONTHS[first]}—${MONTHS[last]}`;
  }, [weeks]);

  return (
    <div className="rounded-lg bg-white p-2 shadow-sm ring-1 ring-gray-100">
      <div className="mb-1.5 flex items-center justify-between">
        <div className="flex items-center gap-1">
          <Flame size={12} className="text-orange-500" />
          <span className="text-[10px] font-semibold text-gray-700">{monthLabel}</span>
        </div>
        <div className="flex items-center gap-1 text-[8px] text-gray-400">
          <div className="flex gap-px">
            {[0, 1, 5, 10, 20].map((c) => (
              <div key={c} className="h-2 w-2 rounded-sm" style={{ backgroundColor: getActivityColor(c) }} />
            ))}
          </div>
        </div>
      </div>

      <div className="flex gap-px">
        {WEEK_DAYS.map((d) => (
          <div key={d} className="flex-1 text-center text-[7px] text-gray-400">{d}</div>
        ))}
      </div>

      <div className="mt-px flex gap-px">
        {weeks.map((week, wi) => (
          <div key={wi} className="flex flex-1 flex-col gap-px">
            {week.map((day) => (
              <button
                key={day.date}
                onClick={() => onDayClick?.(day.date)}
                className={`aspect-square rounded-[2px] transition-all ${selectedDate === day.date ? "ring-1 ring-blue-500" : ""
                  } ${day.isToday ? "ring-1 ring-gray-400" : ""}`}
                title={`${day.date}: ${day.count}`}
                type="button"
              >
                <div
                  className="h-full w-full rounded-[2px]"
                  style={{ backgroundColor: getActivityColor(day.count) }}
                />
              </button>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}
