import { useState, useEffect, useMemo } from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { StudyService } from "../../services/db/StudyService";

interface MiniCalendarProps {
  onDayClick?: (date: string) => void;
  selectedDate?: string;
}

const WEEK_DAYS = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"];
const MONTHS = ["Январь", "Февраль", "Март", "Апрель", "Май", "Июнь", "Июль", "Август", "Сентябрь", "Октябрь", "Ноябрь", "Декабрь"];

function formatDateKey(date: Date): string {
  return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}-${String(date.getDate()).padStart(2, "0")}`;
}

export function MiniCalendar({ onDayClick, selectedDate }: MiniCalendarProps) {
  const [currentMonth, setCurrentMonth] = useState(new Date());
  const [activity, setActivity] = useState<Array<{ date: string; count: number }>>([]);

  useEffect(() => {
    StudyService.getActivityHistory(365).then(setActivity);
  }, []);

  const { year, month, days } = useMemo(() => {
    const y = currentMonth.getFullYear();
    const m = currentMonth.getMonth();

    const firstDay = new Date(y, m, 1);
    const lastDay = new Date(y, m + 1, 0);
    const startOffset = (firstDay.getDay() + 6) % 7;

    const days: Array<{
      date: string;
      dayNum: number;
      count: number;
      isToday: boolean;
      isCurrentMonth: boolean;
    }> = [];

    const todayKey = formatDateKey(new Date());

    for (let i = 0; i < startOffset; i++) {
      const d = new Date(y, m, -startOffset + i + 1);
      const dateKey = formatDateKey(d);
      const activityDay = activity.find((a) => a.date === dateKey);
      days.push({
        date: dateKey,
        dayNum: d.getDate(),
        count: activityDay ? Number(activityDay.count) || 0 : 0,
        isToday: false,
        isCurrentMonth: false,
      });
    }

    for (let d = 1; d <= lastDay.getDate(); d++) {
      const dateKey = `${y}-${String(m + 1).padStart(2, "0")}-${String(d).padStart(2, "0")}`;
      const activityDay = activity.find((a) => a.date === dateKey);
      days.push({
        date: dateKey,
        dayNum: d,
        count: activityDay ? Number(activityDay.count) || 0 : 0,
        isToday: dateKey === todayKey,
        isCurrentMonth: true,
      });
    }

    const remaining = (7 - (days.length % 7)) % 7;
    for (let i = 1; i <= remaining; i++) {
      const d = new Date(y, m + 1, i);
      const dateKey = formatDateKey(d);
      const activityDay = activity.find((a) => a.date === dateKey);
      days.push({
        date: dateKey,
        dayNum: d.getDate(),
        count: activityDay ? Number(activityDay.count) || 0 : 0,
        isToday: false,
        isCurrentMonth: false,
      });
    }

    return { year: y, month: m, days };
  }, [currentMonth, activity]);

  const weeks = useMemo(() => {
    return Array.from({ length: days.length / 7 }, (_, i) => days.slice(i * 7, (i + 1) * 7));
  }, [days]);

  const prevMonth = () => setCurrentMonth(new Date(year, month - 1, 1));
  const nextMonth = () => setCurrentMonth(new Date(year, month + 1, 1));

  const getDayStyle = (day: typeof days[0]) => {
    if (selectedDate === day.date) {
      return "ring-2 ring-blue-500 ring-offset-2 bg-blue-50/50 scale-105";
    }
    if (day.isToday) {
      return "ring-2 ring-emerald-500 ring-offset-2 bg-emerald-50/50 scale-105";
    }
    if (!day.isCurrentMonth) {
      return "text-gray-300 hover:bg-gray-50/50";
    }
    return "hover:bg-gray-50 hover:shadow-sm";
  };

  const getDayTextStyle = (day: typeof days[0]) => {
    if (selectedDate === day.date) {
      return "text-blue-600 font-bold";
    }
    if (day.isToday) {
      return "text-emerald-600 font-bold";
    }
    if (!day.isCurrentMonth) {
      return "text-gray-300 font-medium";
    }
    return "text-gray-700 font-medium";
  };

  return (
    <div className="w-[280px] rounded-2xl bg-gradient-to-br from-white to-gray-50 p-4 shadow-lg shadow-gray-200/50 ring-1 ring-gray-100">
      {/* Header */}
      <div className="mb-3 flex items-center justify-between">
        <button
          onClick={prevMonth}
          className="group rounded-xl bg-gradient-to-br from-gray-50 to-gray-100 p-2 transition-all hover:shadow-md hover:from-blue-50 hover:to-blue-100 hover:scale-105 active:scale-95"
          type="button"
        >
          <ChevronLeft size={18} className="text-gray-400 transition-colors group-hover:text-blue-500" />
        </button>

        <div className="text-center">
          <span className="block text-sm font-black bg-gradient-to-r from-gray-800 to-gray-600 bg-clip-text text-transparent tracking-tight">
            {MONTHS[month]}
          </span>
          <span className="text-[10px] font-semibold text-gray-400 tracking-wider">
            {year}
          </span>
        </div>

        <button
          onClick={nextMonth}
          className="group rounded-xl bg-gradient-to-br from-gray-50 to-gray-100 p-2 transition-all hover:shadow-md hover:from-blue-50 hover:to-blue-100 hover:scale-105 active:scale-95"
          type="button"
        >
          <ChevronRight size={18} className="text-gray-400 transition-colors group-hover:text-blue-500" />
        </button>
      </div>

      {/* Week days */}
      <div className="mb-2 grid grid-cols-7 gap-1">
        {WEEK_DAYS.map((d, i) => (
          <div
            key={d}
            className={`text-center text-[10px] font-bold uppercase tracking-wide ${i >= 5 ? "text-rose-400" : "text-indigo-400"
              }`}
          >
            {d}
          </div>
        ))}
      </div>

      {/* Days grid */}
      <div className="grid grid-cols-7 gap-1">
        {weeks.flatMap((week) =>
          week.map((day) => (
            <button
              key={day.date}
              onClick={() => onDayClick?.(day.date)}
              className={`relative aspect-square rounded-xl p-1 transition-all duration-200 ${getDayStyle(day)}`}
              type="button"
            >
              <div className="flex h-full flex-col items-center justify-between">
                <span className={`text-[12px] ${getDayTextStyle(day)}`}>
                  {day.dayNum}
                </span>
                {day.count > 0 && (
                  <div
                    className="h-2 w-2 rounded-full shadow-md"
                    style={{
                      background: day.count > 10
                        ? "linear-gradient(135deg, #10b981, #059669)"
                        : day.count > 5
                          ? "linear-gradient(135deg, #3b82f6, #2563eb)"
                          : "linear-gradient(135deg, #f59e0b, #d97706)"
                    }}
                  />
                )}
              </div>
            </button>
          ))
        )}
      </div>
    </div>
  );
}
