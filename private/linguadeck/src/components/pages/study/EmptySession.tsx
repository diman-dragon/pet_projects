import { StreakPopup } from "./StreakPopup";
import type { WeekActivityDay } from "../../../types/domain";

interface EmptySessionProps {
  streakOpen: boolean;
  streak: number;
  sessionCount: number;
  reviewedToday: number;
  totalTimeMs: number;
  weekActivity: WeekActivityDay[];
  weekBars: WeekActivityDay[];
  onCloseStreak: () => void;
}

export function EmptySession({
  streakOpen,
  streak,
  sessionCount,
  reviewedToday,
  totalTimeMs,
  weekActivity,
  weekBars,
  onCloseStreak,
}: EmptySessionProps) {
  return (
    <div className="flex h-full items-center justify-center p-4">
      <div className="rounded-2xl border-2 border-dashed border-gray-300 bg-white px-8 py-12 text-center shadow-sm">
        <p className="text-lg font-semibold text-gray-900">Нет активной сессии</p>
        <p className="mt-1 text-sm text-gray-500">Откройте страницу колод и запустите изучение</p>
      </div>
      <StreakPopup
        open={streakOpen}
        streak={streak}
        sessionCount={sessionCount}
        reviewedToday={reviewedToday}
        totalTimeMs={totalTimeMs}
        weekActivity={weekActivity.length > 0 ? weekActivity : weekBars}
        onClose={onCloseStreak}
      />
    </div>
  );
}
