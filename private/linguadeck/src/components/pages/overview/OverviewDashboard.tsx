import type { AppSettings } from "../../../types/domain";
import { MiniCalendar } from "../../overview/MiniCalendar";
import { ViewStats } from "../../overview/ViewStats";
import { DayCardList } from "../../overview/DayCardList";

interface OverviewDashboardProps {
  selectedDate: string;
  settings: AppSettings;
  onSelectDate: (date: string) => void;
}

export function OverviewDashboard({ selectedDate, settings, onSelectDate }: OverviewDashboardProps) {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
      <MiniCalendar onDayClick={onSelectDate} selectedDate={selectedDate} />
      <ViewStats selectedDate={selectedDate} settings={settings} />
      <DayCardList date={selectedDate} />
    </div>
  );
}
