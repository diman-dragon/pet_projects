import { useState } from "react";
import type { AppSettings, DashboardStats } from "../../types/domain";
import { MiniCalendar } from "../overview/MiniCalendar";
import { ViewStats } from "../overview/ViewStats";
import { DayCardList } from "../overview/DayCardList";
import { LearningProgressChart } from "../overview/LearningProgressChart";
import { StatsGrid } from "./overview/StatsGrid";
import { QualityMetrics } from "../overview/QualityMetrics";
import { motion } from "framer-motion";

interface OverviewPageProps {
  settings: AppSettings;
  stats: DashboardStats;
}

export function OverviewPage({ settings, stats }: OverviewPageProps) {
  const [selectedDate, setSelectedDate] = useState<string>(new Date().toISOString().split('T')[0]);

  return (
    <motion.section 
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-8 p-6 max-w-7xl mx-auto pb-20 font-sans"
    >
      <header className="mb-8">
        <h1 className="text-3xl font-semibold text-slate-900 tracking-tight">Дашборд</h1>
        <p className="text-sm text-slate-500 mt-1">Обзор вашей учебной активности и прогресса</p>
      </header>

      {/* 1. Top Section: Date specific and Retention */}
      <div className="grid grid-cols-1 xl:grid-cols-12 gap-6 items-start">
        <div className="xl:col-span-3 rounded-xl border border-white/40 bg-white/60 backdrop-blur-md shadow-sm p-1">
          <MiniCalendar onDayClick={setSelectedDate} selectedDate={selectedDate} />
        </div>
        <div className="xl:col-span-4 h-full">
          <ViewStats selectedDate={selectedDate} settings={settings} />
        </div>
        <div className="xl:col-span-5 flex flex-col gap-6">
           <DayCardList date={selectedDate} />
        </div>
      </div>

      {/* 2. Global Progress */}
      <section className="space-y-4">
        <div className="flex items-center gap-3">
          <h2 className="text-xl font-semibold text-slate-800 tracking-tight">Статистика сегодня</h2>
          <div className="h-px flex-1 bg-slate-200" />
        </div>
        <StatsGrid stats={stats} />
      </section>

      {/* 3. Detailed Metrics */}
      <section className="space-y-4">
        <div className="flex items-center gap-3">
          <h2 className="text-xl font-semibold text-slate-800 tracking-tight">Аналитика качества</h2>
          <div className="h-px flex-1 bg-slate-200" />
        </div>
        <QualityMetrics stats={stats} />
      </section>
      
      {/* 4. Charts */}
      <section className="space-y-4">
        <div className="flex items-center gap-3">
          <h2 className="text-xl font-semibold text-slate-800 tracking-tight">Динамика обучения</h2>
          <div className="h-px flex-1 bg-slate-200" />
        </div>
        <LearningProgressChart stats={stats} />
      </section>
    </motion.section>
  );
}

