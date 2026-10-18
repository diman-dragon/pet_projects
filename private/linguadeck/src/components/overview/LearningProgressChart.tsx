import type { DashboardStats, WeekActivityDay } from "../../types/domain";
import { formatDuration } from "./calendarUtils";
import { useState, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";

interface LearningProgressChartProps {
  stats: DashboardStats;
}

type ChartMode = "time" | "cards";

function ActivityChart({ data, mode }: { data: WeekActivityDay[]; mode: ChartMode }) {
  const [hovered, setHovered] = useState<number | null>(null);
  
  const width = 800;
  const height = 350;
  const pad = { top: 60, right: 40, bottom: 60, left: 60 };
  const innerW = width - pad.left - pad.right;
  const innerH = height - pad.top - pad.bottom;
  
  const values = data.map((day) => (mode === "time" ? day.timeMs || 0 : day.count || 0));
  const maxValue = Math.max(1, ...values) * 1.2;
  
  const getX = (i: number) => pad.left + (data.length <= 1 ? innerW / 2 : (i / (data.length - 1)) * innerW);
  const getY = (value: number) => pad.top + innerH - (value / maxValue) * innerH;
  
  const points = values.map((value, i) => ({ x: getX(i), y: getY(value), value, date: data[i].date }));

  const linePath = useMemo(() => {
    if (points.length < 2) return "";
    let d = `M ${points[0].x} ${points[0].y}`;
    for (let i = 0; i < points.length - 1; i++) {
      const p0 = points[i];
      const p1 = points[i + 1];
      const cp1x = p0.x + (p1.x - p0.x) / 3;
      const cp2x = p1.x - (p1.x - p0.x) / 3;
      d += ` C ${cp1x} ${p0.y}, ${cp2x} ${p1.y}, ${p1.x} ${p1.y}`;
    }
    return d;
  }, [points]);

  const areaPath = useMemo(() => {
    if (!linePath) return "";
    return `${linePath} L ${points[points.length - 1].x} ${pad.top + innerH} L ${points[0].x} ${pad.top + innerH} Z`;
  }, [linePath, points, pad.top, innerH]);

  const color = mode === "time" ? "#0067c0" : "#60cdff"; 

  return (
    <div className="group relative overflow-hidden rounded-lg border border-white/40 bg-white/70 shadow-sm backdrop-blur-xl transition-all hover:bg-white/80">
      <div className="relative p-6">
        <div className="mb-6 flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold text-slate-900">
              {mode === "time" ? "Время в фокусе" : "Объем памяти"}
            </h3>
            <p className="text-xs text-slate-500">
              {mode === "time" ? "Минуты активного обучения" : "Количество повторений"}
            </p>
          </div>
          <div className="text-right">
             <div className="text-[10px] font-bold uppercase tracking-wider text-slate-400">Пик</div>
             <div className="text-lg font-semibold text-slate-700 tabular-nums">
                {mode === 'time' ? formatDuration(Math.max(...values)) : Math.max(...values)}
             </div>
          </div>
        </div>

        <div className="relative">
          <svg viewBox={`0 0 ${width} ${height}`} className="w-full overflow-visible">
            <defs>
              <linearGradient id={`areaGradient-${mode}`} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={color} stopOpacity="0.15" />
                <stop offset="100%" stopColor={color} stopOpacity="0" />
              </linearGradient>
              <filter id="winGlow" x="-20%" y="-20%" width="140%" height="140%">
                <feGaussianBlur stdDeviation="3" result="blur" />
                <feComposite in="SourceGraphic" in2="blur" operator="over" />
              </filter>
            </defs>

            {/* Grid */}
            {[0, 0.5, 1].map((tick) => (
              <line
                key={tick}
                x1={pad.left}
                y1={pad.top + innerH * (1 - tick)}
                x2={pad.left + innerW}
                y2={pad.top + innerH * (1 - tick)}
                stroke="rgba(0,0,0,0.05)"
                strokeWidth="1"
              />
            ))}

            <motion.path
              d={areaPath}
              fill={`url(#areaGradient-${mode})`}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
            />

            <motion.path
              id={`path-${mode}`}
              d={linePath}
              fill="none"
              stroke={color}
              strokeWidth="3"
              strokeLinecap="round"
              initial={{ pathLength: 0 }}
              animate={{ pathLength: 1 }}
              transition={{ duration: 1.5, ease: "easeOut" }}
              filter="url(#winGlow)"
            />

            {/* Character - Non-synchronized movement, FORCED Strictly Vertical */}
            <motion.g
              initial={{ offsetDistance: "0%", offsetRotate: "0deg" }}
              animate={{ offsetDistance: "100%", offsetRotate: "0deg" }}
              transition={{ 
                duration: mode === 'time' ? 14 : 18, 
                repeat: Infinity, 
                ease: "linear",
                delay: mode === 'time' ? 0 : 3 
              }}
              style={{ 
                offsetPath: `path('${linePath}')`,
                offsetRotate: "0deg"
              }}
            >
              <motion.text
                y="-5" 
                x="-19" 
                className="select-none"
                animate={{ 
                  y: [-5, -12, -5] 
                }}
                transition={{
                  duration: 0.6,
                  repeat: Infinity,
                  ease: "easeInOut"
                }}
                style={{ 
                  fontSize: '38px', 
                  filter: 'drop-shadow(0 4px 12px rgba(0,0,0,0.15))',
                  display: 'block' 
                }}
              >
                {mode === 'time' ? '🚶‍♂️' : '🚶‍♀️'}
              </motion.text>
            </motion.g>

            {/* Interactive Points */}
            {points.map((p, i) => (
              <g
                key={i}
                onMouseEnter={() => setHovered(i)}
                onMouseLeave={() => setHovered(null)}
                className="cursor-pointer"
              >
                <rect x={p.x - 20} y={pad.top} width="40" height={innerH} fill="transparent" />
                <motion.circle
                  cx={p.x}
                  cy={p.y}
                  r={hovered === i ? 6 : 0}
                  fill={color}
                  stroke="white"
                  strokeWidth="2"
                />
              </g>
            ))}
          </svg>

          {/* Tooltip Windows 11 Style */}
          <AnimatePresence>
            {hovered !== null && (
              <motion.div
                initial={{ opacity: 0, scale: 0.9, y: 10 }}
                animate={{ opacity: 1, scale: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.9, y: 10 }}
                className="pointer-events-none absolute z-20 rounded-md border border-white/40 bg-white/90 px-3 py-2 text-xs shadow-lg backdrop-blur-md"
                style={{
                  left: `${(points[hovered].x / width) * 100}%`,
                  top: `${((points[hovered].y - 45) / height) * 100}%`,
                  transform: "translateX(-50%)",
                }}
              >
                <div className="font-bold text-slate-900">
                  {new Date(points[hovered].date).toLocaleDateString('ru-RU', { day: 'numeric', month: 'short' })}
                </div>
                <div className="text-slate-600 font-medium">
                  {mode === 'time' ? formatDuration(points[hovered].value) : points[hovered].value + ' карт.'}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
}

export function LearningProgressChart({ stats }: LearningProgressChartProps) {
  const data = stats.weekActivity || [];
  if (!data.length) return (
    <div className="flex h-48 items-center justify-center rounded-lg border border-dashed border-slate-200 bg-white/50 p-8 text-center">
      <p className="text-sm font-medium text-slate-400">Нет данных для анализа</p>
    </div>
  );

  return (
    <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
      <ActivityChart data={data} mode="time" />
      <ActivityChart data={data} mode="cards" />
    </div>
  );
}
