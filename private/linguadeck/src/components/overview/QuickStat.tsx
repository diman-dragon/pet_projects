import { formatDuration } from "./calendarUtils";

interface QuickStatProps {
  icon: React.ReactNode;
  label: string;
  value: string | number;
  sub: string;
  accent?: boolean;
  color?: "rose" | "purple" | "default";
  isTime?: boolean;
}

const STYLES = {
  default: { bg: "bg-white", icon: "bg-blue-50 text-blue-600", border: "border-gray-100", val: "text-gray-900" },
  rose:    { bg: "bg-white", icon: "bg-rose-50 text-rose-500",  border: "border-gray-100", val: "text-rose-600" },
  purple:  { bg: "bg-white", icon: "bg-purple-50 text-purple-500", border: "border-gray-100", val: "text-purple-600" },
};

const ACCENT = { bg: "bg-gradient-to-br from-[#0078D4]/5 to-blue-50", icon: "bg-[#0078D4] text-white", border: "border-[#0078D4]/20", val: "text-[#0078D4]" };

export function QuickStat({ icon, label, value, sub, accent, color = "default", isTime }: QuickStatProps) {
  const s = accent ? ACCENT : STYLES[color];
  const displayValue = isTime ? formatDuration(Number(value)) : value;
  
  return (
    <div className={`flex flex-col gap-2 rounded-2xl border ${s.border} ${s.bg} p-4 shadow-sm`}>
      <div className={`flex h-9 w-9 items-center justify-center rounded-xl ${s.icon}`}>{icon}</div>
      <div>
        <div className="text-[10px] font-semibold uppercase tracking-wider text-gray-400">{label}</div>
        <div className={`text-2xl font-black tabular-nums ${s.val}`}>{displayValue}</div>
        <div className="mt-0.5 text-[11px] text-gray-400">{sub}</div>
      </div>
    </div>
  );
}
