interface SessionCardProps {
  icon: React.ReactNode;
  label: string;
  value: number;
  color: "blue" | "emerald" | "cyan" | "indigo";
}

const STYLES = {
  blue:    { bg: "bg-blue-50",    text: "text-blue-600",    val: "text-blue-700" },
  emerald: { bg: "bg-emerald-50", text: "text-emerald-600", val: "text-emerald-700" },
  cyan:    { bg: "bg-cyan-50",    text: "text-cyan-600",    val: "text-cyan-700" },
  indigo:  { bg: "bg-indigo-50",  text: "text-indigo-600",  val: "text-indigo-700" },
};

export function SessionCard({ icon, label, value, color }: SessionCardProps) {
  const s = STYLES[color];
  return (
    <div className="flex items-center gap-3 rounded-2xl border border-gray-100 bg-white p-4 shadow-sm transition-all hover:shadow-md">
      <div className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-xl ${s.bg} ${s.text}`}>{icon}</div>
      <div>
        <div className="text-[10px] font-semibold uppercase tracking-wider text-gray-400">{label}</div>
        <div className={`text-2xl font-black tabular-nums ${s.val}`}>{value}</div>
      </div>
    </div>
  );
}
