interface DistLegendProps {
  color: string;
  label: string;
  value: number;
  pct: number;
}

export function DistLegend({ color, label, value, pct }: DistLegendProps) {
  return (
    <div className="flex items-center gap-2">
      <div className={`h-2.5 w-2.5 shrink-0 rounded-full ${color}`} />
      <div>
        <div className="text-[10px] text-gray-400">{label}</div>
        <div className="text-xs font-bold text-gray-800">
          {value} <span className="font-normal text-gray-400">({pct}%)</span>
        </div>
      </div>
    </div>
  );
}
