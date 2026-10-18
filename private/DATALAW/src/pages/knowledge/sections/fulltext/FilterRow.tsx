export function FilterRow({
  icon,
  label,
  children,
}: {
  icon: React.ReactNode;
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex items-center gap-3 border-b border-black/[0.04] py-2 last:border-0">
      <div className="flex w-32 shrink-0 items-center gap-1.5">
        <span className="flex h-3.5 w-3.5 shrink-0 items-center justify-center text-black/35">
          {icon}
        </span>
        <span className="text-[10px] font-semibold uppercase tracking-wider text-black/50">
          {label}
        </span>
      </div>
      <div className="flex-1">{children}</div>
    </div>
  );
}
