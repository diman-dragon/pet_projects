interface LicenseInfoRowProps {
  label: string;
  value: string;
  mono?: boolean;
}

export function LicenseInfoRow({ label, value, mono = false }: LicenseInfoRowProps) {
  return (
    <div className="flex items-baseline gap-4 px-5 py-2.5">
      <span className="w-28 shrink-0 text-[9px] font-black uppercase tracking-[0.2em] text-black/30">
        {label}
      </span>
      <span
        className={`break-all leading-relaxed text-black/65 ${mono ? 'font-mono text-[10px]' : 'text-[11px] font-medium'}`}
      >
        {value}
      </span>
    </div>
  );
}
