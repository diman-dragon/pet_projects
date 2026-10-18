import React, { forwardRef } from "react";
import { ChevronDown } from "lucide-react";

interface SelectOption {
  value: string | number;
  label: string;
}

interface SelectProps extends Omit<React.SelectHTMLAttributes<HTMLSelectElement>, "onChange"> {
  label?: string;
  error?: string;
  options: SelectOption[];
  onChange: (value: string) => void;
  containerClassName?: string;
}

export const Select = forwardRef<HTMLSelectElement, SelectProps>(({
  label,
  error,
  options,
  onChange,
  className = "",
  containerClassName = "",
  id,
  value,
  ...props
}, ref) => {
  return (
    <div className={`flex flex-col gap-1.5 ${containerClassName}`}>
      {label && (
        <label htmlFor={id} className="text-[11px] font-bold uppercase tracking-widest text-slate-500 ml-1">
          {label}
        </label>
      )}
      <div className="relative">
        <select
          ref={ref}
          id={id}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className={`
            w-full appearance-none rounded-2xl border bg-white px-4 py-3 text-sm text-slate-900 outline-none transition-all
            ${error 
              ? "border-rose-300 focus:border-rose-500 focus:ring-4 focus:ring-rose-500/10" 
              : "border-slate-200 focus:border-[#0078D4] focus:ring-4 focus:ring-blue-500/10 hover:border-slate-300"}
            disabled:bg-slate-50 disabled:text-slate-400 disabled:cursor-not-allowed
            ${className}
          `}
          {...props}
        >
          {options.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
        <div className="pointer-events-none absolute right-4 top-1/2 -translate-y-1/2 text-slate-400">
          <ChevronDown size={18} />
        </div>
      </div>
      {error && (
        <p className="text-[10px] font-bold text-rose-500 ml-1 uppercase tracking-wide">
          {error}
        </p>
      )}
    </div>
  );
});

Select.displayName = "Select";
