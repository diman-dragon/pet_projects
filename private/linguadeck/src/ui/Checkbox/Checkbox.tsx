import React, { forwardRef } from "react";
import { Check } from "lucide-react";

interface CheckboxProps extends Omit<React.InputHTMLAttributes<HTMLInputElement>, "type"> {
  label?: string;
  error?: string;
}

export const Checkbox = forwardRef<HTMLInputElement, CheckboxProps>(({
  label,
  error,
  className = "",
  id,
  checked,
  ...props
}, ref) => {
  return (
    <div className="flex flex-col gap-1.5">
      <label className={`group flex cursor-pointer items-center gap-3 select-none ${className}`}>
        <div className="relative flex h-5 w-5 items-center justify-center">
          <input
            ref={ref}
            type="checkbox"
            id={id}
            checked={checked}
            className="peer absolute h-full w-full opacity-0 cursor-pointer"
            {...props}
          />
          <div className={`
            flex h-full w-full items-center justify-center rounded-lg border-2 transition-all duration-200
            ${checked 
              ? "bg-[#0078D4] border-[#0078D4] shadow-sm shadow-blue-200" 
              : "bg-white border-slate-200 group-hover:border-slate-300"}
          `}>
            {checked && <Check size={14} className="text-white" strokeWidth={4} />}
          </div>
        </div>
        {label && (
          <span className="text-sm font-medium text-slate-700">
            {label}
          </span>
        )}
      </label>
      {error && (
        <p className="text-[10px] font-bold text-rose-500 ml-8 uppercase tracking-wide">
          {error}
        </p>
      )}
    </div>
  );
});

Checkbox.displayName = "Checkbox";
