import React, { ReactNode, forwardRef } from "react";

interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  error?: string;
  leftIcon?: ReactNode;
  rightIcon?: ReactNode;
  containerClassName?: string;
}

export const Input = forwardRef<HTMLInputElement, InputProps>(({
  label,
  error,
  leftIcon,
  rightIcon,
  className = "",
  containerClassName = "",
  id,
  ...props
}, ref) => {
  return (
    <div className={`flex flex-col gap-1.5 ${containerClassName}`}>
      {label && (
        <label htmlFor={id} className="text-[11px] font-bold uppercase tracking-widest text-slate-500 ml-1">
          {label}
        </label>
      )}
      <div className="relative flex items-center">
        {leftIcon && (
          <div className="absolute left-3.5 flex items-center justify-center text-slate-400">
            {leftIcon}
          </div>
        )}
        <input
          ref={ref}
          id={id}
          className={`
            w-full rounded-2xl border bg-white px-4 py-3 text-sm text-slate-900 outline-none transition-all
            ${leftIcon ? "pl-11" : ""}
            ${rightIcon ? "pr-11" : ""}
            ${error 
              ? "border-rose-300 focus:border-rose-500 focus:ring-4 focus:ring-rose-500/10" 
              : "border-slate-200 focus:border-[#0078D4] focus:ring-4 focus:ring-blue-500/10 hover:border-slate-300"}
            disabled:bg-slate-50 disabled:text-slate-400 disabled:cursor-not-allowed
            placeholder:text-slate-400
            ${className}
          `}
          {...props}
        />
        {rightIcon && (
          <div className="absolute right-3.5 flex items-center justify-center text-slate-400">
            {rightIcon}
          </div>
        )}
      </div>
      {error && (
        <p className="text-[10px] font-bold text-rose-500 ml-1 uppercase tracking-wide">
          {error}
        </p>
      )}
    </div>
  );
});

Input.displayName = "Input";
