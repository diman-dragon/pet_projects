import React, { ReactNode, forwardRef } from "react";
import { Loader2 } from "lucide-react";

export type ButtonVariant = 
  | "primary" 
  | "secondary" 
  | "danger" 
  | "ghost" 
  | "soft" 
  | "link";

export type ButtonSize = "xs" | "sm" | "md" | "lg" | "icon";

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  children?: ReactNode;
  variant?: ButtonVariant;
  size?: ButtonSize;
  isLoading?: boolean;
  leftIcon?: ReactNode;
  rightIcon?: ReactNode;
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(({
  children,
  variant = "secondary",
  size = "md",
  isLoading = false,
  leftIcon,
  rightIcon,
  className = "",
  disabled,
  ...props
}, ref) => {
  const baseClasses = "inline-flex items-center justify-center gap-2 rounded-xl font-bold transition-all duration-200 active:scale-[0.97] disabled:opacity-50 disabled:pointer-events-none select-none";
  
  const variantClasses: Record<ButtonVariant, string> = {
    primary: "bg-[#0078D4] text-white shadow-lg shadow-blue-200/50 hover:bg-[#005A9E] hover:shadow-blue-300/50",
    secondary: "bg-white text-slate-700 border border-slate-200 hover:border-slate-300 hover:bg-slate-50 shadow-sm",
    danger: "bg-rose-500 text-white shadow-lg shadow-rose-200/50 hover:bg-rose-600 hover:shadow-rose-300/50",
    soft: "bg-blue-50 text-[#0078D4] hover:bg-blue-100",
    ghost: "bg-transparent text-slate-600 hover:bg-slate-100",
    link: "bg-transparent text-[#0078D4] hover:underline p-0 h-auto"
  };

  const sizeClasses: Record<ButtonSize, string> = {
    xs: "px-2.5 py-1 text-[10px] uppercase tracking-wider",
    sm: "px-4 py-2 text-xs",
    md: "px-6 py-2.5 text-sm",
    lg: "px-8 py-3.5 text-base",
    icon: "p-2.5"
  };

  const isOnlyIcon = !children && (leftIcon || rightIcon || isLoading);

  return (
    <button
      ref={ref}
      className={`${baseClasses} ${variantClasses[variant]} ${isOnlyIcon ? sizeClasses.icon : sizeClasses[size]} ${className}`}
      disabled={disabled || isLoading}
      {...props}
    >
      {isLoading ? (
        <Loader2 className="animate-spin" size={size === "xs" ? 12 : 16} />
      ) : (
        <>
          {leftIcon && <span className="flex shrink-0">{leftIcon}</span>}
          {children}
          {rightIcon && <span className="flex shrink-0">{rightIcon}</span>}
        </>
      )}
    </button>
  );
});

Button.displayName = "Button";
