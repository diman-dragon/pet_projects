import { ReactNode } from "react";

interface CardProps {
  children: ReactNode;
  className?: string;
  onClick?: () => void;
  noPadding?: boolean;
}

export const Card = ({ children, className = "", onClick, noPadding = false }: CardProps) => {
  return (
    <div
      onClick={onClick}
      className={`
        bg-white rounded-3xl border border-slate-200 shadow-sm overflow-hidden
        ${onClick ? "cursor-pointer transition-all hover:border-slate-300 hover:shadow-md active:scale-[0.99]" : ""}
        ${noPadding ? "" : "p-6"}
        ${className}
      `}
    >
      {children}
    </div>
  );
};
