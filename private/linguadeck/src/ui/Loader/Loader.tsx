import { Loader2 } from "lucide-react";

interface LoaderProps {
  size?: "sm" | "md" | "lg" | "xl";
  className?: string;
  fullPage?: boolean;
}

export const Loader = ({ size = "md", className = "", fullPage = false }: LoaderProps) => {
  const sizeMap = {
    sm: 16,
    md: 24,
    lg: 40,
    xl: 64
  };

  const content = (
    <div className={`flex items-center justify-center ${className}`}>
      <Loader2 
        size={sizeMap[size]} 
        className="animate-spin text-[#0078D4]" 
      />
    </div>
  );

  if (fullPage) {
    return (
      <div className="fixed inset-0 z-[200] flex items-center justify-center bg-white/60 backdrop-blur-sm">
        {content}
      </div>
    );
  }

  return content;
};
