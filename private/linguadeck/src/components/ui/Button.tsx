import { Button as BaseButton } from "../../ui/Button";

interface ButtonProps {
  children: React.ReactNode;
  onClick?: () => void;
  type?: "button" | "submit" | "reset";
  disabled?: boolean;
  variant?: "primary" | "secondary" | "danger" | "danger-soft";
  className?: string;
  ariaLabel?: string;
}

export function Button({
  children,
  onClick,
  type = "button",
  disabled,
  variant,
  className,
  ariaLabel
}: ButtonProps) {
  // Map internal variant to official variants
  const mapVariant = variant === "danger-soft" ? "soft" : (variant === "danger" ? "danger" : (variant === "primary" ? "primary" : "secondary"));
  
  return (
    <BaseButton
      aria-label={ariaLabel}
      className={className}
      disabled={disabled}
      onClick={onClick}
      type={type}
      variant={mapVariant}
    >
      {children}
    </BaseButton>
  );
}
