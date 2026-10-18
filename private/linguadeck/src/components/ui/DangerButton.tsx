import { Trash2 } from "lucide-react";
import { Button } from "./Button";

interface DangerButtonProps {
  label?: string;
  onClick?: () => void;
  disabled?: boolean;
  className?: string;
}

export function DangerButton({
  label = "Удалить",
  onClick,
  disabled,
  className
}: DangerButtonProps) {
  return (
    <Button
      className={className}
      disabled={disabled}
      onClick={onClick}
      variant="danger"
    >
      <Trash2 size={16} />
      {label}
    </Button>
  );
}
