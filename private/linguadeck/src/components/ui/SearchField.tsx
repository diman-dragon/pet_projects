import { Input } from "../../ui/Input";

interface SearchFieldProps {
  value: string;
  placeholder?: string;
  onChange: (value: string) => void;
  className?: string;
}

export function SearchField({ value, placeholder = "Поиск...", onChange, className = "" }: SearchFieldProps) {
  return (
    <Input
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder={placeholder}
      className={className}
      type="search"
    />
  );
}
