import { Select } from "../../ui/Select";

interface SelectFieldProps {
  label: string;
  value: string;
  onChange: (value: string) => void;
  options: Array<{ value: string; label: string }>;
}

export function SelectField({ label, value, onChange, options }: SelectFieldProps) {
  return (
    <Select
      label={label}
      value={value}
      onChange={onChange}
      options={options}
    />
  );
}
