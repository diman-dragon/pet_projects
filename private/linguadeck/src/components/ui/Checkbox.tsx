import { Checkbox as BaseCheckbox } from "../../ui/Checkbox";

interface CheckboxProps {
  checked: boolean;
  onChange: () => void;
  label?: string;
}

export function Checkbox({ checked, onChange, label }: CheckboxProps) {
  return (
    <BaseCheckbox
      checked={checked}
      onChange={onChange}
      label={label}
    />
  );
}
