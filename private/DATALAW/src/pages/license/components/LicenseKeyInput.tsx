// src/pages/license/components/LicenseKeyInput.tsx
//
// Поле ввода лицензионного ключа.
// Поддерживает Enter для отправки, автофокус, форматирование в uppercase.

interface LicenseKeyInputProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  disabled?: boolean;
}

export function LicenseKeyInput({ value, onChange, onSubmit, disabled }: LicenseKeyInputProps) {
  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !disabled) onSubmit();
  };

  return (
    <label className="block">
      <div className="text-[9px] font-black uppercase tracking-[0.25em] text-black/30 mb-2">
        License Key
      </div>
      <input
        value={value}
        onChange={(e) => onChange(e.target.value.replace(/\s+/g, '').trim())}
        onKeyDown={handleKeyDown}
        placeholder="DL1-..."
        autoFocus
        spellCheck={false}
        autoComplete="off"
        disabled={disabled}
        className={[
          'w-full rounded border border-black/[0.1] border-b-black/[0.2]',
          'bg-white/80 px-3 py-2.5',
          'font-mono text-[12px] font-semibold text-black/80',
          'outline-none transition-all',
          'focus:border-[var(--app-accent)]/40 focus:border-b-[#0067c0] focus:bg-white focus:ring-2 focus:ring-[#0067c0]/10',
          'disabled:opacity-50 disabled:pointer-events-none',
          'shadow-[0_1px_2px_rgba(0,0,0,0.04)]',
        ].join(' ')}
      />
    </label>
  );
}
