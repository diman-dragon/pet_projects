import type { ButtonHTMLAttributes, MouseEvent, ReactNode } from 'react';

export type IconButtonVariant = 'default' | 'danger';

export type IconButtonProps = Omit<ButtonHTMLAttributes<HTMLButtonElement>, 'onClick'> & {
  onClick: (e: MouseEvent<HTMLButtonElement>) => void;
  variant?: IconButtonVariant;
  /** Для кнопок внутри строк таблицы — не даём событию всплывать */
  stopPropagation?: boolean;
  children: ReactNode;
  className?: string;
};

export function IconButton({
  onClick,
  variant = 'default',
  stopPropagation = false,
  children,
  className = '',
  type = 'button',
  ...rest
}: IconButtonProps) {
  const handleClick = (e: MouseEvent<HTMLButtonElement>) => {
    if (stopPropagation) e.stopPropagation();
    if (!rest.disabled) onClick(e);
  };

  return (
    <button
      type={type}
      onClick={handleClick}
      className={[
        'group relative inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-md transition-all duration-200',
        'disabled:cursor-not-allowed disabled:opacity-20 disabled:grayscale',
        variant === 'danger'
          ? 'text-[var(--app-text-muted)] hover:bg-[color-mix(in_srgb,var(--app-danger)_12%,transparent)] hover:text-[var(--app-danger)]'
          : 'text-[var(--app-text-muted)] hover:bg-[color-mix(in_srgb,var(--app-text)_6%,transparent)] hover:text-[var(--app-text)]',
        'active:scale-90',
        className,
      ].join(' ')}
      {...rest}
    >
      <span className="relative z-10 transition-transform duration-200 group-hover:scale-110">
        {children}
      </span>
    </button>
  );
}
