// src/components/ui/data/UserAvatar.tsx
import { Building2, User } from 'lucide-react';

type AvatarType = 'legal' | 'individual';
type AvatarSize = 'sm' | 'md' | 'lg';

interface UserAvatarProps {
  name?: string;
  type?: AvatarType;
  size?: AvatarSize;
  className?: string;
}

// Win11 Style: Большие скругления (squircle) и четкая сетка размеров
const SIZE: Record<AvatarSize, { box: string; icon: number; text: string }> = {
  sm: { box: 'w-8 h-8 rounded-lg', icon: 14, text: 'text-[length:var(--text-xs)]' },
  md: { box: 'w-10 h-10 rounded-xl', icon: 18, text: 'text-[length:var(--text-sm)]' },
  lg: { box: 'w-14 h-14 rounded-2xl', icon: 24, text: 'text-[length:var(--text-md)]' },
};

// Более насыщенные системные градиенты Fluent
const TYPE_STYLE: Record<AvatarType, string> = {
  legal: 'bg-[var(--app-info-soft)] border-[var(--app-info-border)] text-[var(--app-accent)] shadow-sm shadow-[color-mix(in_srgb,var(--app-accent)_15%,transparent)]',
  individual: 'bg-[var(--app-info-soft)] border-[var(--app-info-border)] text-[var(--app-accent)] shadow-sm shadow-[color-mix(in_srgb,var(--app-accent)_15%,transparent)]',
};

function getInitials(name: string): string {
  const parts = name.split(' ').filter(Boolean);
  if (parts.length === 0) return '?';

  // Если одно слово (название компании), берем две первые буквы
  if (parts.length === 1) return parts[0].substring(0, 2).toUpperCase();

  // Если имя и фамилия — классика
  return (parts[0][0] + parts[1][0]).toUpperCase();
}

export function UserAvatar({
  name,
  type = 'individual',
  size = 'md',
  className = '',
}: UserAvatarProps) {
  const { box, icon, text } = SIZE[size];
  const style = TYPE_STYLE[type];

  // Контент: Инициалы или иконка
  const content = name ? (
    <span
      className={`${text} font-bold leading-[var(--leading-heading)] tracking-tight select-none`}
    >
      {getInitials(name)}
    </span>
  ) : type === 'legal' ? (
    <Building2 size={icon} strokeWidth={2.2} />
  ) : (
    <User size={icon} strokeWidth={2.2} />
  );

  return (
    <div
      className={[
        'shrink-0 flex items-center justify-center border transition-transform duration-200 active:scale-95',
        box,
        style,
        className,
      ]
        .filter(Boolean)
        .join(' ')}
      title={name}
    >
      {content}
    </div>
  );
}
