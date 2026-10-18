import LogoImg from '../../assets/image.webp';
import { useI18n } from '../../i18n/AppLanguageProvider';

interface DorlaneLogoProps {
  size?: 'sm' | 'md' | 'lg';
  showText?: boolean;
  className?: string;
}

const SIZE_MAP = {
  sm: {
    img: 'size-9',
    glow: 'size-14',
    title: 'text-[length:var(--text-sm)]',
    sub: 'text-[length:var(--text-xs)]',
  },
  md: {
    img: 'size-16',
    glow: 'size-24',
    title: 'text-[length:var(--text-xl)]',
    sub: 'text-[length:var(--text-xs)]',
  },
  lg: {
    img: 'size-28',
    glow: 'size-40',
    title: 'text-[length:var(--text-hero)]',
    sub: 'text-[length:var(--text-xs)]',
  },
};

export function DorlaneLogo({ size = 'md', showText = true, className = '' }: DorlaneLogoProps) {
  const { t } = useI18n();
  const s = SIZE_MAP[size];

  return (
    <div className={`flex flex-col items-center gap-3 ${className}`}>
      <div className="relative flex items-center justify-center">
        <div
          className={`absolute ${s.glow} rounded-full opacity-30 blur-2xl`}
          style={{
            background:
              'radial-gradient(circle, rgba(99,102,241,0.6) 0%, rgba(234,88,12,0.4) 50%, transparent 100%)',
          }}
        />
        <div
          className="absolute size-8 rounded-full opacity-60 blur-md"
          style={{
            background: 'radial-gradient(circle, rgba(251,191,36,0.8) 0%, transparent 70%)',
            top: '10%',
            left: '50%',
            transform: 'translateX(-50%)',
          }}
        />
        <img
          src={LogoImg}
          alt={t('common.logoAlt')}
          draggable={false}
          className={`${s.img} relative z-10 select-none object-contain drop-shadow-[0_4px_16px_rgba(0,0,0,0.25)]`}
          style={{ transform: 'scale(1.15)' }}
        />
      </div>

      {showText && (
        <div className="flex flex-col items-center leading-tight">
          <span
            className={`${s.title} font-black uppercase tracking-[0.18em] text-[var(--app-text)]`}
          >
            Dorlane
          </span>
          <span
            className={`${s.sub} mt-0.5 font-bold uppercase tracking-[0.3em] text-[var(--app-text-muted)]`}
          >
            {t('common.brandTagline')}
          </span>
        </div>
      )}
    </div>
  );
}
