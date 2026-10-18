import { useI18n } from '../../../../i18n/AppLanguageProvider';

/** Анимация ожидания ответа в компактном виджете */
export function LailaThinkingAnimation() {
  const { t } = useI18n();

  return (
    <div className="flex justify-start">
      <div className="flex items-center gap-3 rounded-2xl rounded-tl-none border border-[var(--app-border)] bg-[var(--app-surface-soft)] px-4 py-3.5">
        <div className="flex gap-1.5">
          {[0, 150, 300].map((delay) => (
            <div
              key={delay}
              className="h-2 w-2 animate-bounce rounded-full bg-[color-mix(in_srgb,var(--app-text)_40%,transparent)]"
              style={{ animationDelay: `${delay}ms` }}
            />
          ))}
        </div>
        <span className="text-[length:var(--text-sm)] leading-[var(--leading-body)] text-[var(--app-text-secondary)]">
          {t('laila.thinking')}
        </span>
      </div>
    </div>
  );
}
