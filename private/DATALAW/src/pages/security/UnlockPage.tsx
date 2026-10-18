import { FormEvent, useState } from 'react';
import { LockKeyhole, Loader2, ShieldCheck } from 'lucide-react';
import { DorlaneLogo } from '../../components/common/DorlaneLogo';
import { PublicLanguageSwitcher } from '../../components/auth/PublicLanguageSwitcher';
import { Button, UiText } from '../../components/ui';
import { AppInitializer } from '../../services/init/AppInitializer';

interface UnlockPageProps {
  onUnlocked: (nextStatus: 'welcome' | 'license') => void;
}

export default function UnlockPage({ onUnlocked }: UnlockPageProps) {
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!password.trim()) {
      setError('Введите мастер-пароль.');
      return;
    }

    try {
      setError(null);
      setIsSubmitting(true);
      const result = await AppInitializer.unlock(password);
      if (result.status === 'ready') {
        onUnlocked('welcome');
        return;
      }
      if (result.status === 'license') {
        onUnlocked('license');
        return;
      }
      setError(result.error ?? 'Не удалось разблокировать рабочее пространство.');
    } catch (err: any) {
      setError(err?.message ?? 'Не удалось разблокировать рабочее пространство.');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 z-[1000] flex select-none items-center justify-center overflow-y-auto bg-[var(--app-modal-scrim)] p-6 font-[var(--font-ui)]">
      <div className="pointer-events-none absolute inset-0 backdrop-blur-xl" aria-hidden />
      <form
        onSubmit={handleSubmit}
        className="relative z-10 flex w-full max-w-[440px] flex-col gap-5 rounded-[var(--app-radius-lg)] border border-[var(--app-border)] bg-[var(--app-panel-bg)] p-6 shadow-[var(--app-shadow-md)] backdrop-blur-xl"
      >
        <div className="flex flex-col items-center gap-4 text-center">
          <DorlaneLogo size="lg" showText className="gap-4" />
          <PublicLanguageSwitcher />
          <div className="flex size-14 items-center justify-center rounded-full bg-[var(--app-accent)]/10 text-[var(--app-accent)]">
            <LockKeyhole size={26} />
          </div>
          <div>
            <UiText as="h1" size="md" weight="semibold" heading>
              Разблокировка Dorlane
            </UiText>
            <UiText as="p" size="xs" tone="secondary" className="mt-1">
              Введите мастер-пароль, чтобы открыть защищенные базы и файлы.
            </UiText>
          </div>
        </div>

        <label className="flex flex-col gap-1.5">
          <UiText as="span" size="xs" weight="semibold" tone="muted">
            Мастер-пароль
          </UiText>
          <input
            autoFocus
            type="password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            className="h-11 rounded-lg border border-[var(--app-border)] bg-[var(--app-surface-raised)] px-3 text-[length:var(--text-sm)] outline-none transition-colors focus:border-[var(--app-accent)]"
            autoComplete="current-password"
          />
        </label>

        <div className="flex items-center gap-2 rounded-lg border border-[var(--app-border)] bg-[var(--app-surface-soft)] px-3 py-2">
          <ShieldCheck size={15} className="text-[var(--app-success)]" />
          <UiText as="p" size="xs" tone="muted">
            Пароль не сохраняется в localStorage или на диске.
          </UiText>
        </div>

        {error && (
          <div className="rounded-lg border border-[var(--app-danger)]/20 bg-[var(--app-danger)]/8 px-3 py-2 text-[length:var(--text-xs)] font-semibold text-[var(--app-danger)]">
            {error}
          </div>
        )}

        <Button
          type="submit"
          variant="primary"
          size="md"
          disabled={isSubmitting}
          icon={isSubmitting ? <Loader2 size={14} className="animate-spin" /> : <LockKeyhole size={14} />}
        >
          Разблокировать
        </Button>
      </form>
    </div>
  );
}
