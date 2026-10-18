import { Cpu, Trash2 } from 'lucide-react';
import { useI18n } from '../../../i18n/AppLanguageProvider';
import { useSidecarStatus } from '../../../hooks/useSidecarStatus';

interface SidebarFooterProps {
  onNavigate: (route: string, params?: Record<string, unknown>) => void;
}

export function SidebarFooter({ onNavigate }: SidebarFooterProps) {
  const { t } = useI18n();
  const { status: sidecar } = useSidecarStatus();

  return (
    <div className="mt-auto border-t border-[var(--app-border)] p-3">
      <div className="mb-3 space-y-1">
        <div className="group flex w-full items-center gap-2.5 rounded-lg px-2.5 py-2 text-left text-[var(--app-text-secondary)]">
          <div
            className={`flex size-5 items-center justify-center rounded-md border ${sidecar.running ? 'border-[var(--app-success-border)] bg-[var(--app-success-soft)] text-[var(--app-success)]' : 'border-[var(--app-border)] bg-[var(--app-surface-soft)] text-[var(--app-text-muted)]'}`}
          >
            <Cpu size={12} strokeWidth={2.5} />
          </div>
          <div className="flex flex-1 flex-col leading-tight">
            <span className="text-[11px] font-semibold uppercase tracking-wider opacity-50">
              Ollama AI
            </span>
            <span className="text-[10px] font-medium">
              {sidecar.running
                ? t('laila.footer.systemActive')
                : t('laila.footer.localmindOffline')}
            </span>
          </div>
          {sidecar.running && (
            <div className="size-1.5 animate-pulse rounded-full bg-[var(--app-online)] shadow-[0_0_8px_color-mix(in_srgb,var(--app-online)_60%,transparent)]" />
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 gap-2">
        <button
          type="button"
          onClick={() => onNavigate('trash')}
          className="flex items-center justify-center rounded-lg border border-black/[0.06] bg-black/[0.02] py-2.5 text-black/40 transition-all hover:bg-black/[0.05] hover:text-black/60 active:scale-95"
          title={t('sidebar.menu.trash')}
        >
          <Trash2 size={16} strokeWidth={2.5} />
        </button>
      </div>
    </div>
  );
}
