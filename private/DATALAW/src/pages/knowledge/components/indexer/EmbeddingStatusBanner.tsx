import { Button } from '../../../../components/ui/form/Button';
import type { EmbeddingBannerState } from './useEmbeddingBanner';

export function EmbeddingStatusBanner({
  banner,
  busy,
}: {
  banner: EmbeddingBannerState;
  busy: boolean;
}) {
  const styles =
    banner.tone === 'success'
      ? 'border-[var(--app-success-border)] bg-[var(--app-success-soft)] text-[var(--app-success)]'
      : banner.tone === 'warning'
        ? 'border-[var(--app-warning-border)] bg-[var(--app-warning-soft)] text-[var(--app-warning)]'
        : 'border-[var(--app-info-border)] bg-[var(--app-info-soft)] text-[var(--app-info)]';
  const icon = banner.tone === 'success' ? 'OK' : banner.tone === 'warning' ? '!' : 'i';
  const action = 'actionLabel' in banner ? banner.onAction : undefined;

  return (
    <div
      className={`mt-3 flex items-center justify-between gap-3 rounded-[var(--app-radius-md)] border px-3 py-2 text-[length:var(--text-xs)] ${styles}`}
    >
      <span>
        {icon} {banner.text}
      </span>
      {'actionLabel' in banner && action ? (
        <Button variant="ghost" size="xs" disabled={busy} onClick={() => void action()}>
          {busy ? '...' : banner.actionLabel}
        </Button>
      ) : null}
    </div>
  );
}
