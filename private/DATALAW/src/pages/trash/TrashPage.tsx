import { useEffect, useState } from 'react';
import { RefreshCw, RotateCcw, Trash2 } from 'lucide-react';
import { Button } from '../../components/ui/form/Button';
import { UiPanel } from '../../components/ui/atoms/UiPanel';
import { UiText } from '../../components/ui/atoms/UiText';
import { ConfirmModal } from '../../components/ui/overlays/ConfirmModal';
import { useI18n } from '../../i18n/AppLanguageProvider';
import { PageBase } from '../../shared/ui/PageBase';
import { businessApi, type TrashItem } from '@/services/backend/businessApi';
import { hasPermission } from '../../services/auth/permissions';
import type { UserProfile } from '../../types/entities';

interface TrashPageProps {
  currentUser: UserProfile | null;
}

interface PurgeConfirm {
  item: TrashItem | null;
  isLoading: boolean;
}

export default function TrashPage({ currentUser }: TrashPageProps) {
  const { language, t } = useI18n();
  const [items, setItems] = useState<TrashItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [busyKey, setBusyKey] = useState<string | null>(null);
  const [purgeConfirm, setPurgeConfirm] = useState<PurgeConfirm>({ item: null, isLoading: false });

  const load = async () => {
    setLoading(true);
    try {
      setItems(await businessApi.listTrashItems());
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void load();
  }, []);

  const dateLocale = language === 'sr' ? 'sr-Latn-RS' : language === 'en' ? 'en-US' : 'ru-RU';
  const canPurge = currentUser && hasPermission(currentUser, 'trash:purge');

  const handleRestore = async (item: TrashItem) => {
    const key = `restore:${item.entity_type}:${item.entity_id}`;
    setBusyKey(key);
    try {
      await businessApi.restoreTrashItem(item.entity_type, item.entity_id);
      await load();
    } finally {
      setBusyKey(null);
    }
  };

  const confirmPurge = async () => {
    if (!purgeConfirm.item) return;
    setPurgeConfirm((prev) => ({ ...prev, isLoading: true }));
    try {
      setBusyKey(`purge:${purgeConfirm.item.entity_type}:${purgeConfirm.item.entity_id}`);
      await businessApi.purgeTrashItem(purgeConfirm.item.entity_type, purgeConfirm.item.entity_id);
      await load();
    } finally {
      setBusyKey(null);
      setPurgeConfirm({ item: null, isLoading: false });
    }
  };

  return (
    <PageBase
      title={t('trash.section')}
      subtitle={t('trash.subtitle')}
      icon={<Trash2 size={18} />}
      actions={<Button variant="secondary" size="sm" icon={<RefreshCw size={14} />} onClick={() => void load()} disabled={loading}>{t('trash.refresh')}</Button>}
    >
      <UiPanel padding="none" mica elevation="sm" className="overflow-hidden rounded-[24px]">
        <div className="section-divider flex items-center justify-between">
          <UiText as="p" size="xs" weight="bold" tone="muted" className="label-caps">
            {t('trash.count', { count: items.length })}
          </UiText>
        </div>
        {loading ? (
          <div className="space-y-3 px-6 py-6">
            {[0, 1, 2, 3].map((item) => <div key={item} className="h-20 animate-pulse rounded-[18px] bg-[var(--app-surface-soft)]" />)}
          </div>
        ) : items.length === 0 ? (
          <div className="px-6 py-20 text-center">
            <div className="mx-auto flex size-16 items-center justify-center rounded-full bg-[var(--app-surface-soft)] text-[var(--app-text-muted)]">
              <Trash2 size={32} strokeWidth={1.5} />
            </div>
            <UiText as="p" size="md" tone="secondary" className="mt-4">
              {t('trash.empty')}
            </UiText>
          </div>
        ) : (
          <div className="divide-y divide-[var(--app-border)]">
            {items.map((item) => (
              <div key={`${item.entity_type}:${item.entity_id}`} className="grid gap-4 px-6 py-5 lg:grid-cols-[140px_minmax(0,1fr)_200px_230px]">
                <div>
                  <UiText as="p" size="xs" weight="bold" tone="muted" className="label-caps">{t('trash.type')}</UiText>
                  <div className="mt-2 inline-flex rounded-full bg-[var(--app-surface-soft)] px-2.5 py-1 text-[length:var(--text-xs)] font-semibold text-[var(--app-text-secondary)]">{item.entity_type}</div>
                </div>
                <div>
                  <UiText as="p" size="xs" weight="bold" tone="muted" className="label-caps">{t('trash.name')}</UiText>
                  <UiText as="p" size="sm" weight="semibold" className="mt-2">{item.entity_label}</UiText>
                </div>
                <div>
                  <UiText as="p" size="xs" weight="bold" tone="muted" className="label-caps">{t('trash.deletedAt')}</UiText>
                  <UiText as="p" size="xs" tone="secondary" className="mt-2">{new Date(item.deleted_at).toLocaleString(dateLocale)}</UiText>
                </div>
                <div className="flex flex-wrap items-end justify-end gap-2">
                  <Button
                    variant="secondary"
                    size="sm"
                    icon={<RotateCcw size={13} />}
                    onClick={() => void handleRestore(item)}
                    loading={busyKey === `restore:${item.entity_type}:${item.entity_id}`}
                    disabled={busyKey !== null}
                  >
                    {t('trash.restore')}
                  </Button>
                  {canPurge ? (
                    <Button
                      variant="danger"
                      size="sm"
                      icon={<Trash2 size={13} />}
                      onClick={() => setPurgeConfirm({ item, isLoading: false })}
                      loading={busyKey === `purge:${item.entity_type}:${item.entity_id}`}
                      disabled={busyKey !== null}
                    >
                      {t('trash.purge')}
                    </Button>
                  ) : null}
                </div>
              </div>
            ))}
          </div>
        )}
        <div className="border-t border-[var(--app-border)] bg-[var(--app-surface-soft)] px-6 py-5">
          <UiText as="p" size="xs" tone="secondary" className="max-w-3xl">
            {t('trash.permanentHint')}
          </UiText>
        </div>
      </UiPanel>
      <ConfirmModal
        open={!!purgeConfirm.item}
        title={t('trash.purge')}
        message={`${t('trash.purge')}: ${purgeConfirm.item?.entity_label}?`}
        confirmText={t('trash.purge')}
        cancelText={t('common.cancel')}
        isDangerous
        isLoading={purgeConfirm.isLoading}
        onConfirm={confirmPurge}
        onCancel={() => setPurgeConfirm({ item: null, isLoading: false })}
      />
    </PageBase>
  );
}
