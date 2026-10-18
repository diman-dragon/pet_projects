import { useEffect, useMemo, useState } from 'react';
import {
  RefreshCw,
  Globe,
  Building2,
  Calendar,
  Zap,
  Search,
  CheckCircle2,
  ShieldAlert,
} from 'lucide-react';

import { AISummaryBlock } from '@/components/ui/data/AISummaryBlock';
import { Button, UiPanel, UiText, AlertModal, Badge, Modal } from '@/components/ui';
import { useI18n } from '@/i18n/AppLanguageProvider';
import { businessApi } from '@/services/backend/businessApi';
import type { Client } from '@/types/entities';

import { useClientDossier } from './ClientInfoPanel/useClientDossier';
import { EgrulDetails } from './ClientInfoPanel/EgrulDetails';
import { KadArbitrDetails } from './ClientInfoPanel/KadArbitrDetails';

interface ClientInfoPanelProps {
  client: Client;
  clientId: number;
  isSummaryGenerating?: boolean;
  onGenerateSummary?: () => Promise<void>;
  isRiskScanning?: boolean;
  criticalRisksCount?: number;
  onScanRisks?: () => Promise<void>;
  groupedRisks?: {
    critical: Array<{ description: string; suggestedAction?: string }>;
    warnings: Array<{ description: string; suggestedAction?: string }>;
    ok: string[];
  };
}

export function ClientInfoPanel({
  client,
  clientId,
  isSummaryGenerating,
  onGenerateSummary,
  isRiskScanning,
  criticalRisksCount = 0,
  onScanRisks,
  groupedRisks,
}: ClientInfoPanelProps) {
  const { t } = useI18n();
  const [activePortal, setActivePortal] = useState<'egrul' | 'kad_arbitr' | 'analytics'>('egrul');
  const [currentEmail, setCurrentEmail] = useState(client.email ?? null);
  const [riskModalOpen, setRiskModalOpen] = useState(false);
  const { providers, snapshots, loading, error, success, refreshDossier, setError, setSuccess } =
    useClientDossier(clientId, client.tax_id || undefined);

  useEffect(() => {
    setCurrentEmail(client.email ?? null);
  }, [client.email]);

  const dossier = useMemo(() => {
    if (!snapshots || snapshots.length === 0) return null;
    const latest = snapshots[0];
    try {
      return {
        snapshot: latest,
        manifest: latest.dossier_json ? JSON.parse(latest.dossier_json) : null,
      };
    } catch (e) {
      console.error('Failed to parse dossier JSON', e);
      return null;
    }
  }, [snapshots]);

  const handleSyncEmail = async (email: string) => {
    try {
      await businessApi.saveClient({
        id: client.id,
        full_name: client.full_name,
        client_type: client.type as any,
        status: client.status as any,
        tax_id: client.tax_id || undefined,
        email,
      });
      setCurrentEmail(email);
      setSuccess(t('dossier.email.synced' as any));
    } catch (err) {
      console.error('Email sync failed', err);
      setError(t('dossier.sync.errorPrefix' as any));
    }
  };

  const egrulData = dossier?.manifest?.providers?.egrul;
  const kadArbitrData = dossier?.manifest?.providers?.kad_arbitr;

  return (
    <div className="flex flex-col gap-8 pb-20">
      <UiPanel mica padding="none" className="overflow-hidden border border-[var(--app-border-strong)]">
        <div className="flex flex-col items-center justify-between gap-6 p-8 md:flex-row">
          <div className="flex items-center gap-6">
            <div className="flex size-16 items-center justify-center rounded-2xl bg-[var(--app-accent)] text-white shadow-xl">
              <Building2 size={32} strokeWidth={1.5} />
            </div>

            <div className="space-y-1">
              <UiText
                size="xl"
                weight="black"
                className="text-[var(--app-text)] tracking-tight leading-none"
              >
                {t('dossier.header.title' as any)}
              </UiText>
              <div className="flex items-center gap-3">
                <Badge
                  label={`${t('dossier.label.inn' as any)} ${client.tax_id || '—'}`}
                  variant="gray"
                />
                {dossier?.snapshot?.created_at && (
                  <UiText
                    size="xs"
                    weight="bold"
                    tone="muted"
                    className="flex items-center gap-1.5 uppercase tracking-widest"
                  >
                    <Calendar size={12} />
                    {t('common.refresh' as any)}:{' '}
                    {new Date(dossier.snapshot.created_at).toLocaleDateString()}
                  </UiText>
                )}
              </div>
            </div>
          </div>

          <Button
            variant="primary"
            size="lg"
            loading={loading}
            onClick={refreshDossier}
            disabled={!client.tax_id}
            className="h-12 rounded-xl px-8 font-bold"
            icon={<RefreshCw size={18} className={loading ? 'animate-spin' : ''} />}
          >
            {t('dossier.action.sync' as any)}
          </Button>
        </div>
      </UiPanel>

      <div className="grid grid-cols-1 items-start gap-8 lg:grid-cols-12">
        <div className="space-y-6 lg:col-span-3">
          <div className="flex flex-col gap-1">
            <NavButton
              active={activePortal === 'egrul'}
              onClick={() => setActivePortal('egrul')}
              label={t('dossier.tab.egrul' as any)}
              icon={<Globe size={18} />}
            />
            <NavButton
              active={activePortal === 'kad_arbitr'}
              onClick={() => setActivePortal('kad_arbitr')}
              label={t('dossier.tab.arbitration' as any)}
              icon={<Search size={18} />}
            />
            <NavButton
              active={activePortal === 'analytics'}
              onClick={() => setActivePortal('analytics')}
              label={t('dossier.tab.analytics' as any)}
              icon={<Zap size={18} />}
            />
          </div>

          <UiPanel mica padding="none" className="overflow-hidden border border-[var(--app-border)]">
            <div className="border-b border-[var(--app-border)] bg-[var(--app-bg)]/30 p-4">
              <UiText
                size="xs"
                weight="black"
                className="uppercase tracking-widest text-[var(--app-accent)]"
              >
                {t('laila.tab.status' as any)}
              </UiText>
            </div>
            <div className="space-y-1 p-2">
              {providers.map((provider) => (
                <div
                  key={provider.id}
                  className="flex items-center justify-between rounded-lg p-3 transition-colors hover:bg-[var(--app-border)]"
                >
                  <div className="flex items-center gap-3">
                    <UiText
                      size="xs"
                      weight="bold"
                      tone={provider.status === 'ready' ? 'default' : 'muted'}
                    >
                      {provider.name}
                    </UiText>
                  </div>
                  {provider.status === 'ready' ? (
                    <CheckCircle2 size={14} className="text-[var(--app-online)]" />
                  ) : provider.status === 'pending' ? (
                    <RefreshCw size={14} className="animate-spin text-[var(--app-accent)]" />
                  ) : (
                    <div className="size-1.5 rounded-full bg-[var(--app-offline)]" />
                  )}
                </div>
              ))}
            </div>
          </UiPanel>
        </div>

        <div className="lg:col-span-9">
          <UiPanel mica padding="lg" className="min-h-[500px] border border-[var(--app-border-strong)]">
            {activePortal === 'egrul' &&
              (egrulData ? (
                <EgrulDetails
                  data={egrulData}
                  currentEmail={currentEmail}
                  onSyncEmail={handleSyncEmail}
                />
              ) : (
                <EmptyState
                  title={t('dossier.empty.noData' as any)}
                  description={t('dossier.empty.hint' as any)}
                />
              ))}

            {activePortal === 'kad_arbitr' && (
              <KadArbitrDetails data={kadArbitrData} clientId={clientId} inn={client.tax_id || undefined} />
            )}

            {activePortal === 'analytics' && (
              <div className="space-y-8">
                <div className="flex flex-wrap items-center gap-3">
                  <Button
                    variant="secondary"
                    size="sm"
                    loading={isRiskScanning}
                    onClick={async () => {
                      await onScanRisks?.();
                      setRiskModalOpen(true);
                    }}
                    icon={<ShieldAlert size={14} />}
                  >
                    Проверить риски
                  </Button>
                  {criticalRisksCount > 0 && (
                    <span className="inline-flex items-center rounded-full bg-[var(--app-danger)] px-2 py-0.5 text-[10px] font-bold text-white">
                      {criticalRisksCount}
                    </span>
                  )}
                </div>
                <AISummaryBlock
                  summary={client.summary || dossier?.snapshot?.summary_text || ''}
                  label={t('dossier.tab.analytics' as any)}
                  loading={isSummaryGenerating || loading}
                  onGenerate={onGenerateSummary || (async () => undefined)}
                />
              </div>
            )}
          </UiPanel>
        </div>
      </div>

      <AlertModal
        open={!!error}
        title={t('common.error' as any)}
        message={error || ''}
        onClose={() => setError(null)}
      />

      <AlertModal
        open={!!success}
        title={t('common.refresh' as any)}
        message={success || ''}
        onClose={() => setSuccess(null)}
      />

      <Modal open={riskModalOpen} onClose={() => setRiskModalOpen(false)} title="Риски по клиенту" size="md">
        <div className="space-y-4">
          <RiskSection title="Критические" items={groupedRisks?.critical ?? []} empty="Критических рисков нет" tone="danger" />
          <RiskSection title="Предупреждения" items={groupedRisks?.warnings ?? []} empty="Предупреждений нет" tone="warning" />
          <RiskSection title="OK" items={(groupedRisks?.ok ?? []).map((text) => ({ description: text }))} empty="Нет подтверждённых OK-пунктов" tone="success" />
        </div>
      </Modal>
    </div>
  );
}

function RiskSection({
  title,
  items,
  empty,
  tone,
}: {
  title: string;
  items: Array<{ description: string; suggestedAction?: string }>;
  empty: string;
  tone: 'danger' | 'warning' | 'success';
}) {
  const toneClass =
    tone === 'danger'
      ? 'border-[var(--app-danger)]/20 bg-[var(--app-danger)]/6'
      : tone === 'warning'
        ? 'border-[var(--app-warning)]/20 bg-[var(--app-warning)]/8'
        : 'border-[var(--app-success)]/20 bg-[var(--app-success)]/8';

  return (
    <div className={`rounded-xl border p-4 ${toneClass}`}>
      <UiText as="p" size="xs" weight="bold" className="mb-2 uppercase tracking-widest">{title}</UiText>
      <div className="space-y-2">
        {items.length === 0 && <UiText size="xs" tone="muted">{empty}</UiText>}
        {items.map((item, index) => (
          <div key={`${title}-${index}`} className="rounded-lg bg-[var(--app-surface)] px-3 py-2">
            <UiText as="p" size="xs">{item.description}</UiText>
            {item.suggestedAction && <UiText as="p" size="xs" tone="muted" className="mt-1">{item.suggestedAction}</UiText>}
          </div>
        ))}
      </div>
    </div>
  );
}

function NavButton({
  active,
  label,
  icon,
  onClick,
}: {
  active: boolean;
  label: string;
  icon: React.ReactNode;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={`flex w-full items-center gap-3 rounded-xl border p-4 text-sm font-bold transition-all duration-200 ${active
        ? 'border-[var(--app-accent)] bg-[var(--app-accent)] text-white shadow-md'
        : 'border-transparent bg-transparent text-[var(--app-text-secondary)] hover:bg-[var(--app-border)]'
        }`}
    >
      <div className={active ? 'text-white' : 'text-[var(--app-text-muted)]'}>{icon}</div>
      {label}
    </button>
  );
}

function EmptyState({ title, description }: { title: string; description: string }) {
  return (
    <div className="flex flex-col items-center justify-center py-24 text-center">
      <div className="mb-4 flex size-16 items-center justify-center rounded-full bg-[var(--app-bg)] text-[var(--app-text-muted)]">
        <Globe size={32} strokeWidth={1} />
      </div>
      <UiText size="md" weight="bold" className="mb-1 text-[var(--app-text)]">
        {title}
      </UiText>
      <UiText size="sm" tone="muted" className="max-w-xs">
        {description}
      </UiText>
    </div>
  );
}
