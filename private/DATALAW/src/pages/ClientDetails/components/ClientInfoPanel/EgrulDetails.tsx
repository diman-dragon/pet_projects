import {
  FileText,
  Download,
  ShieldCheck,
  Calendar,
  History,
  Mail,
  MapPin,
  Briefcase,
  Award,
  Copy,
  CheckCircle2
} from 'lucide-react';
import { Badge, UiText, Button } from '@/components/ui';
import { useI18n } from '@/i18n/AppLanguageProvider';

interface EgrulProvider {
  full_name?: string;
  short_name?: string;
  ogrn?: string;
  inn?: string;
  kpp?: string;
  registration_date?: string;
  registration_date_pre2002?: string;
  registration_address?: string;
  director?: string;
  email?: string;
  status?: string;
  liquidation_date?: string | null;
  region?: string;
  okved?: Array<{
    code: string;
    description: string;
    primary: boolean;
  }>;
  responsible_parties?: Array<{
    name: string;
    position: string;
  }>;
  pdf_path?: string;
  error?: string;
}

interface EgrulDetailsProps {
  data: EgrulProvider;
  currentEmail?: string | null;
  onSyncEmail?: (email: string) => void;
}

export function EgrulDetails({ data, currentEmail, onSyncEmail }: EgrulDetailsProps) {
  const { t } = useI18n();
  const formatDate = (dateStr?: string | null) => {
    if (!dateStr) return '—';
    try {
      return new Date(dateStr).toLocaleDateString('ru-RU', {
        day: '2-digit',
        month: 'long',
        year: 'numeric',
      });
    } catch {
      return dateStr;
    }
  };

  const handleCopy = (text?: string) => {
    if (!text) return;
    navigator.clipboard.writeText(text);
  };

  const isNewEmail = data.email && data.email.toLowerCase() !== currentEmail?.toLowerCase();

  return (
    <div className="space-y-8 animate-in fade-in duration-500">
      {/* Auto-Dossier Header Branding */}
      <div className="p-6 rounded-2xl bg-[var(--app-info-soft)] border border-[var(--app-info-border)] relative overflow-hidden">
        <div className="absolute top-0 right-0 p-8 opacity-5">
           <ShieldCheck size={120} />
        </div>
        <div className="flex flex-col gap-4 relative z-10">
          <div className="flex items-center gap-2">
            <Badge label={t('dossier.header.source' as any)} variant="blue" className="px-3" />
            <UiText size="xs" weight="black" className="uppercase tracking-widest text-[var(--app-accent)]">{t('dossier.header.title' as any)}</UiText>
          </div>
          <div className="space-y-1">
             <UiText size="xl" weight="black" className="leading-tight">
               {data.short_name || data.full_name?.split('"')[1] || data.full_name}
             </UiText>
             <div className="flex items-center gap-4 text-[var(--app-text-secondary)]">
               <div className="flex items-center gap-1.5">
                 <UiText size="xs" weight="bold">{t('dossier.label.inn' as any)}:</UiText>
                 <UiText size="xs" className="font-mono">{data.inn}</UiText>
               </div>
               <div className="flex items-center gap-1.5">
                 <UiText size="xs" weight="bold">{t('dossier.label.ogrn' as any)}:</UiText>
                 <UiText size="xs" className="font-mono">{data.ogrn}</UiText>
               </div>
               {data.region && (
                 <div className="flex items-center gap-1.5">
                   <UiText size="xs" weight="bold">{t('dossier.label.region' as any)}:</UiText>
                   <UiText size="xs">{data.region}</UiText>
                 </div>
               )}
             </div>
          </div>
        </div>
      </div>

      {/* Primary Details Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <div className="space-y-6">
          <div className="flex flex-col gap-1">
            <UiText size="xs" weight="black" tone="muted" className="uppercase tracking-widest">
              {t('dossier.label.fullName' as any)}
            </UiText>
            <div className="group flex items-start gap-2">
              <UiText size="sm" weight="bold" className="leading-tight">
                {data.full_name || t('dossier.placeholder.noData' as any)}
              </UiText>
              <button
                onClick={() => handleCopy(data.full_name)}
                className="mt-0.5 p-1 rounded hover:bg-[var(--app-surface)] text-[var(--app-text-muted)] hover:text-[var(--app-text)] transition-all"
                title={t('dossier.action.copy' as any)}
              >
                <Copy size={12} />
              </button>
            </div>
          </div>

          <div className="flex flex-col gap-1 p-4 rounded-xl border border-[var(--app-border)] bg-[var(--app-bg)]/30">
            <UiText size="xs" weight="black" tone="muted" className="uppercase tracking-widest">
              {t('dossier.label.director' as any)}
            </UiText>
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-[var(--app-surface)] shadow-sm border border-[var(--app-border)] text-[var(--app-accent)]">
                <Award size={16} />
              </div>
              <UiText size="xs" weight="bold">
                {data.director?.includes(':') ? data.director : `${t('dossier.label.director' as any).toUpperCase()}: ${data.director || t('dossier.placeholder.noData' as any)}`}
              </UiText>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4 content-start">
          <div className="relative group space-y-1 p-4 rounded-xl bg-[var(--app-info-soft)] border border-[var(--app-info-border)]">
            <UiText size="xs" weight="black" className="uppercase tracking-widest text-[var(--app-accent)]">
              {t('dossier.label.ogrn' as any)}
            </UiText>
            <div className="flex items-center justify-between">
              <UiText size="sm" weight="bold" className="font-mono">
                {data.ogrn || '—'}
              </UiText>
              <button onClick={() => handleCopy(data.ogrn)} className="opacity-0 group-hover:opacity-100 p-1 text-[var(--app-accent)] transition-opacity">
                 <Copy size={12} />
              </button>
            </div>
          </div>
          <div className="relative group space-y-1 p-4 rounded-xl bg-[var(--app-info-soft)] border border-[var(--app-info-border)]">
            <UiText size="xs" weight="black" className="uppercase tracking-widest text-[var(--app-accent)]">
              {t('dossier.label.inn' as any)}
            </UiText>
            <div className="flex items-center justify-between">
              <UiText size="sm" weight="bold" className="font-mono">
                {data.inn || '—'}
              </UiText>
              <button onClick={() => handleCopy(data.inn)} className="opacity-0 group-hover:opacity-100 p-1 text-[var(--app-accent)] transition-opacity">
                 <Copy size={12} />
              </button>
            </div>
          </div>
          <div className="relative group col-span-2 space-y-1 p-4 rounded-xl border border-[var(--app-border)] bg-[var(--app-surface)]">
            <UiText size="xs" weight="black" tone="muted" className="uppercase tracking-widest">
              {t('dossier.label.kpp' as any)}
            </UiText>
            <div className="flex items-center justify-between">
              <UiText size="sm" weight="bold" className="font-mono">
                {data.kpp || '—'}
              </UiText>
              <button onClick={() => handleCopy(data.kpp)} className="opacity-0 group-hover:opacity-100 p-1 text-[var(--app-text-muted)] transition-opacity">
                 <Copy size={12} />
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* PDF Documentation Block */}
      {data.pdf_path && (
        <div className="p-4 rounded-xl bg-[var(--app-bg)]/30 border border-[var(--app-border)] flex items-center justify-between gap-4">
          <div className="flex items-center gap-3">
             <div className="p-2.5 rounded-lg bg-red-50 text-red-600 border border-red-100">
                <FileText size={20} />
             </div>
             <div className="min-w-0">
                <UiText size="xs" weight="black" tone="muted" className="uppercase tracking-widest">{t('dossier.label.pdfSaved' as any)}</UiText>
                <UiText size="xs" className="truncate block text-[var(--app-text-secondary)] font-mono">
                  {data.pdf_path}
                </UiText>
             </div>
          </div>
          <Button variant="secondary" size="sm" className="shrink-0" onClick={() => window.open(`file://${data.pdf_path}`)}>
             <Download size={14} className="mr-2" />
             {t('dossier.action.open' as any)}
          </Button>
        </div>
      )}

      {/* Dates & Status Strip */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="flex items-center gap-3 p-4 bg-[var(--app-bg)]/50 rounded-xl border border-[var(--app-border)]">
          <Calendar size={18} className="text-[var(--app-text-muted)]" />
          <div>
            <UiText size="xs" weight="black" tone="muted" className="uppercase tracking-widest">{t('dossier.label.registrationDate' as any)}</UiText>
            <UiText size="sm" weight="bold">{formatDate(data.registration_date)}</UiText>
          </div>
        </div>

        {data.registration_date_pre2002 && (
          <div className="flex items-center gap-3 p-4 bg-[var(--app-bg)]/50 rounded-xl border border-[var(--app-border)]">
            <History size={18} className="text-[var(--app-text-muted)]" />
            <div>
              <UiText size="xs" weight="black" tone="muted" className="uppercase tracking-widest text-[8px]">{t('dossier.label.pre2002Date' as any)}</UiText>
              <UiText size="sm" weight="bold">{formatDate(data.registration_date_pre2002)}</UiText>
            </div>
          </div>
        )}

        <div className="flex items-center gap-3 p-4 bg-[var(--app-bg)]/50 rounded-xl border border-[var(--app-border)]">
          <Mail size={18} className={isNewEmail ? 'text-[var(--app-warning)]' : 'text-[var(--app-text-muted)]'} />
          <div className="flex-1 min-w-0">
            <UiText size="xs" weight="black" tone="muted" className="uppercase tracking-widest">{t('dossier.label.email' as any)}</UiText>
            <div className="flex items-center justify-between gap-2">
              <UiText size="xs" weight="bold" className="truncate">
                {data.email || t('dossier.placeholder.noData' as any)}
              </UiText>
              {isNewEmail && onSyncEmail && (
                <button
                  onClick={() => onSyncEmail(data.email!)}
                  className="p-1 px-2 rounded-lg bg-[var(--app-warning-soft)] text-[var(--app-warning)] text-[10px] font-bold border border-[var(--app-warning-border)] hover:brightness-95 transition-all"
                >
                  {t('common.refresh' as any)}
                </button>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Address - Using system panel styles */}
      {data.registration_address && (
        <div className="p-6 rounded-xl border border-[var(--app-border-strong)] bg-[var(--app-text)] text-white shadow-lg overflow-hidden relative group">
          <div className="absolute -right-4 -bottom-4 opacity-5 group-hover:scale-110 transition-transform duration-700">
            <MapPin size={100} />
          </div>
          <div className="relative z-10 flex items-start gap-4">
            <div className="p-3 rounded-lg bg-white/10 backdrop-blur-sm border border-white/10">
              <MapPin size={20} className="text-white/80" />
            </div>
            <div className="space-y-1 flex-1">
              <UiText size="xs" weight="black" className="uppercase tracking-widest text-white/40">
                {t('dossier.label.address' as any)}
              </UiText>
              <UiText size="sm" weight="semibold" className="text-white leading-relaxed">
                {data.registration_address}
              </UiText>
            </div>
            <Button
               variant="secondary"
               size="sm"
               className="bg-white/10 border-white/20 hover:bg-white/20 text-white"
               onClick={() => handleCopy(data.registration_address)}
            >
              {t('dossier.action.copy' as any)}
            </Button>
          </div>
        </div>
      )}

      {/* OKVED & responsible parties */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <div className="space-y-4">
          <div className="flex items-center gap-2">
            <Briefcase size={14} className="text-[var(--app-text-muted)]" />
            <UiText size="xs" weight="black" className="uppercase tracking-widest">{t('dossier.label.okved' as any)}</UiText>
            <Badge label={`${data.okved?.length || 0}`} variant="gray" />
          </div>
          <div className="space-y-2 max-h-[300px] overflow-y-auto pr-2 custom-scroll">
            {data.okved?.sort((a) => a.primary ? -1 : 1).map((activity, idx) => (
              <div
                key={idx}
                className={`p-3 rounded-lg border transition-colors ${activity.primary ? 'border-[var(--app-info-border)] bg-[var(--app-info-soft)]' : 'border-[var(--app-border)] bg-[var(--app-surface)]'}`}
              >
                <div className="flex items-center gap-2 mb-1">
                  <UiText size="xs" weight="black" className={activity.primary ? 'text-[var(--app-accent)]' : 'text-[var(--app-text-muted)]'}>
                    {activity.code}
                  </UiText>
                  {activity.primary && <Badge label={t('dossier.status.primary' as any)} variant="blue" />}
                </div>
                <UiText size="xs" tone="muted" className="leading-snug">
                  {activity.description}
                </UiText>
              </div>
            ))}
          </div>
        </div>

        <div className="space-y-4">
          <div className="flex items-center gap-2">
            <ShieldCheck size={14} className="text-[var(--app-text-muted)]" />
            <UiText size="xs" weight="black" className="uppercase tracking-widest">{t('dossier.label.responsibleParties' as any)}</UiText>
          </div>
          <div className="grid grid-cols-1 gap-3">
            {data.responsible_parties?.map((person, idx) => (
              <div key={idx} className="flex items-center gap-3 p-4 rounded-lg bg-[var(--app-surface)] border border-[var(--app-border)]">
                <div className="size-8 rounded-full bg-[var(--app-bg)] flex items-center justify-center text-[var(--app-text-muted)] font-bold text-xs">
                  {person.name[0]}
                </div>
                <div className="flex-1 min-w-0">
                  <UiText size="xs" weight="bold" className="truncate block">
                    {person.name}
                  </UiText>
                  <UiText size="xs" tone="muted" className="italic">
                    {person.position}
                  </UiText>
                </div>
                {person.position.toLowerCase().includes('директор') && (
                  <CheckCircle2 size={14} className="text-[var(--app-online)]" />
                )}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
