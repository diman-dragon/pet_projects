import { useMemo } from 'react';
import { Case } from '../../../types/entities';
import { CasesTable } from './CasesTable';
import { ContractsTable } from './ContractsTable';
import { OtherCasesTable } from './OtherCasesTable';
import { ConsultingCasesTable } from './ConsultingCasesTable';
import { Select } from '../../../components/ui/form/Select';
import { RegistryToolbar } from '../../../shared/registry/RegistryToolbar';
import { UiText } from '../../../components/ui/atoms/UiText';
import { useI18n } from '../../../i18n/AppLanguageProvider';

interface Props {
  cases: Case[];
  onEdit: (c: Case) => void;
  onDelete: (c: Case) => void;
  onNavigate: (c: Case) => void;
  searchQuery: string;
  setSearchQuery: (val: string) => void;
  filterType: string;
  setFilterType: (val: string) => void;
  filterStatus: string;
  setFilterStatus: (val: string) => void;
  sortType: string;
  setSortType: (val: string) => void;
}

const TYPE_CONFIG_KEY: Record<string, { labelKey: string; icon: string }> = {
  litigation_civil: { labelKey: 'clients.casesTab.type.litigationCivil', icon: '⚖' },
  litigation_arbitration: { labelKey: 'clients.casesTab.type.litigationArbitration', icon: '🏦' },
  litigation_criminal: { labelKey: 'clients.casesTab.type.litigationCriminal', icon: '🔨' },
  litigation_administrative: { labelKey: 'clients.casesTab.type.litigationAdministrative', icon: '📑' },
  contract: { labelKey: 'clients.casesTab.type.contract', icon: '💼' },
  consulting: { labelKey: 'clients.casesTab.type.consulting', icon: '📋' },
  other: { labelKey: 'clients.casesTab.type.other', icon: '📁' },
};

export function CasesTab({
  cases,
  onEdit,
  onDelete,
  onNavigate,
  searchQuery,
  setSearchQuery,
  filterType,
  setFilterType,
  filterStatus,
  setFilterStatus,
  sortType,
  setSortType,
}: Props) {
  const { t } = useI18n();

  const typeOptions = useMemo(() => {
    const options = [{ label: t('clients.casesTab.filter.allTypes' as any), value: 'all' }];
    const displayOrder = [
      'litigation_civil',
      'litigation_arbitration',
      'litigation_criminal',
      'litigation_administrative',
      'contract',
      'consulting',
      'other',
    ];

    displayOrder.forEach((type) => {
      const config = TYPE_CONFIG_KEY[type];
      options.push({
        label: `${config.icon} ${t(config.labelKey as any)}`,
        value: type,
      });
    });

    return options;
  }, [t]);

  return (
    <div className="h-full flex flex-col overflow-hidden bg-[var(--app-bg)]">
      <RegistryToolbar
        title={t('clients.casesTab.title' as any)}
        subtitle={t('clients.casesTab.subtitle' as any)}
        searchPlaceholder={t('clients.casesTab.searchPlaceholder' as any)}
        searchValue={searchQuery}
        onSearchChange={setSearchQuery}
        filters={
          <>
            <div className="w-[180px]">
              <Select
                value={sortType}
                onChange={setSortType}
                options={[
                  { label: t('clients.casesTab.sort.dateDesc' as any), value: 'date_desc' },
                  { label: t('clients.casesTab.sort.dateAsc' as any), value: 'date_asc' },
                  { label: t('clients.casesTab.sort.nameAsc' as any), value: 'name_asc' },
                  { label: t('clients.casesTab.sort.nameDesc' as any), value: 'name_desc' },
                ]}
              />
            </div>

            <div className="w-[220px]">
              <Select value={filterType} onChange={setFilterType} options={typeOptions} />
            </div>

            <div className="w-[160px]">
              <Select
                value={filterStatus}
                onChange={setFilterStatus}
                options={[
                  { label: t('clients.casesTab.filter.allStatuses' as any), value: 'all' },
                  { label: t('clients.casesTab.filter.active' as any), value: 'ACTIVE' },
                  { label: t('clients.casesTab.filter.paused' as any), value: 'PAUSED' },
                  { label: t('clients.casesTab.filter.done' as any), value: 'DONE' },
                  { label: t('clients.casesTab.filter.archive' as any), value: 'ARCHIVE' },
                ]}
              />
            </div>
          </>
        }
        rightContent={
          <div className="ml-2 text-[11px] font-black text-[var(--app-text)] opacity-40 uppercase tracking-widest">
            {t('clients.casesTab.total' as any)}: {cases.length}
          </div>
        }
      />

      <div className="flex-1 overflow-auto p-4 space-y-8">
        {cases.length > 0 ? (
          <>
            {['litigation', 'consulting', 'contract', 'other'].map((type) => {
              const filtered = cases.filter((c) => {
                if (type === 'litigation') {
                  return ['litigation', 'administrative', 'criminal', 'civil'].includes(c.case_type || '');
                }
                return c.case_type === type;
              });

              if (filtered.length === 0) return null;

              const config = TYPE_CONFIG_KEY[type === 'litigation' ? 'litigation_civil' : type];
              const titleKey =
                type === 'litigation'
                  ? 'clients.casesTab.section.litigation'
                  : type === 'consulting'
                    ? 'clients.casesTab.section.consulting'
                    : type === 'contract'
                      ? 'clients.casesTab.section.contract'
                      : 'clients.casesTab.section.other';

              return (
                <div key={type} className="space-y-3">
                  <div className="flex items-center gap-2 px-2">
                    <span className="text-[length:var(--text-lg)]">{config?.icon || '📁'}</span>
                    <UiText size="md" weight="black" tracking="tight" uppercase className="opacity-80">
                      {t(titleKey as any)}
                    </UiText>
                    <div className="h-px flex-1 bg-black/[0.05] ml-2" />
                    <UiText size="xs" weight="bold" tone="muted" className="ml-2 uppercase opacity-50">
                      {filtered.length}
                    </UiText>
                  </div>

                  <div className="rounded-xl border border-black/[0.05] bg-white shadow-sm overflow-hidden mica-effect">
                    {type === 'contract' ? (
                      <ContractsTable cases={filtered} onEdit={onEdit} onDelete={onDelete} onNavigate={onNavigate} />
                    ) : type === 'consulting' ? (
                      <ConsultingCasesTable cases={filtered} onEdit={onEdit} onDelete={onDelete} onNavigate={onNavigate} />
                    ) : type === 'other' ? (
                      <OtherCasesTable cases={filtered} onEdit={onEdit} onDelete={onDelete} onNavigate={onNavigate} />
                    ) : (
                      <CasesTable cases={filtered} onEdit={onEdit} onDelete={onDelete} onNavigate={onNavigate} />
                    )}
                  </div>
                </div>
              );
            })}
          </>
        ) : (
          <div className="py-20 text-center flex flex-col items-center gap-3">
            <div className="text-[var(--app-text-muted)] text-[13px]">
              {filterType === 'all' && searchQuery === ''
                ? t('clients.casesTab.empty.base' as any)
                : t('clients.casesTab.empty.filtered' as any)}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
