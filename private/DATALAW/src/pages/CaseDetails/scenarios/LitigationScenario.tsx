import { Shield, Info, Plus, FileText } from 'lucide-react';
import { PageHeader } from '../../../components/ui/layout/PageHeader';
import { Window } from '../../../components/ui/layout/Window';
import { SegmentedControl } from '../../../components/ui/layout/SegmentedControl';
import { PageLoader } from '../../../components/ui/feedback/PageLoader';
import { Button } from '../../../components/ui/form/Button';
import { useI18n } from '../../../i18n/AppLanguageProvider';
import { CaseInfoPanel } from '../components/CaseInfoPanel';
import { Case, Line, Appeal } from '../../../types/entities';
import { useState, useMemo } from 'react';
import { AppealsTable } from '../../LineDetails/components/AppealsTable';
import { AppealEditModal } from '../../LineDetails/components/AppealEditModal';
import { RegistryToolbar } from '../../../shared/registry/RegistryToolbar';
import { Select } from '../../../components/ui/form/Select';

interface LitigationScenarioProps {
  case_: Case | null;
  loading: boolean;
  view: 'lines' | 'info';
  setView: (view: 'lines' | 'info') => void;
  lines: Line[];
  onBack: () => void;
  onNavigateLine: (line: Line) => void;
  defaultLine: Line | null;
  appeals: Appeal[];
  handleSaveAppeal: (data: any) => Promise<void>;
  handleUploadDocument: (file?: File) => Promise<void>;
  handleDeleteAppeal: (id: number) => Promise<void>;
  onNavigateAppeal?: (appeal: Appeal) => void;
  isEditOpen: boolean;
  isSaving: boolean;
  formData: any;
  setFormData: (data: any) => void;
  openCreate: () => void;
  openEdit: (line: Line) => void;
  closeEdit: () => void;
  handleSave: () => void;
  isDeleteOpen: boolean;
  selectedItem: Line | null;
  openDelete: (line: Line) => void;
  closeDelete: () => void;
  confirmDelete: () => void;
}

export function LitigationScenario({
  case_,
  loading,
  view,
  setView,
  onBack,
  defaultLine,
  appeals,
  handleSaveAppeal,
  handleDeleteAppeal,
  onNavigateAppeal,
}: LitigationScenarioProps) {
  const { t } = useI18n();
  const [searchQuery, setSearchQuery] = useState('');
  const [sortType, setSortType] = useState<'date_desc' | 'date_asc' | 'name_asc' | 'name_desc'>('date_desc');
  const [statusFilter, setStatusFilter] = useState<'all' | 'draft' | 'sent' | 'active' | 'done' | 'archive'>('all');
  const [isAppealModalOpen, setIsAppealModalOpen] = useState(false);
  const [editingAppeal, setEditingAppeal] = useState<Appeal | null>(null);

  const getLitigationTitle = () => {
    if (!case_) return t('scenario.litigation.title');
    const subtype = case_.litigation_subtype;
    if (subtype === 'civil') return t('scenario.litigation.civil');
    if (subtype === 'arbitration') return t('scenario.litigation.arbitration');
    if (subtype === 'criminal') return t('scenario.litigation.criminal');
    if (subtype === 'administrative') return t('scenario.litigation.administrative');
    return t('scenario.litigation.title');
  };

  const filteredAppeals = useMemo(() => {
    let result = appeals;

    // Search filter
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase().trim();
      result = result.filter(
        (a) => a.title.toLowerCase().includes(q) || a.agency_code?.toLowerCase().includes(q),
      );
    }

    // Status filter
    if (statusFilter !== 'all') {
      result = result.filter((a) => a.status?.toLowerCase() === statusFilter);
    }

    // Sort
    result = [...result].sort((a, b) => {
      switch (sortType) {
        case 'date_desc':
          return new Date(b.appeal_date || 0).getTime() - new Date(a.appeal_date || 0).getTime();
        case 'date_asc':
          return new Date(a.appeal_date || 0).getTime() - new Date(b.appeal_date || 0).getTime();
        case 'name_asc':
          return a.title.localeCompare(b.title);
        case 'name_desc':
          return b.title.localeCompare(a.title);
        default:
          return 0;
      }
    });

    return result;
  }, [appeals, searchQuery, statusFilter, sortType]);

  if (loading || !case_ || !defaultLine) return <PageLoader />;

  return (
    <div className="flex flex-col h-full bg-[var(--app-bg)] overflow-hidden animate-in fade-in duration-300">
      <PageHeader
        title={getLitigationTitle()}
        subtitle={case_.title}
        icon={<Shield size={20} className="text-[var(--app-accent)]" />}
        onBack={onBack}
        rightContent={
          <div className="flex items-center gap-3">
            <SegmentedControl
              active={view}
              onChange={(id) => setView(id as 'lines' | 'info')}
              segments={[
                { id: 'lines', label: t('caseDetails.litigation.appealsTab'), icon: <FileText size={14} /> },
                { id: 'info', label: t('appeal.info'), icon: <Info size={14} /> },
              ]}
            />
            <Button
              variant="primary"
              size="md"
              onClick={() => {
                setEditingAppeal(null);
                setIsAppealModalOpen(true);
              }}
              icon={<Plus size={16} />}
            >
              {t('caseDetails.litigation.newAppeal')}
            </Button>
          </div>
        }
      />

      <div className="flex-1 overflow-hidden p-4">
        {view === 'info' ? (
          <Window noPadding className="h-full overflow-auto mica-effect">
            <CaseInfoPanel case_={case_} />
          </Window>
        ) : (
          <Window noPadding className="h-full flex flex-col overflow-hidden mica-effect">
            <RegistryToolbar
              searchPlaceholder={t('caseDetails.litigation.searchPlaceholder')}
              searchValue={searchQuery}
              onSearchChange={setSearchQuery}
              filters={
                <>
                  <Select
                    value={sortType}
                    onChange={(v) => setSortType(v as typeof sortType)}
                    options={[
                      { label: t('clients.sort.newest'), value: 'date_desc' },
                      { label: t('clients.sort.oldest'), value: 'date_asc' },
                      { label: t('clients.sort.nameAsc'), value: 'name_asc' },
                      { label: t('clients.sort.nameDesc'), value: 'name_desc' },
                    ]}
                  />
                  <Select
                    value={statusFilter}
                    onChange={(v) => setStatusFilter(v as typeof statusFilter)}
                    options={[
                      { label: t('clients.filter.status.all'), value: 'all' },
                      { label: t('clients.filter.status.active'), value: 'active' },
                      { label: t('clients.filter.status.done'), value: 'done' },
                      { label: t('clients.filter.status.archive'), value: 'archive' },
                    ]}
                  />
                </>
              }
              rightContent={
                <div className="text-[11px] font-black text-[var(--app-text)] opacity-40 uppercase tracking-widest">
                  {t('caseDetails.litigation.total')}: {filteredAppeals.length}
                </div>
              }
            />

            <div className="flex-1 overflow-auto">
              <AppealsTable
                items={filteredAppeals}
                onEdit={(a) => {
                  setEditingAppeal(a);
                  setIsAppealModalOpen(true);
                }}
                onDelete={(a) => handleDeleteAppeal(a.id)}
                onNavigate={(a) => onNavigateAppeal?.(a)}
              />
              {filteredAppeals.length === 0 && (
                <div className="py-32 text-center flex flex-col items-center gap-4 opacity-20">
                  <FileText size={48} strokeWidth={1} />
                  <div className="flex flex-col gap-1">
                    <div className="text-base font-bold">{t('caseDetails.litigation.emptyTitle')}</div>
                    <div className="text-xs opacity-70">{t('caseDetails.litigation.emptyHint')}</div>
                  </div>
                </div>
              )}
            </div>
          </Window>
        )}
      </div>

      {isAppealModalOpen && (
        <AppealEditModal
          open={isAppealModalOpen}
          onClose={() => setIsAppealModalOpen(false)}
          onSave={async () => {
            await handleSaveAppeal(editingAppeal || {});
            setIsAppealModalOpen(false);
          }}
          formData={editingAppeal || {}}
          setFormData={(data) => setEditingAppeal(data as Appeal)}
          isNew={!editingAppeal}
          isSaving={false}
          isLitigation={true}
        />
      )}
    </div>
  );
}
