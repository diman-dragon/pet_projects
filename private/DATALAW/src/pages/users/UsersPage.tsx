import { LayoutGrid, Plus, TrendingUp, Users } from 'lucide-react';
import { Button } from '../../components/ui/form/Button';
import { DeleteConfirmModal } from '../../components/ui/overlays/DeleteConfirmModal';
import { useI18n } from '../../i18n/AppLanguageProvider';
import { PageBase } from '../../shared/ui/PageBase';
import RegistryAnalytics from '../../shared/registry/RegistryAnalytics';
import type { UserProfile } from '../../types/entities';
import { UserEditModal } from './components/UserEditModal';
import { UsersTable } from './components/UsersTable';
import { useUsersLogic } from './useUsersLogic';
import { userAnalytics } from './users_analytics';

interface UsersPageProps {
  currentUser: UserProfile | null;
}

export default function UsersPage({ currentUser }: UsersPageProps) {
  const { language, t } = useI18n();
  const logic = useUsersLogic(currentUser);

  return (
    <PageBase
      title={t('users.title')}
      subtitle={t('users.subtitle')}
      icon={<Users size={20} />}
      loading={logic.loading}
      activeTab={logic.activeTab}
      onTabChange={(id) => logic.setActiveTab(id as 'list' | 'analytics')}
      tabs={[
        { id: 'list', label: t('users.tab.list'), icon: <LayoutGrid size={14} /> },
        { id: 'analytics', label: t('users.tab.analytics'), icon: <TrendingUp size={14} /> },
      ]}
      action={
        <Button variant="primary" size="md" icon={<Plus size={16} />} onClick={logic.openCreate}>
          {t('users.add')}
        </Button>
      }
      noPadding={logic.activeTab === 'list'}
      extra={
        <>
          <UserEditModal open={logic.isEditOpen} onClose={logic.closeEdit} logic={logic} />
          <DeleteConfirmModal
            target={logic.isDeleteOpen ? logic.selectedItem : null}
            entityLabel={t('users.title').toLowerCase()}
            entityName={logic.selectedItem?.full_name ?? ''}
            onCancel={logic.closeDelete}
            onConfirm={logic.confirmDelete}
          />
        </>
      }
    >
      {logic.activeTab === 'list' ? (
        <UsersTable
          items={logic.items}
          onEdit={logic.openEdit}
          onDelete={logic.openDelete}
          canEdit={logic.canEditUser}
          canDelete={logic.canDeleteUser}
        />
      ) : (
        <div className="h-full overflow-y-auto rounded-[var(--app-radius-md)] border border-[var(--app-border)] bg-[var(--app-surface)] p-6">
          <RegistryAnalytics
            items={logic.rawItems}
            config={{
              ...userAnalytics,
              stats: (items) => userAnalytics.stats(items, language),
              chartData: (items) => userAnalytics.chartData(items, language),
              chartLabel: t('common.analytics'),
            }}
          />
        </div>
      )}
    </PageBase>
  );
}
