import { Brain } from 'lucide-react';
import { PageBase } from '../../shared/ui/PageBase';
import { useI18n } from '../../i18n/AppLanguageProvider';
import { AlertModal } from '../../components/ui';
import { useSystematizerLogic } from './useSystematizerLogic';
import { useJobEditing } from './hooks/useJobEditing';
import { useJobSelection } from './hooks/useJobSelection';
import { SystematizerWorkspace } from './components/SystematizerWorkspace';

export default function SystematizerPage() {
  const { t } = useI18n();
  const logic = useSystematizerLogic();
  const editing = useJobEditing(logic.hierarchy, logic.suggestions);
  const selection = useJobSelection(logic.suggestions.map((job) => job.id));

  return (
    <PageBase title={t('sidebar.menu.systematizer')} subtitle={t('systematizer.subtitle')} icon={<Brain size={22} strokeWidth={1.8} />}>
      <SystematizerWorkspace logic={logic} editing={editing} selection={selection} />
      <AlertModal open={!!logic.alertMessage} title={t('common.error')} message={logic.alertMessage || ''} onClose={logic.clearAlert} />
    </PageBase>
  );
}
