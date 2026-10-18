import { AlertTriangle, Loader2, RotateCcw } from 'lucide-react';
import { useI18n } from '../../../i18n/AppLanguageProvider';
import { Button } from '../../../components/ui/form/Button';
import { Modal } from '../../../components/ui/overlays/Modal';
import { ModalTitle } from '../../../components/ui/overlays/ModalTitle';

interface LicenseResetConfirmModalProps {
  open: boolean;
  loading: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}

export function LicenseResetConfirmModal({
  open,
  loading,
  onConfirm,
  onCancel,
}: LicenseResetConfirmModalProps) {
  const { t } = useI18n();

  return (
    <Modal
      open={open}
      onClose={onCancel}
      title={
        <ModalTitle
          icon={<AlertTriangle size={18} />}
          title={t('license.resetTitle')}
          subtitle={t('license.resetSubtitle')}
        />
      }
      size="sm"
      showCloseButton={!loading}
      footer={
        <div className="flex w-full items-center justify-end gap-2">
          <Button variant="secondary" size="sm" onClick={onCancel} disabled={loading}>
            {t('license.resetCancel')}
          </Button>
          <Button
            variant="danger"
            size="sm"
            icon={
              loading ? <Loader2 size={12} className="animate-spin" /> : <RotateCcw size={12} />
            }
            onClick={onConfirm}
            disabled={loading}
          >
            {loading ? t('license.resetPending') : t('license.resetConfirm')}
          </Button>
        </div>
      }
    >
      <div className="space-y-4">
        <p className="text-[12px] leading-relaxed text-black/70">{t('license.resetBody')}</p>

        <div className="rounded-xl border border-amber-200 bg-amber-50 px-4 py-3">
          <p className="text-[11px] leading-relaxed text-amber-700">{t('license.resetWarning')}</p>
        </div>
      </div>
    </Modal>
  );
}
