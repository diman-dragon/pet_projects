import { useEffect, useMemo, useState } from 'react';

import { Field, Input, Select } from '../../../components/ui';
import { Button } from '../../../components/ui/form/Button';
import { Modal } from '../../../components/ui/overlays/Modal';
import { ModalTitle } from '../../../components/ui/overlays/ModalTitle';
import { useI18n } from '../../../i18n/AppLanguageProvider';
import {
  businessApi,
  type AgencyDictionaryEntry,
  type AppErrorLike,
} from '../../../services/backend/businessApi';
import type { Appeal } from '../../../types/entities';
import { APPEAL_STATUS_OPTIONS } from './appealStatusOptions';

interface Props {
  formData: Partial<Appeal>;
  setFormData: (data: Partial<Appeal>) => void;
  isNew: boolean;
  isLitigation?: boolean;
}

function normalizeAgencyCode(value: string) {
  return value.toUpperCase().replace(/[^A-Z0-9]/g, '').slice(0, 4);
}

export function AppealEditForm({ formData, setFormData, isNew, isLitigation }: Props) {
  const { t } = useI18n();
  const [agencies, setAgencies] = useState<AgencyDictionaryEntry[]>([]);
  const [isLoadingAgencies, setIsLoadingAgencies] = useState(false);
  const [isAddAgencyModalOpen, setIsAddAgencyModalOpen] = useState(false);
  const [newAgencyName, setNewAgencyName] = useState('');
  const [newAgencyCode, setNewAgencyCode] = useState('');
  const [isSavingAgency, setIsSavingAgency] = useState(false);
  const [isSuggestingCode, setIsSuggestingCode] = useState(false);
  const [agencyErrorKey, setAgencyErrorKey] = useState<string | null>(null);

  useEffect(() => {
    let active = true;

    void (async () => {
      setIsLoadingAgencies(true);
      try {
        const entries = await businessApi.listAgencyDictionary();
        if (active) {
          setAgencies(entries);
        }
      } catch (error) {
        console.error('Failed to load agency dictionary', error);
      } finally {
        if (active) {
          setIsLoadingAgencies(false);
        }
      }
    })();

    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (isNew && !formData.appeal_date) {
      const today = new Date().toISOString().split('T')[0];
      setFormData({ ...formData, appeal_date: today });
    }
  }, [formData, isNew, setFormData]);

  useEffect(() => {
    if (!isAddAgencyModalOpen) {
      return;
    }

    const trimmedName = newAgencyName.trim();
    if (!trimmedName) {
      setNewAgencyCode('');
      setAgencyErrorKey(null);
      return;
    }

    let active = true;
    setIsSuggestingCode(true);
    setAgencyErrorKey(null);

    void businessApi
      .suggestAgencyCode(trimmedName)
      .then((suggestion) => {
        if (active) {
          setNewAgencyCode(suggestion.code);
        }
      })
      .catch((error) => {
        console.error('Failed to suggest agency code', error);
        if (active) {
          setAgencyErrorKey((error as AppErrorLike).key || 'line.error.save');
        }
      })
      .finally(() => {
        if (active) {
          setIsSuggestingCode(false);
        }
      });

    return () => {
      active = false;
    };
  }, [isAddAgencyModalOpen, newAgencyName]);

  const statusOptions = APPEAL_STATUS_OPTIONS.map((item) => ({
    label: t(item.labelKey),
    value: item.value,
  }));

  const agencyOptions = useMemo(() => {
    const options = agencies.map((agency) => ({
      label: `${agency.name} (${agency.code})`,
      value: agency.code,
    }));

    const currentCode = formData.agency_code?.trim();
    if (currentCode && !options.some((agency) => agency.value === currentCode)) {
      options.unshift({
        label: `${currentCode} (${currentCode})`,
        value: currentCode,
      });
    }

    return [
      { label: t('appeal.edit.none' as any), value: '' },
      ...options,
      { label: t('appeal.edit.addAgencyAction' as any), value: 'ADD_NEW' },
    ];
  }, [agencies, formData.agency_code, t]);

  const set = (key: keyof Appeal, value: string) => setFormData({ ...formData, [key]: value });

  const handleAgencyChange = (value: string) => {
    if (value === 'ADD_NEW') {
      setIsAddAgencyModalOpen(true);
      setNewAgencyName('');
      setNewAgencyCode('');
      setAgencyErrorKey(null);
      return;
    }

    set('agency_code', value);
  };

  const handleAddAgency = async () => {
    const trimmedName = newAgencyName.trim();
    const normalizedCode = normalizeAgencyCode(newAgencyCode);
    if (!trimmedName || !normalizedCode) {
      return;
    }

    setIsSavingAgency(true);
    setAgencyErrorKey(null);

    try {
      const savedEntry = await businessApi.saveAgencyDictionaryEntry({
        name: trimmedName,
        code: normalizedCode,
      });

      setAgencies((current) => {
        const next = current.filter((entry) => entry.name !== savedEntry.name && entry.code !== savedEntry.code);
        return [...next, savedEntry].sort((left, right) => left.name.localeCompare(right.name));
      });
      set('agency_code', savedEntry.code);
      setIsAddAgencyModalOpen(false);
      setNewAgencyName('');
      setNewAgencyCode('');
    } catch (error) {
      console.error('Failed to save agency dictionary entry', error);
      setAgencyErrorKey((error as AppErrorLike).key || 'line.error.save');
    } finally {
      setIsSavingAgency(false);
    }
  };

  return (
    <>
      <div className="space-y-5">
        <Field label={t('appeal.edit.titleLabel')} required hint={t('appeal.edit.titleHint')}>
          <Input
            autoFocus
            placeholder={t('appeal.edit.titlePlaceholder')}
            value={formData.title ?? ''}
            onChange={(event) => set('title', event.target.value)}
          />
        </Field>

        <div className="grid grid-cols-2 gap-4">
          <Field label={t('appeal.edit.lineOrAgencyLabel' as any)}>
            <Select
              value={formData.agency_code ?? ''}
              onChange={handleAgencyChange}
              options={agencyOptions}
              disabled={isLoadingAgencies}
            />
          </Field>

          <Field label={t('appeal.edit.statusLabel')}>
            <Select
              value={formData.status ?? 'DRAFT'}
              onChange={(value) => set('status', value)}
              options={statusOptions}
            />
          </Field>
        </div>

        <div className="grid grid-cols-2 gap-4">
          {isLitigation ? (
            <Field label={t('appeal.edit.incomingLabel' as any)}>
              <Input
                className="font-mono text-[11px]"
                placeholder={t('appeal.edit.incomingPlaceholder' as any)}
                value={formData.incoming_number ?? ''}
                onChange={(event) => set('incoming_number' as keyof Appeal, event.target.value)}
              />
            </Field>
          ) : (
            <Field label={t('appeal.edit.outgoingLabel')}>
              <Input
                className="font-mono text-[11px]"
                placeholder={t('appeal.edit.outgoingPlaceholder')}
                value={formData.outgoing_number ?? ''}
                onChange={(event) => set('outgoing_number', event.target.value)}
              />
            </Field>
          )}

          <Field label={t('appeal.edit.dateLabel')}>
            <Input
              type="date"
              className="font-mono text-[11px]"
              value={formData.appeal_date ?? ''}
              onChange={(event) => set('appeal_date', event.target.value)}
            />
          </Field>
        </div>

        {isNew && (
          <p className="rounded-xl border border-black/[0.05] bg-[var(--app-surface-soft)] p-3 text-[11px] leading-relaxed text-black/50">
            {t('appeal.edit.folderNote')}
          </p>
        )}
      </div>

      <Modal
        open={isAddAgencyModalOpen}
        onClose={() => setIsAddAgencyModalOpen(false)}
        title={
          <ModalTitle
            icon={<div />}
            title={t('appeal.edit.addAgencyTitle' as any)}
            subtitle={t('appeal.edit.addAgencySubtitle' as any)}
          />
        }
        footer={
          <div className="flex w-full items-center justify-end gap-2">
            <Button variant="ghost" onClick={() => setIsAddAgencyModalOpen(false)}>
              {t('common.cancel')}
            </Button>
            <Button
              variant="primary"
              disabled={!newAgencyName.trim() || !newAgencyCode.trim() || isSavingAgency}
              onClick={() => void handleAddAgency()}
            >
              {t('common.add' as any)}
            </Button>
          </div>
        }
      >
        <div className="space-y-4">
          <Field label={t('appeal.edit.agencyNameLabel' as any)} required>
            <Input
              value={newAgencyName}
              onChange={(event) => setNewAgencyName(event.target.value)}
              placeholder={t('appeal.edit.agencyNamePlaceholder' as any)}
              autoFocus
            />
          </Field>

          <Field
            label={t('appeal.edit.agencyCodeLabel' as any)}
            hint={isSuggestingCode ? t('common.loading' as any) : t('appeal.edit.agencyCodeHint' as any)}
            error={agencyErrorKey ? t(agencyErrorKey as any) : undefined}
          >
            <Input
              value={newAgencyCode}
              onChange={(event) => setNewAgencyCode(normalizeAgencyCode(event.target.value))}
              placeholder={t('appeal.edit.agencyCodePlaceholder' as any)}
              maxLength={4}
              className="font-mono uppercase"
            />
          </Field>
        </div>
      </Modal>
    </>
  );
}
