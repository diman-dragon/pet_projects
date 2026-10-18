import type { ChangeEvent } from 'react';
import { useMemo, useState, useEffect } from 'react';
import { Info } from 'lucide-react';
import { useI18n } from '../../../i18n/AppLanguageProvider';
import type { Case } from '../../../types/entities';
import { Field } from '../../../components/ui/form/Field';
import { Input } from '../../../components/ui/form/Input';
import { Select } from '../../../components/ui/form/Select';
import { Textarea } from '../../../components/ui/form/Textarea';
import { UiText } from '../../../components/ui/atoms/UiText';

interface Props {
  formData: Partial<Case>;
  setFormData: (data: Partial<Case>) => void;
  isNew: boolean;
}

export function CaseEditForm({ formData, setFormData, isNew }: Props) {
  const { t } = useI18n();

  const statusOptions = useMemo(
    () =>
      (['LEAD', 'ACTIVE', 'PAUSED', 'ARCHIVE'] as const).map((value) => ({
        value,
        label: t(`case.status.${value}` as any) || value,
      })),
    [t],
  );

  const typeOptions = useMemo(
    () => [
      { value: 'litigation', label: t('caseEdit.caseType.litigation' as any) },
      { value: 'consulting', label: t('caseEdit.caseType.consulting' as any) },
      { value: 'contract', label: t('caseEdit.caseType.contract' as any) },
      { value: 'other', label: t('caseEdit.caseType.other' as any) },
    ],
    [t],
  );

  const litigationSubtypeOptions = useMemo(
    () => [
      { value: 'civil', label: t('caseEdit.litigationSubtype.civil' as any) },
      { value: 'arbitration', label: t('caseEdit.litigationSubtype.arbitration' as any) },
      { value: 'criminal', label: t('caseEdit.litigationSubtype.criminal' as any) },
      { value: 'administrative', label: t('caseEdit.litigationSubtype.administrative' as any) },
    ],
    [t],
  );

  const [customContractTypes, setCustomContractTypes] = useState<string[]>([]);

  useEffect(() => {
    const saved = localStorage.getItem('Dorlane_custom_contract_types');
    if (saved) {
      setCustomContractTypes(JSON.parse(saved));
    }
  }, []);

  const contractTypeOptions = useMemo(() => {
    const base = [
      { value: 'supply', label: t('caseEdit.contractType.supply' as any) },
      { value: 'service', label: t('caseEdit.contractType.service' as any) },
      { value: 'lease', label: t('caseEdit.contractType.lease' as any) },
      { value: 'other', label: t('caseEdit.contractType.other' as any) },
    ];
    const custom = customContractTypes.map((contractType) => ({
      value: contractType,
      label: contractType,
    }));
    return [...base, ...custom];
  }, [customContractTypes, t]);

  const handleAddCustomContractType = (type: string) => {
    if (!type || customContractTypes.includes(type)) return;
    const updated = [...customContractTypes, type];
    setCustomContractTypes(updated);
    localStorage.setItem('Dorlane_custom_contract_types', JSON.stringify(updated));
  };

  const set = (key: keyof Case) => (event: ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) =>
    setFormData({ ...formData, [key]: event.target.value });

  return (
    <div className="space-y-5">
      <div className="grid grid-cols-12 gap-4">
        <div className="col-span-6">
          <Field label={t('caseEdit.field.workType' as any)}>
            <Select
              value={formData.case_type ?? 'litigation'}
              onChange={(value) =>
                setFormData({
                  ...formData,
                  case_type: value,
                  litigation_subtype: value === 'litigation' ? 'civil' : undefined,
                  counterparty: undefined,
                  metadata: undefined,
                  title: '',
                })
              }
              options={typeOptions}
            />
          </Field>
        </div>

        <div className="col-span-6">
          <Field label={t('caseEdit.field.status' as any)}>
            <Select
              value={formData.status ?? 'ACTIVE'}
              onChange={(value) => setFormData({ ...formData, status: value })}
              options={statusOptions}
            />
          </Field>
        </div>
      </div>

      {formData.case_type === 'litigation' && (
        <div className="space-y-4">
          <div className="grid grid-cols-12 gap-4">
            <div className="col-span-6">
              <Field label={t('caseEdit.field.litigationType' as any)} required>
                <Select
                  value={formData.litigation_subtype ?? 'civil'}
                  onChange={(value) => setFormData({ ...formData, litigation_subtype: value })}
                  options={litigationSubtypeOptions}
                />
              </Field>
            </div>
            <div className="col-span-6">
              <Field label={t('caseEdit.field.caseNumber' as any)}>
                <Input
                  placeholder={t('caseEdit.placeholder.caseNumber' as any)}
                  value={formData.case_number ?? ''}
                  onChange={set('case_number')}
                />
              </Field>
            </div>
          </div>
          <Field label={t('caseEdit.field.title' as any)} required>
            <Input
              placeholder={t('caseEdit.placeholder.title' as any)}
              value={formData.title ?? ''}
              onChange={set('title')}
            />
          </Field>
          <Field label={t('caseEdit.field.summary' as any)}>
            <Textarea
              className="min-h-[100px]"
              placeholder={t('caseEdit.placeholder.summary' as any)}
              value={formData.summary ?? ''}
              onChange={set('summary')}
            />
          </Field>
        </div>
      )}

      {formData.case_type === 'consulting' && (
        <div className="space-y-4">
          <Field label={t('caseEdit.field.topic' as any)} required>
            <Input
              placeholder={t('caseEdit.placeholder.topic' as any)}
              value={formData.title ?? ''}
              onChange={(e) => {
                const topic = e.target.value;
                const now = new Date();
                const yymm = `${String(now.getFullYear()).slice(-2)}${String(now.getMonth() + 1).padStart(2, '0')}`;
                setFormData({
                  ...formData,
                  title: topic,
                  metadata: yymm,
                });
              }}
            />
          </Field>
          <Field label={t('caseEdit.field.topicSummary' as any)}>
            <Textarea
              className="min-h-[100px]"
              placeholder={t('caseEdit.placeholder.topicSummary' as any)}
              value={formData.summary ?? ''}
              onChange={set('summary')}
            />
          </Field>
        </div>
      )}

      {formData.case_type === 'contract' && (
        <div className="space-y-4">
          <div className="grid grid-cols-12 gap-4">
            <div className="col-span-6">
              <Field label={t('caseEdit.field.counterparty' as any)} required>
                <Input
                  placeholder={t('caseEdit.placeholder.counterparty' as any)}
                  value={formData.counterparty ?? ''}
                  onChange={(e) => {
                    const counterparty = e.target.value;
                    setFormData({
                      ...formData,
                      counterparty,
                      title: counterparty,
                    });
                  }}
                />
              </Field>
            </div>
            <div className="col-span-6">
              <Field label={t('caseEdit.field.contractDate' as any)}>
                <Input
                  type="date"
                  value={formData.contract_date ?? ''}
                  onChange={(e) => {
                    setFormData({
                      ...formData,
                      contract_date: e.target.value || undefined,
                    });
                  }}
                />
              </Field>
            </div>
          </div>
          <div className="grid grid-cols-12 gap-4">
            <div className="col-span-6">
              <Field label={t('caseEdit.field.contractNumber' as any)}>
                <Input
                  placeholder={t('caseEdit.placeholder.contractNumber' as any)}
                  value={formData.case_number ?? ''}
                  onChange={set('case_number')}
                />
              </Field>
            </div>
            <div className="col-span-6">
              <Field label={t('caseEdit.field.contractType' as any)}>
                <Select
                  value={formData.contract_type ?? ''}
                  onChange={(value) => {
                    if (value === 'other') {
                      const newType = prompt(t('caseEdit.prompt.contractType' as any));
                      if (newType) {
                        handleAddCustomContractType(newType);
                        setFormData({ ...formData, contract_type: newType });
                      }
                    } else {
                      setFormData({ ...formData, contract_type: value });
                    }
                  }}
                  options={[
                    { value: '', label: t('caseEdit.contractType.unspecified' as any) },
                    ...contractTypeOptions,
                  ]}
                />
              </Field>
            </div>
          </div>
          <div className="grid grid-cols-12 gap-4">
            <div className="col-span-6">
              <Field label={t('caseEdit.field.contractAmount' as any)}>
                <Input
                  type="number"
                  placeholder="0.00"
                  value={formData.contract_amount ?? ''}
                  onChange={(e) => setFormData({ ...formData, contract_amount: e.target.value })}
                />
              </Field>
            </div>
          </div>
          <Field label={t('caseEdit.field.notes' as any)}>
            <Textarea
              className="min-h-[100px]"
              placeholder={t('caseEdit.placeholder.notes' as any)}
              value={formData.summary ?? ''}
              onChange={set('summary')}
            />
          </Field>
        </div>
      )}

      {formData.case_type === 'other' && (
        <div className="space-y-4">
          <Field label={t('caseEdit.field.subject' as any)} required>
            <Input
              placeholder={t('caseEdit.placeholder.subject' as any)}
              value={formData.title ?? ''}
              onChange={set('title')}
            />
          </Field>
          <Field label={t('caseEdit.field.summary' as any)}>
            <Textarea
              className="min-h-[100px]"
              placeholder={t('caseEdit.placeholder.otherSummary' as any)}
              value={formData.summary ?? ''}
              onChange={set('summary')}
            />
          </Field>
        </div>
      )}

      {isNew && (
        <div className="flex items-start gap-3 rounded-xl border border-[var(--app-border)] bg-[var(--app-surface-soft)] p-3">
          <div className="mt-0.5 shrink-0 text-[var(--app-accent)]">
            <Info size={16} strokeWidth={2} />
          </div>
          <div className="space-y-0.5">
            <UiText as="p" size="xs" weight="semibold" tone="default">
              {t('caseEdit.systemActionTitle')}
            </UiText>
            <UiText as="p" size="xs" tone="secondary">
              {t('caseEdit.systemActionBody')}
            </UiText>
          </div>
        </div>
      )}
    </div>
  );
}
