import type { Dispatch, SetStateAction } from 'react';
import { AppInput, UiCaption } from '../../../components/ui';
import type { JobEditValues } from '../hooks/useJobEditing';
import { HierarchyEditPanel } from './HierarchyEditPanel';

export function JobEditForm({
  values,
  setValues,
  options,
}: {
  values: JobEditValues;
  setValues: Dispatch<SetStateAction<JobEditValues | null>>;
  options: {
    clientOptions: { label: string; value: string }[];
    caseOptions: { label: string; value: string }[];
    lineOptions: { label: string; value: string }[];
    appealOptions: { label: string; value: string }[];
  };
}) {
  const update = (patch: Partial<JobEditValues>) => setValues((prev) => (prev ? { ...prev, ...patch } : prev));
  return (
    <div className="space-y-3 rounded-xl border border-[var(--app-border)] bg-[var(--app-surface-soft)] p-3">
      <HierarchyEditPanel values={values} setValues={setValues} options={options} />
      <div className="grid gap-3 md:grid-cols-2">
        <div>
          <UiCaption>Дата</UiCaption>
          <AppInput type="date" value={values.date || ''} onChange={(e) => update({ date: e.target.value })} />
        </div>
        <div>
          <UiCaption>Имя файла</UiCaption>
          <AppInput value={values.filename || ''} onChange={(e) => update({ filename: e.target.value })} />
        </div>
      </div>
    </div>
  );
}
