import type { Dispatch, SetStateAction } from 'react';
import { HierarchySelector } from './HierarchySelector';
import type { JobEditValues } from '../hooks/useJobEditing';

export function HierarchyEditPanel({
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
    <div className="grid gap-3 md:grid-cols-2">
      <HierarchySelector type="client" value={values.client} options={options.clientOptions} clientType={values.client_type} onChange={(value) => update({ client: value, case: '', line: '', appeal: '' })} />
      <HierarchySelector type="case" value={values.case} options={options.caseOptions} onChange={(value) => update({ case: value, line: '', appeal: '' })} />
      <HierarchySelector type="line" value={values.line} options={options.lineOptions} onChange={(value) => update({ line: value, appeal: '' })} />
      <HierarchySelector type="appeal" value={values.appeal} options={options.appealOptions} onChange={(value) => update({ appeal: value })} />
      <div className="flex gap-2">
        <button type="button" onClick={() => update({ client_type: 'LEGAL' })} className={`rounded-md border px-2 py-1 text-[10px] font-bold ${values.client_type === 'LEGAL' ? 'border-[var(--app-accent)] bg-[var(--app-accent)] text-white' : 'border-[var(--app-border)] bg-[var(--app-surface)] text-[var(--app-text-muted)]'}`}>LEGAL</button>
        <button type="button" onClick={() => update({ client_type: 'INDIVIDUAL' })} className={`rounded-md border px-2 py-1 text-[10px] font-bold ${values.client_type === 'INDIVIDUAL' ? 'border-[var(--app-accent)] bg-[var(--app-accent)] text-white' : 'border-[var(--app-border)] bg-[var(--app-surface)] text-[var(--app-text-muted)]'}`}>INDIVIDUAL</button>
      </div>
    </div>
  );
}
