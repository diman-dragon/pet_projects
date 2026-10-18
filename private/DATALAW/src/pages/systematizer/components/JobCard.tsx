import type { Dispatch, SetStateAction } from 'react';
import { Check, CheckCircle2, Edit2, FileText, Loader2, X, XCircle } from 'lucide-react';
import { UiCaption, UiPanel, UiText } from '../../../components/ui';
import type { SystematizerJob } from '../../../store/systematizerStore';
import { NamingService } from '../../../services/utils/NamingService';
import { JobEditForm } from './JobEditForm';
import type { JobEditValues } from '../hooks/useJobEditing';

export function JobCard({
  job,
  checked,
  editing,
  values,
  options,
  setValues,
  onToggle,
  onEdit,
  onApprove,
  onReject,
  onCancel,
  onSave,
  onResetStatus,
}: {
  job: SystematizerJob;
  checked: boolean;
  editing: boolean;
  values: JobEditValues | null;
  options: { clientOptions: any[]; caseOptions: any[]; lineOptions: any[]; appealOptions: any[] };
  setValues: Dispatch<SetStateAction<JobEditValues | null>>;
  onToggle: () => void;
  onEdit: () => void;
  onApprove: () => void;
  onReject: () => void;
  onCancel: () => void;
  onSave: () => void;
  onResetStatus: () => void;
}) {
  const icon = job.status === 'processing' || job.status === 'applying' ? <Loader2 size={18} className="animate-spin text-[var(--app-accent)]" /> : job.status === 'applied' ? <CheckCircle2 size={18} className="text-[var(--app-success)]" /> : <FileText size={18} />;
  return (
    <UiPanel className={`flex gap-4 border-black/[0.05] p-4 ${job.status === 'applied' ? 'bg-[var(--app-success)]/5 opacity-60' : 'hover:shadow-md'}`}>
      <input type="checkbox" checked={checked} onChange={onToggle} className="mt-1" />
      <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-[var(--app-surface-soft)]">{icon}</div>
      <div className="min-w-0 flex-1 space-y-3">
        <div className="flex items-center justify-between gap-2">
          <UiText weight="bold" className="truncate">{job.fileName}</UiText>
          <div className="flex items-center gap-1">
            {job.status === 'completed' && job.userStatus === 'pending' && !editing && <>
              <button onClick={onEdit} className="rounded-md p-1.5 text-[var(--app-accent)] hover:bg-[var(--app-accent)]/10"><Edit2 size={15} /></button>
              <button onClick={onApprove} className="rounded-md p-1.5 text-[var(--app-success)] hover:bg-[var(--app-success)]/10"><CheckCircle2 size={17} /></button>
              <button onClick={onReject} className="rounded-md p-1.5 text-[var(--app-danger)] hover:bg-[var(--app-danger)]/10"><XCircle size={17} /></button>
            </>}
            {editing && <>
              <button onClick={onSave} className="rounded-md p-1.5 text-[var(--app-success)] hover:bg-[var(--app-success)]/10"><Check size={17} /></button>
              <button onClick={onCancel} className="rounded-md p-1.5 text-[var(--app-danger)] hover:bg-[var(--app-danger)]/10"><X size={17} /></button>
            </>}
            {job.userStatus !== 'pending' && !editing && job.status !== 'applied' && <button onClick={onResetStatus} className="rounded-md border border-[var(--app-border)] px-2 py-1 text-[10px] font-bold">{job.userStatus}</button>}
          </div>
        </div>

        {job.result && <div className="grid gap-3 md:grid-cols-4">
          <Meta label="Client" value={NamingService.toLegalName(job.result.client) || '?'} />
          <Meta label="Case" value={NamingService.toStandardName(job.result.case) || '?'} />
          <Meta label="Line" value={NamingService.toStandardName(job.result.line) || '?'} />
          <Meta label="Appeal" value={NamingService.toStandardName(job.result.appeal) || '?'} />
        </div>}

        {editing && values && <JobEditForm values={values} setValues={setValues} options={options} />}
        {job.error && <UiText size="xs" className="text-[var(--app-danger)]">{job.error}</UiText>}
      </div>
    </UiPanel>
  );
}

function Meta({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg bg-[var(--app-surface-soft)] px-3 py-2">
      <UiCaption>{label}</UiCaption>
      <UiText size="xs" weight="medium" className="font-mono">{value}</UiText>
    </div>
  );
}
