import { FileText, FolderSearch, Loader2, Upload } from 'lucide-react';
import { AppButton, UiPanel, UiText } from '../../../components/ui';
import { JobCard } from './JobCard';

export function SystematizerWorkspace({
  logic,
  editing,
  selection,
}: {
  logic: any;
  editing: any;
  selection: any;
}) {
  return (
    <div className="mx-auto flex w-full max-w-5xl flex-col gap-6 py-6">
      <div className={`relative flex flex-col items-center justify-center gap-4 rounded-2xl border-2 border-dashed p-12 transition-all ${logic.isDragging ? 'border-[var(--app-accent)] bg-[var(--app-accent)]/5' : 'border-black/10 bg-gray-50/50 hover:border-black/20 hover:bg-white'}`} onDragOver={logic.handleDragOver} onDragLeave={logic.handleDragLeave} onDrop={logic.handleDrop}>
        <div className="flex h-16 w-16 items-center justify-center rounded-2xl border border-black/[0.03] bg-white text-[var(--app-accent)] shadow-sm"><Upload size={32} /></div>
        <div className="text-center"><h3 className="text-[length:var(--text-lg)] font-bold text-black/80">{logic.title}</h3><p className="mt-1 text-[length:var(--text-sm)] text-black/40">{logic.subtitle}</p></div>
        <div className="flex items-center gap-3">
          <AppButton variant="secondary" icon={<FileText size={16} />} onClick={logic.pickFiles} disabled={logic.isProcessing}>{logic.pickFilesLabel}</AppButton>
          <AppButton variant="secondary" icon={<FolderSearch size={16} />} onClick={logic.pickFolder} disabled={logic.isProcessing}>{logic.pickFolderLabel}</AppButton>
        </div>
      </div>

      {logic.isProcessing && <UiPanel className="border-[var(--app-accent)]/20 bg-[var(--app-accent)]/5 p-6"><div className="flex items-center gap-3"><Loader2 className="animate-spin text-[var(--app-accent)]" size={20} /><div className="flex-1"><UiText weight="bold">{logic.processingLabel}</UiText><UiText size="xs" tone="muted">{logic.currentFile || logic.scanningLabel}</UiText></div><UiText size="xs" weight="bold">{logic.progress}%</UiText></div><div className="mt-4 h-1.5 overflow-hidden rounded-full bg-[var(--app-border)]"><div className="h-full bg-[var(--app-accent)] transition-all duration-300" style={{ width: `${logic.progress}%` }} /></div></UiPanel>}

      {logic.suggestions.length > 0 && <div className="space-y-4">
        <div className="flex items-center justify-between px-2">
          <UiText weight="bold" className="label-caps">{logic.resultsLabel} ({logic.suggestions.length})</UiText>
          <div className="flex items-center gap-2">
            <label className="flex items-center gap-2 text-[length:var(--text-xs)]"><input type="checkbox" checked={selection.allSelected} onChange={selection.toggleAll} /> {selection.selectedIds.length}</label>
            <AppButton size="sm" variant="ghost" onClick={logic.clearAll}>{logic.clearLabel}</AppButton>
            <AppButton size="sm" variant="secondary" onClick={logic.approveAll}>{logic.approveAllLabel}</AppButton>
            <AppButton size="sm" variant="primary" onClick={logic.applyAll}>{logic.applyAllLabel}</AppButton>
          </div>
        </div>
        <div className="grid gap-3">
          {logic.suggestions.map((job: any) => <JobCard key={job.id} job={job} checked={selection.isSelected(job.id)} editing={editing.editingId === job.id} values={editing.values} options={editing.options} setValues={editing.setValues} onToggle={() => selection.toggle(job.id)} onEdit={() => editing.startEditing(job)} onApprove={() => logic.approve(job.id)} onReject={() => logic.reject(job.id)} onCancel={editing.cancelEditing} onSave={() => editing.saveEditing(job.id, logic.updateJobResult)} onResetStatus={() => logic.updateUserStatus(job.id, 'pending')} />)}
        </div>
      </div>}
    </div>
  );
}
